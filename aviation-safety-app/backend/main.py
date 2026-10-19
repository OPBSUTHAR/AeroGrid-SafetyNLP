"""FastAPI application for Safety NLP Pipeline - Full Web Control."""
import logging
import subprocess
import sys
import threading
import time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from queue import Queue

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings
from schemas import (
    ClassifyRequest, ClassifyResponse, AnalyzeRequest, DataAssistantResponse,
    DatasetStats, ModelPerformanceResponse, AlertsResponse,
    SystemControlResponse, ServiceStatus, PipelineRunRequest, PipelineRunResponse,
    HealthResponse, MonitorControlRequest, FetchDataRequest, FetchResult,
    TrainModelRequest, TrainResult, PipelineStage, PipelineStageProgress,
    ServiceName, ServiceAction, RiskLevel, EvidenceItem, OllamaStatusResponse,
)
from ml_service import ml_service, PipelineProgress
from ollama_service import ollama_service
from suggestions import build_alerts_summary

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

def sanitize_system_text(value: str) -> str:
    if not value:
        return value
    sanitized = value.replace("\\", "/")
    sanitized = sanitized.replace("C:/", "")
    sanitized = sanitized.replace("/Users/", "[user]/")
    sanitized = sanitized.replace("/home/", "[home]/")
    sanitized = sanitized.replace("/tmp/", "[tmp]/")
    sanitized = sanitized.replace("/var/", "[var]/")
    sanitized = sanitized.replace("/opt/", "[opt]/")
    sanitized = sanitized.replace("/Program Files/", "[program-files]/")
    sanitized = sanitized.replace("/workspace/", "[workspace]/")
    sanitized = sanitized.replace("/vs code/", "[workspace]/")
    return sanitized

# Process management
class ProcessManager:
    SERVICES = {
        "pipeline": {
            "cmd": [sys.executable, "main.py"],
            "label": "Pipeline",
            "desc": "Full pipeline: fetch → preprocess → train → evaluate → RAG → report",
        },
        "monitor": {
            "cmd": [sys.executable, "-m", "src.monitor"],
            "label": "Monitor",
            "desc": "Real-time incident ingestion, classification, risk scoring & alerting",
        },
    }
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_buffers: Dict[str, List[str]] = {name: [] for name in self.SERVICES}
        self.log_queues: Dict[str, Queue] = {name: Queue() for name in self.SERVICES}
        self.log_threads: Dict[str, threading.Thread] = {}
        self.lock = threading.Lock()
    
    def _read_stream(self, stream, service_name: str):
        try:
            for line in iter(stream.readline, ''):
                if line:
                    with self.lock:
                        raw_line = line.rstrip()
                        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {sanitize_system_text(raw_line)}"
                        self.log_buffers[service_name].append(entry)
                        self.log_queues[service_name].put(entry)
                        # Keep buffer bounded
                        if len(self.log_buffers[service_name]) > 1000:
                            self.log_buffers[service_name] = self.log_buffers[service_name][-1000:]
        except Exception:
            pass
        finally:
            stream.close()
    
    def start(self, service_name: str) -> tuple[bool, str]:
        with self.lock:
            if service_name in self.processes and self.processes[service_name].poll() is None:
                return False, f"{service_name} already running"
            
            svc = self.SERVICES[service_name]
            cwd = settings.BASE_DIR
            
            try:
                proc = subprocess.Popen(
                    svc["cmd"],
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception as e:
                return False, f"Failed to start {service_name}: {e}"
            
            self.processes[service_name] = proc
            
            # Start log reader thread
            t = threading.Thread(
                target=self._read_stream,
                args=(proc.stdout, service_name),
                daemon=True,
            )
            t.start()
            self.log_threads[service_name] = t
            
            return True, f"{svc['label']} started (PID: {proc.pid})"
    
    def stop(self, service_name: str) -> tuple[bool, str]:
        with self.lock:
            if service_name not in self.processes:
                return False, f"{service_name} not running"
            
            proc = self.processes[service_name]
            if proc.poll() is not None:
                self._cleanup(service_name)
                return True, f"{service_name} was already stopped"
            
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            except Exception as e:
                return False, f"Error stopping {service_name}: {e}"
            finally:
                self._cleanup(service_name)
            
            return True, f"{self.SERVICES[service_name]['label']} stopped"
    
    def _cleanup(self, service_name: str):
        self.processes.pop(service_name, None)
        self.log_threads.pop(service_name, None)
    
    def is_running(self, service_name: str) -> bool:
        with self.lock:
            proc = self.processes.get(service_name)
            return proc is not None and proc.poll() is None
    
    def get_logs(self, service_name: str, since: Optional[int] = None) -> List[str]:
        with self.lock:
            logs = self.log_buffers.get(service_name, [])[:]
            if since is not None:
                # Return last N entries
                return logs[-since:]
            return logs
    
    def get_all_status(self) -> Dict[str, bool]:
        return {name: self.is_running(name) for name in self.SERVICES}
    
    def stop_all(self):
        for name in list(self.processes.keys()):
            self.stop(name)


proc_mgr = ProcessManager()

# WebSocket connections for real-time logs
active_websockets: Dict[str, List[WebSocket]] = {"pipeline": [], "monitor": []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Safety NLP API...")
    # Load ML artifacts in background
    load_thread = threading.Thread(target=ml_service.load_artifacts, daemon=True)
    load_thread.start()
    yield
    # Shutdown
    logger.info("Shutting down...")
    proc_mgr.stop_all()
    # Close all websockets
    for ws_list in active_websockets.values():
        for ws in ws_list:
            try:
                await ws.close()
            except:
                pass


app = FastAPI(
    title="Safety NLP Pipeline API",
    description="Aviation & Power-Grid Incident Analysis API with TF-IDF + SGD + RAG Explainability - Full Web Control",
    version="2.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for plots
if settings.PLOTS_DIR.exists():
    app.mount("/api/plots", StaticFiles(directory=settings.PLOTS_DIR), name="plots")


# ============================================================
# HEALTH & STATUS
# ============================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if ml_service.is_data_ready() else "loading",
        version="2.1.0",
        model_loaded=ml_service._model is not None,
        data_loaded=ml_service._df is not None,
        monitor_running=proc_mgr.is_running("monitor"),
        ollama_connected=ollama_service.get_status().connected,
    )

# Load ML Artifacts endpoint
class LoadArtifactsResponse(BaseModel):
    success: bool
    message: str
    model_loaded: bool
    data_loaded: bool

@app.post("/api/ml/load-artifacts", response_model=LoadArtifactsResponse)
async def load_ml_artifacts():
    """Manually load ML artifacts (model, vectorizer, dataset, RAG index)"""
    try:
        success = ml_service.load_artifacts()
        return LoadArtifactsResponse(
            success=success,
            message="ML artifacts loaded successfully" if success else "Failed to load ML artifacts - check if pipeline has been run",
            model_loaded=ml_service._model is not None,
            data_loaded=ml_service._df is not None,
        )
    except Exception as e:
        return LoadArtifactsResponse(
            success=False,
            message=f"Error loading artifacts: {str(e)}",
            model_loaded=False,
            data_loaded=False,
        )


# ============================================================
# DATASET OVERVIEW
# ============================================================

@app.get("/api/overview", response_model=DatasetStats)
async def get_overview():
    if not ml_service.is_data_ready():
        raise HTTPException(status_code=503, detail="ML artifacts not loaded yet - retrying")
    stats = ml_service.get_dataset_stats()
    return DatasetStats(**stats)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@app.get("/api/model-performance", response_model=ModelPerformanceResponse)
async def get_model_performance():
    if not ml_service.is_data_ready():
        raise HTTPException(status_code=503, detail="ML artifacts not loaded yet - retrying")
    perf = ml_service.get_model_performance()
    if not perf:
        raise HTTPException(status_code=404, detail="Performance data not available. Run pipeline first.")
    return ModelPerformanceResponse(**perf)


# ============================================================
# RAG CLASSIFICATION
# ============================================================

@app.post("/api/classify", response_model=ClassifyResponse)
async def classify_incident(request: ClassifyRequest):
    if not ml_service.is_ready():
        raise HTTPException(status_code=503, detail="ML artifacts not loaded yet")
    
    try:
        predicted, evidence, processing_time = ml_service.classify_with_evidence(
            request.narrative, request.top_k
        )
        return ClassifyResponse(
            predicted_label=predicted,
            evidence=[EvidenceItem(**e) for e in evidence],
            processing_time_ms=processing_time
        )
    except Exception as e:
        logger.error("Classification failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# DATA ASSISTANT
# ============================================================

@app.post("/api/analyze", response_model=DataAssistantResponse)
async def analyze_data(request: AnalyzeRequest):
    """Data Assistant: Ollama LLM (domain-constrained) with rule-based fallback."""
    if not ml_service.is_data_ready():
        raise HTTPException(status_code=503, detail="ML artifacts not loaded yet - retrying")
    
    # 1) Try the local Ollama LLM when enabled and connected
    if request.use_llm:
        ollama_status = ollama_service.get_status()
        if ollama_status.connected:
            summary = ml_service.get_dataset_stats()
            alerts_data = ml_service.get_alerts(limit=20)
            context = ollama_service.build_context(
                dataset_summary=summary,
                domain_text=build_alerts_summary(alerts_data.get("alerts", []), alerts_data.get("total", 0)),
            )
            handled, reply = ollama_service.chat(request.query, model=request.model, context=context)
            if handled and reply:
                return DataAssistantResponse(
                    lines=reply.split("\n"),
                    reply=reply,
                    used_llm=True,
                    model=request.model or ollama_status.model,
                    ollama_connected=True,
                )
    
    # 2) Fallback: keyless rule-based analyst (always available)
    lines = ml_service.analyze_data(request.query)
    return DataAssistantResponse(
        lines=lines,
        reply="\n".join(lines),
        used_llm=False,
        model=None,
        ollama_connected=ollama_service.get_status().connected,
    )


@app.get("/api/ollama/status", response_model=OllamaStatusResponse)
async def ollama_status(force: bool = Query(False, description="Force a refresh of the connection check")):
    """Check the local Ollama server and list live models."""
    status = ollama_service.get_status(force=force)
    return OllamaStatusResponse(**status.to_dict())


# ============================================================
# LIVE ALERTS
# ============================================================

@app.get("/api/alerts", response_model=AlertsResponse)
async def get_alerts(limit: int = Query(50, ge=1, le=500)):
    data = ml_service.get_alerts(limit)
    return AlertsResponse(**data)


# ============================================================
# SYSTEM CONTROL
# ============================================================

@app.get("/api/system/status", response_model=SystemControlResponse)
async def get_system_status():
    status = proc_mgr.get_all_status()
    services = []
    for name, info in ProcessManager.SERVICES.items():
        running = status[name]
        pid = None
        if running and name in proc_mgr.processes:
            pid = proc_mgr.processes[name].pid
        services.append(ServiceStatus(
            name=name,
            label=info["label"],
            description=info["desc"],
            running=running,
            pid=pid
        ))
    
    logs = {}
    for name in ProcessManager.SERVICES:
        logs[name] = proc_mgr.get_logs(name, since=200)
    
    return SystemControlResponse(
        services=services,
        logs=logs,
        model_loaded=ml_service._model is not None,
        data_loaded=ml_service._df is not None,
    )


@app.post("/api/system/control/{service_name}/{action}")
async def control_service(service_name: ServiceName, action: ServiceAction):
    if action == ServiceAction.START:
        ok, msg = proc_mgr.start(service_name.value)
    elif action == ServiceAction.STOP:
        ok, msg = proc_mgr.stop(service_name.value)
    elif action == ServiceAction.RESTART:
        proc_mgr.stop(service_name.value)
        await asyncio.sleep(1)
        ok, msg = proc_mgr.start(service_name.value)
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {action}")
    
    return {"success": ok, "message": msg}


@app.post("/api/system/control/all/{action}")
async def control_all_services(action: ServiceAction):
    if action == ServiceAction.START:
        for name in ProcessManager.SERVICES:
            if not proc_mgr.is_running(name):
                proc_mgr.start(name)
        return {"success": True, "message": "All services started"}
    elif action == ServiceAction.STOP:
        proc_mgr.stop_all()
        return {"success": True, "message": "All services stopped"}
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {action}")


# ============================================================
# PIPELINE EXECUTION (Full Web Control)
# ============================================================

@app.post("/api/pipeline/run", response_model=PipelineRunResponse)
async def run_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    if ml_service.is_pipeline_running():
        raise HTTPException(status_code=409, detail="Pipeline already running")
    
    progress = PipelineProgress()
    
    def run_pipeline_task():
        result = ml_service.run_pipeline_stages(
            request.model_dump(),
            progress
        )
        return result
    
    # Run in background
    background_tasks.add_task(run_pipeline_task)
    
    # Return initial response with pending stages
    stages = [
        PipelineStageProgress(stage=PipelineStage.FETCH, status="pending", progress=0, message="Queued"),
        PipelineStageProgress(stage=PipelineStage.PREPROCESS, status="pending", progress=0, message="Queued"),
        PipelineStageProgress(stage=PipelineStage.TRAIN, status="pending", progress=0, message="Queued"),
        PipelineStageProgress(stage=PipelineStage.EVALUATE, status="pending", progress=0, message="Queued"),
        PipelineStageProgress(stage=PipelineStage.RAG, status="pending", progress=0, message="Queued"),
        PipelineStageProgress(stage=PipelineStage.REPORT, status="pending", progress=0, message="Queued"),
    ]
    
    return PipelineRunResponse(
        success=True,
        message="Pipeline started",
        duration_seconds=None,
        stages=stages,
    )


@app.get("/api/pipeline/progress", response_model=List[PipelineStageProgress])
async def get_pipeline_progress():
    progress = ml_service.get_pipeline_progress()
    return progress.to_list()


@app.post("/api/pipeline/fetch", response_model=FetchResult)
async def fetch_data(request: FetchDataRequest):
    result = ml_service.fetch_data_only(request.force_refresh, request.nrows_aviation)
    return FetchResult(**result)


@app.post("/api/pipeline/train", response_model=TrainResult)
async def train_model(request: TrainModelRequest):
    # Note: This uses the config from config.py, not the request params
    # For full param control, would need to modify trainer
    result = ml_service.train_model_only(request.model_dump())
    return TrainResult(**result)


# ============================================================
# MONITOR CONTROL
# ============================================================

@app.post("/api/monitor/control")
async def control_monitor(request: MonitorControlRequest):
    if request.action == ServiceAction.START:
        ok, msg = proc_mgr.start("monitor")
    elif request.action == ServiceAction.STOP:
        ok, msg = proc_mgr.stop("monitor")
    elif request.action == ServiceAction.RESTART:
        proc_mgr.stop("monitor")
        await asyncio.sleep(1)
        ok, msg = proc_mgr.start("monitor")
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
    
    return {"success": ok, "message": msg, "poll_seconds": request.poll_seconds}


# ============================================================
# WEBSOCKET FOR REAL-TIME LOGS
# ============================================================

@app.websocket("/api/ws/logs/{service_name}")
async def websocket_logs(websocket: WebSocket, service_name: str):
    if service_name not in ProcessManager.SERVICES:
        await websocket.close(code=4004, reason="Invalid service")
        return
    
    await websocket.accept()
    active_websockets[service_name].append(websocket)
    
    # Send initial logs
    logs = proc_mgr.get_logs(service_name, since=100)
    await websocket.send_json({"type": "initial", "logs": logs})
    
    try:
        # Stream new logs from queue
        queue = proc_mgr.log_queues[service_name]
        while True:
            try:
                # Non-blocking check
                if not queue.empty():
                    log_entry = queue.get_nowait()
                    await websocket.send_json({
                        "type": "log",
                        "service": service_name,
                        "entry": log_entry
                    })
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("WebSocket error: %s", e)
                break
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in active_websockets[service_name]:
            active_websockets[service_name].remove(websocket)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )