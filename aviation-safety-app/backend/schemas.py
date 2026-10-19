"""Pydantic schemas for the Safety NLP API."""
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


# Enums
class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ServiceName(str, Enum):
    PIPELINE = "pipeline"
    MONITOR = "monitor"


class ServiceAction(str, Enum):
    START = "start"
    STOP = "stop"
    RESTART = "restart"


class PipelineStage(str, Enum):
    FETCH = "fetch"
    PREPROCESS = "preprocess"
    TRAIN = "train"
    EVALUATE = "evaluate"
    RAG = "rag"
    REPORT = "report"


# Request schemas
class ClassifyRequest(BaseModel):
    narrative: str = Field(..., min_length=10, max_length=10000, description="Incident narrative text")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of evidence items to retrieve")


class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Analysis query")
    model: Optional[str] = Field(default=None, description="Ollama model to use (optional)")
    use_llm: bool = Field(default=True, description="Prefer Ollama LLM when connected, fall back to rule-based analyst")


class MonitorControlRequest(BaseModel):
    action: ServiceAction
    poll_seconds: Optional[int] = Field(default=60, ge=10, le=3600)
    enable_api: bool = True


class PipelineRunRequest(BaseModel):
    force_refresh: bool = False
    no_fetch: bool = False
    no_rag: bool = False
    samples: int = Field(default=100, ge=10, le=1000)
    stages: Optional[List[PipelineStage]] = None  # If None, run all stages


class FetchDataRequest(BaseModel):
    force_refresh: bool = False
    nrows_aviation: int = Field(default=2000, ge=100, le=10000)


class TrainModelRequest(BaseModel):
    max_features: int = Field(default=5000, ge=1000, le=20000)
    ngram_range: List[int] = Field(default=[1, 2])
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    cv_folds: int = Field(default=3, ge=2, le=10)
    alphas: List[float] = Field(default=[1e-5, 1e-4, 1e-3])
    class_weights: List[Optional[str]] = Field(default=[None, "balanced"])
    max_iter: int = Field(default=1000, ge=100, le=5000)


class EvaluateRequest(BaseModel):
    plot_top_n: int = Field(default=10, ge=5, le=20)


class RAGRequest(BaseModel):
    n_samples: int = Field(default=100, ge=10, le=1000)
    top_k: int = Field(default=3, ge=1, le=10)


# Response schemas
class EvidenceItem(BaseModel):
    rank: int
    similarity: float
    label: str
    domain: str
    snippet: str


class ClassifyResponse(BaseModel):
    predicted_label: str
    evidence: List[EvidenceItem]
    processing_time_ms: float


class DatasetStats(BaseModel):
    total_reports: int
    domains: Dict[str, int]
    anomaly_classes: int
    class_distribution: Dict[str, int]
    narrative_stats: Dict[str, float]


class ModelMetrics(BaseModel):
    accuracy: float
    n_classes: int
    test_size: int
    best_cv_f1: Optional[float] = None
    training_config: Optional[Dict[str, Any]] = None
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None


class ClassificationReportRow(BaseModel):
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int


class ModelPerformanceResponse(BaseModel):
    metrics: ModelMetrics
    classification_report: List[ClassificationReportRow]
    confusion_matrix_url: Optional[str] = None
    class_distribution_url: Optional[str] = None


class DataAssistantResponse(BaseModel):
    lines: List[str]
    reply: str = ""
    used_llm: bool = False
    model: Optional[str] = None
    ollama_connected: bool = False


class OllamaStatusResponse(BaseModel):
    connected: bool
    base_url: str
    model: str
    models: List[str]
    default_model: str


class AlertItem(BaseModel):
    timestamp: str
    incident_id: str
    source: str
    risk_level: RiskLevel
    predicted_label: str
    narrative: str
    suggestion: str = ""
    evidence: List[EvidenceItem]


class AlertsResponse(BaseModel):
    counts: Dict[str, int]
    alerts: List[AlertItem]
    total: int


class ServiceStatus(BaseModel):
    name: str
    label: str
    description: str
    running: bool
    pid: Optional[int] = None


class ProcessLogEntry(BaseModel):
    service: str
    line: str
    timestamp: str


class SystemControlResponse(BaseModel):
    services: List[ServiceStatus]
    logs: Dict[str, List[str]]
    model_loaded: bool = False
    data_loaded: bool = False


class PipelineStageProgress(BaseModel):
    stage: PipelineStage
    status: Literal["pending", "running", "completed", "failed"]
    progress: float  # 0-100
    message: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class PipelineRunResponse(BaseModel):
    success: bool
    message: str
    duration_seconds: Optional[float] = None
    stages: List[PipelineStageProgress] = []
    artifacts: Optional[Dict[str, str]] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    data_loaded: bool
    monitor_running: bool
    ollama_connected: bool = False


class LoadArtifactsResponse(BaseModel):
    success: bool
    message: str
    model_loaded: bool
    data_loaded: bool


class FetchResult(BaseModel):
    success: bool
    message: str
    aviation_count: int
    power_grid_count: int
    total_count: int
    duration_seconds: float


class TrainResult(BaseModel):
    success: bool
    message: str
    accuracy: float
    n_classes: int
    best_params: Dict[str, Any]
    duration_seconds: float