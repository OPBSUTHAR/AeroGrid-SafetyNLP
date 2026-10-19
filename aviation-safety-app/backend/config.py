"""Backend configuration for Safety NLP API."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # CORS - React dev server
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # Data paths - relative to project root
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent / "safety_nlp_pipeline"
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PLOTS_DIR: Path = DATA_DIR / "plots"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    WATCH_DIR: Path = BASE_DIR / "new_incidents"
    MODEL_PATH: Path = DATA_DIR / "safety_model.pkl"
    VECTORIZER_PATH: Path = DATA_DIR / "tfidf_vectorizer.pkl"
    DATASET_PATH: Path = DATA_DIR / "real_safety_dataset.csv"
    ALERT_LOG_PATH: Path = DATA_DIR / "alerts.csv"
    METRICS_JSON: Path = DATA_DIR / "metrics.json"
    CLASSIFICATION_REPORT_CSV: Path = DATA_DIR / "classification_report.csv"
    CLASSIFICATION_REPORT_TXT: Path = DATA_DIR / "classification_report.txt"
    TRAIN_CONFIG_PATH: Path = DATA_DIR / "training_config.json"
    MONITOR_STATE_PATH: Path = DATA_DIR / "monitor_state.json"
    
    # Model settings
    RAG_TOP_K: int = 3
    RAG_BATCH: int = 200
    RAG_EVIDENCE_SNIPPET: int = 170
    MONITOR_ALERT_SNIPPET: int = 200
    MONITOR_POLL_SECONDS: int = 60
    
    # External APIs
    NTSB_API_URL: str = "https://api.ai-analytics.org/api/v1/ntsb/aviation/recent"
    NTSB_POLL_SECONDS: int = 3600
    UKPN_API_URL: str = "https://ukpowernetworks.opendatasoft.com/api/explore/v2.1/catalog/datasets/ukpn-live-faults/records"
    UKPN_POLL_SECONDS: int = 60
    ALERT_HIGH_MIN_CUSTOMERS: int = 100
    
    # Ollama (local LLM) integration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_STATUS_TTL: int = 10
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# Column aliases for dataset normalization (matching safety_nlp_pipeline/config.py)
COLUMN_ALIASES = {
    "Report 1_Narrative": "narrative",
    "Events_Anomaly": "label",
    "Narrative": "narrative",
    "human_factors_groundtruth": "label",
    "Domain": "domain",
    "narrative": "narrative",
    "label": "label",
    "domain": "domain",
}

# Canonical column names
NARRATIVE_COL = "narrative"
LABEL_COL = "label"
DOMAIN_COL = "domain"
PROCESSED_COL = "processed_text"

# Create directories (best-effort: serverless hosts such as Vercel mount the
# bundle read-only, so directory creation must never break the import)
for d in (settings.DATA_DIR, settings.RAW_DIR, settings.PLOTS_DIR,
          settings.REPORTS_DIR, settings.WATCH_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass