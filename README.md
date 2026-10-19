# Safety NLP Pipeline — Aviation & Power-Grid Incident Analysis

A production-grade NLP pipeline for classifying safety incident reports from **aviation (NASA ASRS)** and **power-grid (NERC)** domains, with **RAG explainability**, **real-time monitoring**, **local Ollama LLM integration**, and a **modern React + FastAPI web dashboard**.

---

## 🚀 Quick Start

### 1. Run the Core Pipeline (Generate Artifacts)
```bash
cd safety_nlp_pipeline
pip install -r requirements.txt
python main.py           # Full pipeline → generates models, reports, metrics
```

### 2. Launch the Web Dashboard
```bash
cd aviation-safety-app

# Terminal 1 - Backend API
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend
npm install
npm run dev
```

- **Dashboard**: http://localhost:5173
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 3. (Optional) Enable Local LLM for Data Assistant
```bash
ollama serve
ollama pull llama3   # or any model of your choice
```
The Data Assistant will automatically detect and use the local Ollama instance — no API key required.

---

## ✨ Dashboard Features

| Page | Visualizations | Controls |
|------|---------------|----------|
| **Overview** | Domain pie chart, class bar charts, metric cards | Refresh data |
| **Model Performance** | Per-class P/R/F1 bars, support charts, confusion matrix | View training config |
| **RAG Explorer** | Similarity progress bars, evidence cards, sample narratives | Top-K selector, classify button |
| **Data Assistant** | **Ollama LLM chat** (domain-constrained), model selector, history, fallback analyst | Natural language queries, quick analysis buttons |
| **Live Alerts** | **Critical/High/Medium/Low** tabs with counts, risk/source filters, **auto-generated safety suggestions**, evidence | Category tabs, source filter, expandable details |
| **System Control** | Pipeline stage progress bars, service status badges, live logs | Start/stop services, run pipeline with options |

---

## 🏗️ Project Structure

```
Aviation_NLP_Project/
├── aviation-safety-app/           # Modern Web Dashboard (React + FastAPI)
│   ├── backend/                   # FastAPI REST API
│   │   ├── main.py               # All API endpoints
│   │   ├── ml_service.py         # ML wrapper with progress tracking
│   │   ├── config.py             # Settings (incl. Ollama)
│   │   ├── schemas.py            # Pydantic models
│   │   ├── ollama_service.py     # Ollama client with domain chat
│   │   ├── suggestions.py        # Safety suggestion generator
│   │   └── requirements.txt
│   ├── frontend/                 # React + TypeScript + Vite
│   │   ├── src/
│   │   │   ├── components/
│   │   │   │   ├── charts/       # Recharts: Bar, Pie, Line, Progress
│   │   │   │   ├── layout/       # Sidebar, Header, Layout, StatusBar
│   │   │   │   └── ui/           # Card, Table, Badge, MetricCard, Input, Button
│   │   │   ├── pages/            # 6 dashboard pages
│   │   │   ├── hooks/            # useApi, useOllamaStatus hooks
│   │   │   ├── store/            # Zustand state
│   │   │   ├── api/              # Axios client
│   │   │   ├── types/            # TypeScript types
│   │   │   └── utils/            # Helpers
│   │   └── package.json
│   └── README.md
│
├── safety_nlp_pipeline/          # Core Python Pipeline (Headless)
│   ├── src/                      # Pipeline modules
│   │   ├── data_fetcher.py       # ASRS + NERC ingestion
│   │   ├── preprocessor.py       # NLTK preprocessing
│   │   ├── trainer.py            # TF-IDF + SGD + GridSearchCV
│   │   ├── evaluator.py          # Metrics, plots, reports
│   │   ├── rag_explainer.py      # Semantic evidence retrieval
│   │   ├── analyst.py            # Keyless data assistant
│   │   ├── monitor.py            # Real-time monitoring
│   │   └── report_generator.py   # HTML report generation
│   ├── main.py                   # CLI entry point
│   ├── config.py                 # All parameters
│   └── requirements.txt
│
├── PROJECT_OVERVIEW.txt          # Complete documentation
├── README.md                     # This file
└── .gitignore
```

---

## 🔬 Pipeline Details

### Data Sources (Live, Not Pre-baked)
| Domain | Source | Size |
|--------|--------|------|
| **Aviation** | NASA ASRS (Hugging Face datasets-server) | 2,000 reports with expert anomaly labels |
| **Power Grid** | NERC Event Analysis (public PDFs) | 12 reports → ~1,700 narrative chunks |

### ML Pipeline
1. **Fetch** → Live download ASRS + NERC, clean, merge by domain
2. **Preprocess** → NLTK tokenize → stopword removal → lemmatize
3. **Train** → TF-IDF (bigrams, 5000 features) + SGDClassifier (log-loss) with GridSearchCV
4. **Evaluate** → Classification report, confusion matrix, class distribution plots
5. **RAG** → Batch cosine similarity retrieval (top-3 evidence per prediction)
6. **Report** → Self-contained HTML with all results

### Real-Time Monitoring
- **Drop-in folder**: `new_incidents/*.csv` or `*.txt`
- **Master dataset**: Appended rows to `real_safety_dataset.csv`
- **NTSB API**: Daily US aviation accidents (probable cause)
- **UKPN Live Faults**: Minute-level UK power cuts (≥100 customers → high risk)

---

## 🎯 Results

- **Dataset**: ~3,700 cleaned reports (2,000 aviation + ~1,700 power-grid)
- **Model**: ~71% accuracy / ~70% weighted F1 over 28 classes
- **RAG**: Every prediction auditable with top-3 historical evidence
- **Cross-domain**: Shared TF-IDF vocabulary surfaces genuine similarities
- **Live Alerts**: Auto-generated safety suggestions per incident (NLP-based)

---

## 🛠️ Development

### Dashboard Development
```bash
cd aviation-safety-app/frontend
npm run dev          # Hot reload
npm run build        # Production build
npm run lint         # TypeScript check (eslint config needs update for ESLint 9)
```

### Backend Development
```bash
cd aviation-safety-app/backend
uvicorn main:app --reload
# API docs at http://localhost:8000/docs
```

---

## 📦 Requirements

### Core Pipeline
```
textual>=8.2
pandas>=2.2
scikit-learn>=1.5
datasets>=3.0
joblib>=1.4
nltk>=3.8
python-docx>=1.1
pypdf>=4.0
```

### Backend API
```
fastapi==0.115.0
uvicorn[standard]==0.34.0
pydantic==2.10.6
pydantic-settings==2.9.0
joblib==1.4.2
pandas==2.2.3
scikit-learn==1.6.1
nltk==3.9.1
httpx==0.28.1
websockets==13.1
```

### Frontend
```
react: ^18.3.1
react-dom: ^18.3.1
react-router-dom: ^6.26.2
axios: ^1.7.7
recharts: ^2.12.7
lucide-react: ^0.441.0
zustand: ^5.0.0
tailwindcss: ^3.4.13
vite: ^5.4.8
typescript: ^5.6.2
```

---

## 📄 Documentation

| File | Description |
|------|-------------|
| `PROJECT_OVERVIEW.txt` | Complete project documentation |
| `aviation-safety-app/README.md` | Dashboard-specific documentation |
| `safety_nlp_pipeline/README.md` | Core pipeline documentation |

---

## ☁️ Live Demo

Deployed on Vercel: **https://aerogrid-safetynlp.vercel.app**

- The React + Vite frontend is served from the CDN; the FastAPI backend runs as a Vercel
  Python function (entrypoint [`index.py`](index.py)) with the trained model, dataset,
  plots and NLTK corpora bundled in `aviation-safety-app/backend/{artifacts,nltk_data}`.
- The serverless deployment has no local Ollama instance, so the Data Assistant
  automatically falls back to its built-in rule-based analyst (`ollama_connected: false`).
- Pipeline execution and monitoring controls (System page) are intended for local use.

## 👥 Authors & Contributors

This project is a **joint development by [@OPBSUTHAR](https://github.com/OPBSUTHAR) and [@Hamilwt](https://github.com/Hamilwt)** — both are co-authors of this work. This public copy preserves the full joint commit history (originally hosted as **[Hamilwt/Aviation_NLP_Project](https://github.com/Hamilwt/Aviation_NLP_Project)**) so that the shared research and engineering contributions remain permanently available.

| Co-Author | Key contributions |
|-----------|-------------------|
| **[@OPBSUTHAR](https://github.com/OPBSUTHAR)** — OMPRAKASH SUTHAR | Menu-driven TUI & complete aviation NLP pipeline, refactor to power-grid safety data, research document/report generation, `safety_nlp_pipeline` core, Streamlit dashboard, real-time monitoring & alerts, React + FastAPI web dashboard (all 6 pages), Ollama LLM data assistant, one-click launcher scripts |
| **[@Hamilwt](https://github.com/Hamilwt)** — Mohammed Hamil P R | Project inception & workspace setup, NASA ASRS data ingestion, domain preprocessing + model training, RAG semantic retrieval & explainable classification |

Both co-authors are credited in the [contributors graph](https://github.com/OPBSUTHAR/AeroGrid-SafetyNLP/graphs/contributors) — the complete commit history of the joint work is preserved here.

---

## 📄 License

MIT License