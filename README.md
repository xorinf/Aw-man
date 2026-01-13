# SENTINEL - Autonomous AI Cyber Defense Platform

<div align="center">

![SENTINEL](https://img.shields.io/badge/SENTINEL-AI%20Cyber%20Defense-7c3aed?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-1.0.0--alpha-orange?style=for-the-badge)

**Predict. Detect. Explain. Defend.**

*An AI-first security platform that detects unknown attacks, predicts attacker behavior, and explains threats using explainable AI.*

[📖 Documentation](#-quick-start) · [🚀 Quick Start](#-quick-start) · [📊 API Docs](#-api-endpoints) · [🗺️ Roadmap](#-development-roadmap)

</div>

---

## 📋 Current Development Status

> **Version: 1.0.0-alpha** | **Status: Active Development** | **API: ✅ Working**

| Component | Status | Description |
|-----------|--------|-------------|
| 🟢 FastAPI Server | **Complete** | REST API with Swagger docs |
| 🟢 Feature Extraction | **Complete** | Network flow & system log processing |
| 🟢 Anomaly Detection | **Complete** | VAE + Isolation Forest hybrid |
| 🟢 Behavior Prediction | **Complete** | LSTM & Transformer models |
| 🟢 XAI Explainer | **Complete** | SHAP, counterfactuals, attention |
| 🟢 Red Team Simulation | **Complete** | APT29 & Opportunistic agents |
| 🟡 Model Training | **Pending** | Requires dataset integration |
| 🟢 React Dashboard | **Complete** | Real-time UI with WebSocket alerts |
| 🔴 Production Deploy | **Not Started** | Kubernetes deployment |

---

## 🎯 Features

- **🔍 Zero-Day Detection** - Unsupervised ML detects unknown attacks without signatures
- **🔮 Behavior Prediction** - LSTM/Transformer predicts attacker next moves in kill chain
- **🕸️ Lateral Movement Detection** - Graph Neural Networks track attack paths
- **💡 Explainable AI** - SHAP/LIME explanations for every alert (why it was flagged)
- **👹 AI Red Team** - Multi-agent attack simulation (APT29, Opportunistic personas)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         SENTINEL                                 │
├─────────────────────────────────────────────────────────────────┤
│  DATA INGESTION  →  FEATURE EXTRACTION  →  AI CORE  →  XAI     │
│  (Kafka/Zeek)       (NetworkFlow)           ↓          ↓       │
│                                        ┌────┴────┐  ┌──┴──┐    │
│                                        │ Anomaly │  │SHAP │    │
│                                        │ (VAE)   │  │LIME │    │
│                                        ├─────────┤  └─────┘    │
│                                        │Sequence │     ↓       │
│                                        │ (LSTM)  │  RESPONSE   │
│                                        └─────────┘  (FastAPI)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
sentinel/
├── src/
│   ├── api/main.py           # FastAPI endpoints ✅
│   ├── config.py             # Settings & environment ✅
│   ├── pipeline/
│   │   └── feature_extractor.py    # Network/log feature extraction ✅
│   ├── models/
│   │   ├── anomaly/vae_detector.py      # VAE + Isolation Forest ✅
│   │   └── sequence/behavior_predictor.py # LSTM/Transformer ✅
│   ├── xai/explainer.py      # SHAP, counterfactuals ✅
│   └── redteam/attacker_agent.py  # APT simulation ✅
├── tests/                    # Unit & integration tests
├── data/                     # Datasets (add CICIDS2017 here)
├── configs/                  # Configuration files
├── dashboard/                # React frontend (coming soon)
├── docker-compose.yml        # Container orchestration
├── Dockerfile                # API container
├── pyproject.toml            # Python package config
├── requirements.txt          # Dependencies
└── LICENSE                   # MIT License
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/xorinf/Aw-man.git
cd Aw-man

# Install dependencies
pip install -r requirements.txt

# Run the API server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Run the Dashboard (in a separate terminal)
cd dashboard
npm install
npm run dev
```

### Access Points
| Service | URL |
|---------|-----|
| 🎨 Dashboard | http://localhost:3000 |
| 📚 Swagger Docs | http://localhost:8000/docs |
| 💚 Health Check | http://localhost:8000/health |
| 🎯 OpenAPI Schema | http://localhost:8000/openapi.json |
| 🔌 WebSocket | ws://localhost:8000/ws/alerts |

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check - returns status, version, models loaded |
| `/analyze` | POST | Analyze network flows for threats |
| `/simulate-attack` | POST | Generate simulated attack traffic (APT, Opportunistic) |
| `/mitre-coverage` | GET | Get MITRE ATT&CK coverage statistics |
| `/attack-graph` | GET | Get attack graph visualization data |
| `/mitre/technique/{id}` | GET | Get MITRE technique details |
| `/chat` | POST | AI Security Copilot conversation |
| `/ws/alerts` | WebSocket | Real-time alert streaming |

### Example: Simulate an APT Attack
```bash
curl -X POST "http://localhost:8000/simulate-attack?attack_type=apt&num_actions=5"
```

Response:
```json
{
  "campaign_id": "APT-1234",
  "persona": "APT29_Cozy_Bear",
  "actions": [
    {"stage": "reconnaissance", "technique": "Active Scanning", "mitre_id": "T1595"},
    {"stage": "initial_access", "technique": "Phishing", "mitre_id": "T1566"},
    ...
  ]
}
```

---

## 🧠 AI Models

| Model | Purpose | Architecture |
|-------|---------|--------------|
| **HybridAnomalyDetector** | Zero-day detection | VAE + Isolation Forest |
| **BehaviorPredictor** | Attack sequence prediction | LSTM with Attention / Transformer |
| **ThreatExplainerPipeline** | Alert explanation | SHAP + Counterfactuals |
| **RedTeamSimulator** | Attack generation | Rule-based + LLM agents |

---

## 🗺️ Development Roadmap

### ✅ v1.0.0-alpha (Current)
- [x] Core API with FastAPI
- [x] Feature extraction pipeline
- [x] Anomaly detection models (VAE + Isolation Forest)
- [x] Sequence models (LSTM, Transformer)
- [x] XAI module (SHAP, counterfactuals)
- [x] AI Red Team simulation
- [x] Docker configuration

### 🔜 v1.1.0-beta (Next)
- [ ] Train models on CICIDS2017 dataset
- [ ] Add model persistence (save/load trained models)
- [ ] WebSocket streaming for real-time alerts
- [ ] Basic React dashboard

### 📋 v2.0.0 (Future)
- [ ] Graph Neural Network for lateral movement
- [ ] LLM-powered threat intelligence
- [ ] Kubernetes deployment
- [ ] Federated learning support

---

## 🔬 Datasets

For training, download and place in `data/` folder:

| Dataset | Use Case | Link |
|---------|----------|------|
| CICIDS2017 | Network intrusion detection | [Download](https://www.unb.ca/cic/datasets/ids-2017.html) |
| UNSW-NB15 | Modern attack types | [Download](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| Mordor | ATT&CK-mapped attacks | [GitHub](https://github.com/OTRF/mordor) |

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📜 License

MIT License - See [LICENSE](LICENSE)

---

<div align="center">

**Built with 💜 by [xorinf](https://github.com/xorinf)**

⭐ Star this repo if you find it useful!

</div>
