# SENTINEL - Autonomous AI Cyber Defense Platform

<div align="center">

![SENTINEL](https://img.shields.io/badge/SENTINEL-AI%20Cyber%20Defense-7c3aed?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Predict. Detect. Explain. Defend.**

*An AI-first security platform that detects unknown attacks, predicts attacker behavior, and explains threats using explainable AI.*

</div>

---

## 🎯 Features

- **Zero-Day Detection** - Unsupervised ML detects attacks without signatures
- **Behavior Prediction** - LSTM/Transformer predicts attacker next moves
- **Lateral Movement Detection** - Graph Neural Networks track attack paths
- **Explainable AI** - SHAP/LIME explanations for every alert
- **AI Red Team** - LLM-powered attacker simulation for adversarial training

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  DATA INGESTION  →  AI CORE  →  XAI  →  RESPONSE           │
│  (Kafka/Zeek)       (VAE/LSTM/GNN)  (SHAP)  (Dashboard)    │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
sentinel/
├── src/
│   ├── ingestion/      # Data ingestion (Kafka, Zeek parsers)
│   ├── pipeline/       # Feature extraction & preprocessing
│   ├── models/
│   │   ├── anomaly/    # VAE, Isolation Forest
│   │   ├── sequence/   # LSTM, Transformer
│   │   └── graph/      # GNN for lateral movement
│   ├── xai/            # Explainable AI (SHAP, LIME)
│   ├── api/            # FastAPI endpoints
│   ├── redteam/        # AI attacker simulation
│   └── utils/          # Helpers
├── tests/              # Unit & integration tests
├── data/               # Datasets & generated data
├── configs/            # Configuration files
├── dashboard/          # React frontend
└── docker-compose.yml  # Container orchestration
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/xorinf/sentinel.git
cd sentinel

# Start with Docker
docker-compose up -d

# Or run locally
pip install -r requirements.txt
python -m src.api.main
```

## 📊 Models

| Model | Purpose | Performance |
|-------|---------|-------------|
| VAE + Isolation Forest | Anomaly Detection | AUC: 0.95+ |
| LSTM/Transformer | Behavior Prediction | Accuracy: 87% |
| Graph Attention Network | Lateral Movement | F1: 0.89 |
| PPO (RL) | Threat Prioritization | 50% alert reduction |

## 🔬 Datasets

- CICIDS2017, UNSW-NB15, LANL, Mordor
- Synthetic attack generation included

## 📜 License

MIT License - See [LICENSE](LICENSE)

---

<div align="center">

**Built with 💜 by [xorinf](https://github.com/xorinf)**

</div>
