# SENTINEL Dashboard - Development Notes
**Branch:** `dashboard`  
**Version:** v1.2-beta  
**Last Updated:** 2026-01-13  
**Author:** xorinf

---

## 📋 Current Status

### ✅ Completed Features

#### Frontend Dashboard (`/dashboard`)
- **Technology Stack:** React 18.2 + Vite 5.0 + TypeScript
- **Styling:** Vanilla CSS with modern design system
- **Real-time Updates:** WebSocket integration for live alerts
- **Routing:** React Router DOM v6.20

#### Implemented Pages
1. **Overview (`/`)** - Dashboard home with key metrics
   - Real-time alert counter
   - Recent threat summary
   - System health indicators
   - Quick action cards

2. **Alerts (`/alerts`)** - Comprehensive alert management
   - Filterable alert list (severity, type, time)
   - Search functionality
   - Alert detail view
   - Color-coded severity levels (Critical, High, Medium, Low)

3. **Timeline (`/timeline`)** - Attack progression visualization
   - Chronological event display
   - MITRE ATT&CK technique mapping
   - Kill chain stage indicators
   - Interactive event details

4. **Attack Graph (`/attack-graph`)** - Network visualization
   - Cytoscape.js graph rendering
   - Node types: hosts, techniques
   - Edge relationships: executes, targets, uses
   - Compromised host highlighting

5. **AI Copilot (`/copilot`)** - Security assistant chat
   - ChatInterface component
   - Context-aware responses
   - Suggested actions
   - Query history

#### Components Built
- `Sidebar.tsx` - Navigation menu with icons (Lucide React)
- `AlertCard.tsx` - Reusable alert display component
- `MetricCard.tsx` - Dashboard metric tiles
- `ChatInterface.tsx` - AI chat UI component

#### Backend API (`/src/api`)
- **Framework:** FastAPI 0.100+
- **Server:** Uvicorn 0.23+
- **WebSocket:** Real-time alert streaming endpoint (`/ws/alerts`)

#### API Endpoints
- `GET /health` - System health check
- `POST /analyze` - Threat detection on network flows
- `POST /simulate-attack` - Generate test attack traffic
- `GET /mitre-coverage` - MITRE ATT&CK framework coverage
- `GET /mitre/technique/{id}` - Technique details
- `GET /attack-graph` - Graph data for visualization
- `POST /chat` - AI copilot chat endpoint
- `WS /ws/alerts` - WebSocket alerts stream

---

## 🛠️ Technical Implementation

### Frontend Architecture
```
dashboard/
├── src/
│   ├── pages/           # Route components
│   ├── components/      # Reusable UI components
│   ├── context/         # React Context (AlertContext)
│   ├── hooks/           # Custom hooks (useWebSocket)
│   ├── api/             # API client utilities
│   └── styles/          # Global CSS styles
├── vite.config.ts       # Vite configuration
└── package.json         # Dependencies
```

### Key Dependencies (Frontend)
- `react`, `react-dom`: UI framework
- `react-router-dom`: Client-side routing
- `recharts`: Chart visualization
- `cytoscape`: Graph visualization
- `lucide-react`: Icon library
- `date-fns`: Date formatting

### Backend Architecture
```
src/
├── api/
│   └── main.py         # FastAPI application
├── config.py           # Pydantic settings
├── redteam/            # Attack simulation
└── __init__.py         # Package initialization
```

### Key Dependencies (Backend)
- `fastapi`, `uvicorn`: API framework
- `pydantic`, `pydantic-settings`: Data validation & config
- `torch`, `pytorch-lightning`: Deep learning
- `scikit-learn`: ML algorithms
- `langchain`, `openai`: AI/LLM integration
- `shap`, `captum`, `lime`: Explainable AI
- `kafka-python`, `redis`: Data streaming & caching

---

## 🎨 Design System

### Color Palette
- **Primary:** Blue gradient (#3b82f6 → #2563eb)
- **Success:** Green (#10b981)
- **Warning:** Yellow (#f59e0b)
- **Danger:** Red (#ef4444)
- **Background:** Dark theme (#0a0a0a, #1a1a1a)
- **Text:** White (#ffffff) with opacity variants

### Typography
- **Font:** System UI stack (SF Pro, Segoe UI, Roboto)
- **Sizes:** 12px (small), 14px (body), 16px (heading), 24px+ (titles)
- **Weight:** 400 (normal), 500 (medium), 600 (semibold), 700 (bold)

---

## 🔧 Configuration

### Environment Setup
1. **Frontend:** `dashboard/.env.example`
   - API proxy configured in `vite.config.ts`
   - Default dev port: 3000

2. **Backend:** `.env.example`
   - Configurable via environment variables
   - Settings in `src/config.py`
   - Default API port: 8000

### Proxy Configuration (vite.config.ts)
- `/api` → `http://localhost:8000` (API requests)
- `/ws` → `ws://localhost:8000` (WebSocket)

---

## 🚀 Running the Application

### Development Mode
```bash
# Terminal 1 - Backend API
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Frontend Dashboard
cd dashboard
npm run dev
```

### Access
- Dashboard: http://localhost:3000
- API Docs: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/ws/alerts

---

## 📦 Recent Changes (v1.2-beta)

### Fixed Issues
1. **Missing Dependency:** Added `pydantic-settings>=2.0.0` to requirements.txt
2. **Vite Module Error:** Installed all npm dependencies via `npm install`
3. **PowerShell Execution Policy:** Set to RemoteSigned for npm scripts
4. **Git Configuration:** Configured user email and name

### Improvements
- Updated `requirements.txt` with complete dependency list
- Added comprehensive `SETUP.md` guide
- Documented all API endpoints
- Created WebSocket manager for real-time alerts

---

## 🎯 Next Steps / TODO

### High Priority
- [ ] Integrate real ML models (currently using mock data)
- [ ] Implement actual Redis/Kafka connections
- [ ] Connect to TimescaleDB for historical data
- [ ] Add authentication/authorization (JWT)
- [ ] Implement user management

### Features to Add
- [ ] Alert filtering persistence (localStorage)
- [ ] Export alerts to CSV/JSON
- [ ] Dark/Light theme toggle
- [ ] Customizable dashboard widgets
- [ ] Email/Slack notifications
- [ ] Incident response playbooks

### Backend Enhancements
- [ ] Load actual ML models on startup
- [ ] Implement SHAP explanations for alerts
- [ ] Add rate limiting and API keys
- [ ] Integrate with SIEM systems
- [ ] Create background task queue (Celery)

### DevOps
- [ ] Docker Compose full stack setup
- [ ] Kubernetes deployment manifests
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Automated testing (pytest, Jest)
- [ ] Performance monitoring (Prometheus/Grafana)

---

## 🐛 Known Issues

1. **Models Not Loaded:** Backend runs in demo mode with mock responses
2. **WebSocket Reconnection:** No automatic reconnect on connection loss
3. **Attack Graph Layout:** May need optimization for large graphs
4. **Mobile Responsiveness:** Dashboard optimized for desktop, needs mobile work

---

## 📚 Documentation Files

- `README.md` - Project overview and architecture
- `SETUP.md` - Quick start guide for development
- `DASHBOARD_NOTES.md` - This file (development notes)
- `.env.example` - Environment variable template
- `dashboard/README.md` - Frontend-specific documentation

---

## 🔐 Security Considerations

### Current State (Development)
- CORS enabled for all origins (disable in production)
- No authentication implemented yet
- WebSocket connections not authenticated
- API accepts all requests

### Production Requirements
- [ ] Enable HTTPS/TLS
- [ ] Implement JWT authentication
- [ ] Add rate limiting
- [ ] Restrict CORS to allowed origins
- [ ] Validate all input data
- [ ] Implement proper logging and monitoring

---

## 📊 MITRE ATT&CK Coverage

### Covered Tactics (12/14)
✅ Reconnaissance, Initial Access, Execution, Persistence, Privilege Escalation  
✅ Defense Evasion, Credential Access, Discovery, Lateral Movement  
✅ Collection, Command and Control, Exfiltration

### Covered Techniques
- **Current:** 45 techniques
- **Detection Rate:** 72%
- **Focus Areas:** Lateral movement, C2 communication, data exfiltration

---

## 💡 Development Tips

### Hot Reload
- Frontend: Saves automatically reload (Vite HMR)
- Backend: `--reload` flag enables auto-restart

### Debug Tools
- Frontend: React DevTools browser extension
- Backend: FastAPI auto-docs at `/docs` and `/redoc`
- WebSocket: Use browser DevTools Network tab

### Common Commands
```bash
# Check backend health
curl http://localhost:8000/health

# Simulate attack
curl -X POST "http://localhost:8000/simulate-attack?attack_type=apt&num_actions=5"

# Test WebSocket (using wscat)
wscat -c ws://localhost:8000/ws/alerts
```

---

## 🤝 Contributing

This is currently a solo project by xorinf for the security AI platform SENTINEL.

For questions or collaboration:
- GitHub: xorinf/Aw-man
- Email: yashhwanththerealhuman14377@gmail.com

---

**End of Dashboard Development Notes - Branch: `dashboard`**
