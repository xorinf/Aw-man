# SENTINEL Dashboard

React + TypeScript dashboard for the SENTINEL AI Cyber Defense Platform.

## Setup

### Prerequisites
- Node.js 18+ and npm

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Environment Variables

Create a `.env` file in the dashboard directory:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws/alerts
```

## Features

- 🔴 **Real-time WebSocket alerts**
- 💬 **AI Security Copilot chat**
- 📊 **Interactive dashboards and metrics**
- 🕸️ **Attack graph visualization**
- ⏱️ **Timeline analysis**
- 🎨 **Premium cybersecurity dark theme**

## Development

The dashboard connects to the SENTINEL API backend running on port 8000.

Make sure the backend is running:

```bash
cd ..
python -m uvicorn src.api.main:app --reload
```

Then start the dashboard:

```bash
cd dashboard
npm run dev
```

Visit: http://localhost:3000
