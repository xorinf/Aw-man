# 🚀 SENTINEL Dashboard - Quick Setup Guide

## Prerequisites

> **⚠️ IMPORTANT**: Install Node.js before proceeding!  
> Download: https://nodejs.org/ (Choose LTS version 18 or higher)

## Installation Steps

### 1️⃣ Install Dashboard Dependencies

```powershell
cd C:\Users\Meow\Documents\GenTic\dashboard
npm install
```

This will install all required packages including:
- React, React Router
- Recharts (charts)
- Cytoscape (attack graphs)
- Lucide React (icons)
- date-fns (dates)

### 2️⃣ Start the Backend API

```powershell
# In a new terminal
cd C:\Users\Meow\Documents\GenTic
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3️⃣ Start the Dashboard

```powershell
# In the dashboard directory
cd C:\Users\Meow\Documents\GenTic\dashboard
npm run dev
```

### 4️⃣ Open Your Browser

Navigate to: **http://localhost:3000**

The dashboard will automatically connect to the backend WebSocket!

---

## Testing Real-Time Alerts

In a new terminal, simulate an attack:

```powershell
curl -X POST "http://localhost:8000/simulate-attack?attack_type=apt&num_actions=5"
```

You should see alerts appear in real-time in the dashboard! 🔴

---

## Dashboard Features

✅ **Overview** - Key metrics and recent alerts  
✅ **Alerts** - Filter and search through all threats  
✅ **Timeline** - Visualize attack progression  
✅ **Attack Graph** - Interactive network visualization  
✅ **AI Copilot** - Chat with security assistant  

---

## Troubleshooting

### Port Already in Use?

Change the port in `dashboard/vite.config.ts`:
```typescript
server: {
  port: 3001, // Change this
```

### WebSocket Not Connecting?

Check that the backend is running on port 8000:
```powershell
curl http://localhost:8000/health
```

### Module Not Found Errors?

Reinstall dependencies:
```powershell
rm -rf node_modules package-lock.json
npm install
```

---

## Docker Deployment

For production deployment:

```powershell
docker-compose up --build
```

This starts:
- Backend API (port 8000)
- Dashboard (port 3000)
- Redis, Kafka, TimescaleDB

---

**Happy Threat Hunting! 🛡️**
