"""
SENTINEL API - FastAPI endpoints for threat detection
"""
"""
SENTINEL API Gateway
====================

FastAPI application serving the core platform functionality.
Exposes endpoints for analysis, simulation, and health monitoring.

Endpoints:
    /analyze: Threat detection on network flows
    /simulate-attack: Generate adversarial traffic
    /mitre-coverage: Reporting
    /health: System status

Author: xorinf
Version: 1.0.0
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Set
from datetime import datetime
import numpy as np
import asyncio
import json
from loguru import logger

# App instance
app = FastAPI(
    title="SENTINEL API",
    description="Autonomous AI Cyber Defense Platform",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Schemas =============

class NetworkFlowInput(BaseModel):
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str = "TCP"
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    duration: float
    timestamp: Optional[float] = None


class ThreatAlert(BaseModel):
    alert_id: str
    timestamp: str
    verdict: str
    confidence: float
    source_ip: str
    destination_ip: str
    threat_type: Optional[str] = None
    kill_chain_stage: Optional[str] = None
    explanation: Dict[str, Any]
    recommended_actions: List[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    timestamp: str


class AnalysisRequest(BaseModel):
    flows: List[NetworkFlowInput]


class AnalysisResponse(BaseModel):
    analyzed_count: int
    alerts: List[ThreatAlert]
    summary: Dict[str, Any]


class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    suggestions: List[str]


# ============= State =============

class AppState:
    """Application state for models"""
    anomaly_detector = None
    behavior_predictor = None
    explainer = None
    models_loaded = False

state = AppState()


# ============= WebSocket Manager =============

class ConnectionManager:
    """Manage WebSocket connections for real-time alerts"""
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast alert to all connected clients"""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()


# ============= Endpoints =============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        models_loaded=state.models_loaded,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_traffic(request: AnalysisRequest):
    """Analyze network flows for threats"""
    
    if not state.models_loaded:
        # Generate mock response for demo
        return _mock_analysis(request.flows)
    
    alerts = []
    for flow in request.flows:
        # Convert to feature vector
        features = _flow_to_features(flow)
        
        # Run anomaly detection
        result = state.anomaly_detector.predict(features)
        
        if result.is_anomaly:
            alert = _create_alert(flow, result)
            alerts.append(alert)
    
    response = AnalysisResponse(
        analyzed_count=len(request.flows),
        alerts=alerts,
        summary={
            "total_flows": len(request.flows),
            "threats_detected": len(alerts),
            "critical": len([a for a in alerts if a.verdict == "CRITICAL"]),
            "high": len([a for a in alerts if a.verdict == "HIGH"]),
        }
    )
    
    # Broadcast alerts via WebSocket
    for alert in alerts:
        await manager.broadcast({
            "type": "alert",
            "data": alert.model_dump(),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    return response


@app.post("/simulate-attack")
async def simulate_attack(attack_type: str = "apt", num_actions: int = 10):
    """Generate simulated attack traffic for testing"""
    from src.redteam.attacker_agent import RedTeamSimulator
    
    simulator = RedTeamSimulator(personas=[attack_type])
    campaigns = simulator.generate_campaigns(num_campaigns=1)
    
    if not campaigns:
        raise HTTPException(status_code=400, detail=f"Unknown attack type: {attack_type}")
    
    campaign = campaigns[0]
    
    return {
        "campaign_id": campaign.campaign_id,
        "persona": campaign.persona,
        "duration_hours": campaign.duration_hours,
        "actions": [
            {
                "stage": a.stage.value,
                "technique": a.technique,
                "mitre_id": a.mitre_id,
                "description": a.description,
                "timestamp": a.timestamp
            }
            for a in campaign.actions[:num_actions]
        ]
    }


@app.get("/mitre-coverage")
async def get_mitre_coverage():
    """Get MITRE ATT&CK coverage"""
    return {
        "covered_tactics": [
            "Reconnaissance", "Initial Access", "Execution",
            "Persistence", "Privilege Escalation", "Defense Evasion",
            "Credential Access", "Discovery", "Lateral Movement",
            "Collection", "Command and Control", "Exfiltration"
        ],
        "covered_techniques": 45,
        "detection_rate": 0.72
    }


@app.get("/mitre/technique/{technique_id}")
async def get_technique_details(technique_id: str):
    """Get MITRE ATT&CK technique details"""
    # Mock data - in production, load from MITRE ATT&CK database
    techniques = {
        "T1595": {
            "name": "Active Scanning",
            "tactic": "Reconnaissance",
            "description": "Adversaries may execute active reconnaissance scans to gather information.",
            "mitigations": ["Network Intrusion Prevention", "Pre-compromise"]
        },
        "T1566": {
            "name": "Phishing",
            "tactic": "Initial Access",
            "description": "Adversaries may send phishing messages to gain access to victim systems.",
            "mitigations": ["User Training", "Email Gateway Filtering"]
        },
        "T1059": {
            "name": "Command and Scripting Interpreter",
            "tactic": "Execution",
            "description": "Adversaries may abuse command and script interpreters to execute commands.",
            "mitigations": ["Execution Prevention", "Privileged Account Management"]
        }
    }
    
    if technique_id not in techniques:
        raise HTTPException(status_code=404, detail="Technique not found")
    
    return techniques[technique_id]


@app.get("/attack-graph")
async def get_attack_graph():
    """Get attack graph data for visualization"""
    # Mock attack graph - in production, build from real alert correlations
    return {
        "nodes": [
            {"id": "192.168.1.100", "label": "192.168.1.100", "type": "host", "compromised": True},
            {"id": "192.168.1.50", "label": "192.168.1.50", "type": "host", "compromised": True},
            {"id": "10.0.0.5", "label": "10.0.0.5", "type": "host", "compromised": False},
            {"id": "T1059", "label": "Command Execution", "type": "technique"},
            {"id": "T1021", "label": "Remote Services", "type": "technique"},
        ],
        "edges": [
            {"source": "192.168.1.100", "target": "T1059", "label": "executes"},
            {"source": "T1059", "target": "192.168.1.50", "label": "targets"},
            {"source": "192.168.1.50", "target": "T1021", "label": "uses"},
            {"source": "T1021", "target": "10.0.0.5", "label": "attempts"},
        ]
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_copilot(message: ChatMessage):
    """AI Security Copilot chat endpoint"""
    user_msg = message.message.lower()
    
    # Simple rule-based responses - in production, use LLM
    if "alert" in user_msg or "threat" in user_msg:
        response = """Based on recent alerts, I've detected:
        
• **3 high-severity threats** in the past hour
• Primary concern: Potential lateral movement from 192.168.1.100
• Affected hosts: 192.168.1.50, 192.168.1.75

**Recommended Actions:**
1. Isolate host 192.168.1.100 from the network
2. Investigate processes running on compromised hosts
3. Review authentication logs for unauthorized access

Would you like me to generate a detailed incident report?"""
        suggestions = ["Show attack timeline", "Block source IP", "Generate report"]
    
    elif "mitre" in user_msg or "attack" in user_msg:
        response = """**MITRE ATT&CK Coverage Analysis:**

Detected techniques in recent activity:
• T1059 - Command and Scripting Interpreter
• T1021 - Remote Services  
• T1078 - Valid Accounts

These align with **Lateral Movement** and **Execution** tactics.

Coverage: 45 techniques across 12 tactics (Detection rate: 72%)"""
        suggestions = ["Show coverage heatmap", "View technique details", "Simulate APT attack"]
    
    else:
        response = """Hello! I'm your AI Security Copilot. I can help you:

• Analyze threats and alerts
• Explain detection decisions
• Recommend response actions
• Map attacks to MITRE ATT&CK
• Simulate adversary behavior

What would you like to investigate?"""
        suggestions = ["Show latest alerts", "Explain last detection", "Run attack simulation"]
    
    return ChatResponse(
        response=response,
        suggestions=suggestions
    )


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alert streaming"""
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to SENTINEL alert stream",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and listen for client messages
        while True:
            try:
                # Wait for any client messages (heartbeat, commands, etc.)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo back for heartbeat
                await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
            except asyncio.TimeoutError:
                # Send periodic heartbeat
                await websocket.send_json({"type": "heartbeat", "timestamp": datetime.utcnow().isoformat()})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ============= Helpers =============

def _flow_to_features(flow: NetworkFlowInput) -> np.ndarray:
    """Convert flow to feature vector"""
    return np.array([
        flow.src_port,
        flow.dst_port,
        flow.bytes_sent,
        flow.bytes_recv,
        flow.packets_sent,
        flow.packets_recv,
        flow.duration,
        0,  # protocol encoded
        (flow.bytes_sent + flow.bytes_recv) / (flow.packets_sent + flow.packets_recv + 1),
        (flow.packets_sent + flow.packets_recv) / (flow.duration + 0.01),
        flow.bytes_sent / (flow.bytes_recv + 1),
        1.5,  # port_entropy placeholder
        10  # connection_count placeholder
    ], dtype=np.float32)


def _create_alert(flow: NetworkFlowInput, result) -> ThreatAlert:
    """Create alert from detection result"""
    import uuid
    
    return ThreatAlert(
        alert_id=f"SENT-{uuid.uuid4().hex[:8].upper()}",
        timestamp=datetime.utcnow().isoformat(),
        verdict="HIGH" if result.anomaly_score > 0.7 else "MEDIUM",
        confidence=min(result.anomaly_score, 0.99),
        source_ip=flow.src_ip,
        destination_ip=flow.dst_ip,
        threat_type="Anomalous Traffic",
        explanation={
            "anomaly_score": result.anomaly_score,
            "reconstruction_error": result.reconstruction_error,
            "feature_contributions": result.feature_contributions
        },
        recommended_actions=[
            "Investigate source host",
            "Check for lateral movement",
            "Review recent file access"
        ]
    )


def _mock_analysis(flows: List[NetworkFlowInput]) -> AnalysisResponse:
    """Generate mock analysis for demo"""
    import uuid
    
    alerts = []
    for i, flow in enumerate(flows[:3]):  # Cap for demo
        if flow.dst_port in [4444, 5555, 1234] or flow.bytes_sent > 100000:
            alerts.append(ThreatAlert(
                alert_id=f"SENT-{uuid.uuid4().hex[:8].upper()}",
                timestamp=datetime.utcnow().isoformat(),
                verdict="HIGH",
                confidence=0.87,
                source_ip=flow.src_ip,
                destination_ip=flow.dst_ip,
                threat_type="Potential C2 Communication",
                kill_chain_stage="Command and Control",
                explanation={
                    "top_factors": [
                        {"feature": "dst_port", "impact": 0.45, "reason": "Uncommon destination port"},
                        {"feature": "bytes_sent", "impact": 0.32, "reason": "Unusual outbound volume"}
                    ],
                    "counterfactual": "Would not trigger if destination was a known service port"
                },
                recommended_actions=[
                    "Block destination IP",
                    "Isolate source host",
                    "Capture forensic image"
                ]
            ))
    
    return AnalysisResponse(
        analyzed_count=len(flows),
        alerts=alerts,
        summary={
            "total_flows": len(flows),
            "threats_detected": len(alerts),
            "critical": 0,
            "high": len(alerts)
        }
    )


# ============= Startup =============

@app.on_event("startup")
async def startup_event():
    logger.info("SENTINEL API starting...")
    # Models would be loaded here in production
    logger.info("API ready (demo mode - models not loaded)")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
