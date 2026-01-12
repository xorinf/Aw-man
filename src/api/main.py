"""
SENTINEL API - FastAPI endpoints for threat detection
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np
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


# ============= State =============

class AppState:
    """Application state for models"""
    anomaly_detector = None
    behavior_predictor = None
    explainer = None
    models_loaded = False

state = AppState()


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
    
    return AnalysisResponse(
        analyzed_count=len(request.flows),
        alerts=alerts,
        summary={
            "total_flows": len(request.flows),
            "threats_detected": len(alerts),
            "critical": len([a for a in alerts if a.verdict == "CRITICAL"]),
            "high": len([a for a in alerts if a.verdict == "HIGH"]),
        }
    )


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
