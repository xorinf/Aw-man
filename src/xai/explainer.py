"""
Explainable AI Module - SHAP and Attention-based Explanations
"""
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


@dataclass
class ThreatExplanation:
    """Complete explanation of a detected threat"""
    alert_id: str
    verdict: str
    confidence: float
    top_factors: List[Dict[str, Any]]
    counterfactual: Optional[str] = None
    similar_attacks: Optional[List[str]] = None
    attention_highlights: Optional[List[Dict[str, float]]] = None
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
    
    def to_summary(self) -> str:
        """Human-readable summary"""
        lines = [
            f"🚨 Alert: {self.alert_id}",
            f"📋 Verdict: {self.verdict} ({self.confidence*100:.1f}% confidence)",
            "",
            "🔍 Key Factors:"
        ]
        for i, factor in enumerate(self.top_factors[:5], 1):
            impact = "↑" if factor.get("impact", 0) > 0 else "↓"
            lines.append(f"  {i}. {factor['feature']}: {factor['reason']} ({impact}{abs(factor.get('impact', 0)):.2f})")
        
        if self.counterfactual:
            lines.append(f"\n💡 Counterfactual: {self.counterfactual}")
        
        if self.similar_attacks:
            lines.append(f"\n🔗 Similar to: {', '.join(self.similar_attacks)}")
        
        return "\n".join(lines)


class SHAPExplainer:
    """SHAP-based model explanation"""
    
    def __init__(self, model, feature_names: List[str], background_data: np.ndarray):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        
        if SHAP_AVAILABLE:
            self.explainer = shap.KernelExplainer(
                self._model_predict,
                shap.sample(background_data, 100)
            )
        else:
            self.explainer = None
            
    def _model_predict(self, X: np.ndarray) -> np.ndarray:
        """Wrapper for model prediction"""
        # Adapt based on model type
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'predict'):
            result = self.model.predict(X)
            if hasattr(result, 'anomaly_score'):
                return np.array([[r.anomaly_score] for r in result])
            return result.reshape(-1, 1)
        return np.zeros((len(X), 1))
    
    def explain(self, sample: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get SHAP explanation for a sample"""
        if not SHAP_AVAILABLE or self.explainer is None:
            return self._fallback_explain(sample, top_k)
        
        shap_values = self.explainer.shap_values(sample.reshape(1, -1))
        
        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        shap_values = shap_values.flatten()
        
        # Sort by absolute impact
        indices = np.argsort(np.abs(shap_values))[::-1][:top_k]
        
        factors = []
        for idx in indices:
            factors.append({
                "feature": self.feature_names[idx],
                "impact": float(shap_values[idx]),
                "value": float(sample[idx]),
                "reason": self._generate_reason(self.feature_names[idx], sample[idx], shap_values[idx])
            })
        
        return factors
    
    def _fallback_explain(self, sample: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Fallback when SHAP not available - use feature deviation"""
        mean = self.background_data.mean(axis=0)
        std = self.background_data.std(axis=0) + 1e-6
        
        deviations = np.abs(sample - mean) / std
        indices = np.argsort(deviations)[::-1][:top_k]
        
        factors = []
        for idx in indices:
            factors.append({
                "feature": self.feature_names[idx],
                "impact": float(deviations[idx]),
                "value": float(sample[idx]),
                "reason": f"Value deviates {deviations[idx]:.1f} std from normal"
            })
        
        return factors
    
    def _generate_reason(self, feature: str, value: float, impact: float) -> str:
        """Generate human-readable reason for feature impact"""
        direction = "increases" if impact > 0 else "decreases"
        
        reason_templates = {
            "port_entropy": f"Port diversity is {'unusually high' if value > 2 else 'normal'} ({value:.2f})",
            "bytes_per_packet": f"Packet size ratio {'suspicious' if impact > 0 else 'normal'}",
            "connection_count": f"Connection rate {'elevated' if value > 50 else 'normal'} ({int(value)})",
            "unusual_time": f"Activity during {'unusual hours' if value > 0.5 else 'business hours'}",
            "process_depth": f"Process tree depth {'unusually deep' if value > 3 else 'normal'} ({int(value)})",
        }
        
        return reason_templates.get(feature, f"Feature {direction} anomaly score by {abs(impact):.2f}")


class AttentionExplainer:
    """Explain sequence model predictions using attention weights"""
    
    def __init__(self, event_labels: List[str] = None):
        self.event_labels = event_labels or []
    
    def explain(self, attention_weights: np.ndarray, 
                sequence_events: List[str] = None) -> List[Dict[str, float]]:
        """Convert attention weights to explanation"""
        if sequence_events is None:
            sequence_events = [f"Event_{i}" for i in range(len(attention_weights))]
        
        # Normalize attention
        attention_weights = attention_weights / (attention_weights.sum() + 1e-6)
        
        highlights = []
        for i, (event, weight) in enumerate(zip(sequence_events, attention_weights)):
            if weight > 0.05:  # Only show significant attention
                highlights.append({
                    "event": event,
                    "position": i,
                    "attention": float(weight),
                    "importance": "high" if weight > 0.15 else "medium"
                })
        
        return sorted(highlights, key=lambda x: x["attention"], reverse=True)


class CounterfactualGenerator:
    """Generate counterfactual explanations"""
    
    def __init__(self, feature_names: List[str], normal_distribution: Dict[str, tuple]):
        self.feature_names = feature_names
        self.normal_distribution = normal_distribution  # {feature: (mean, std)}
    
    def generate(self, sample: np.ndarray, top_factors: List[Dict]) -> str:
        """Generate counterfactual explanation"""
        if not top_factors:
            return "No significant anomalous features detected"
        
        top_feature = top_factors[0]["feature"]
        current_value = top_factors[0]["value"]
        
        if top_feature in self.normal_distribution:
            mean, std = self.normal_distribution[top_feature]
            normal_range = f"{mean-std:.2f} to {mean+std:.2f}"
            return f"Alert would NOT trigger if '{top_feature}' was in normal range ({normal_range}) instead of {current_value:.2f}"
        
        return f"Alert would likely not trigger if '{top_feature}' had a normal value"


class ThreatExplainerPipeline:
    """Complete explanation pipeline for SENTINEL"""
    
    KNOWN_ATTACK_PATTERNS = {
        "port_scan": ["Reconnaissance", "NMAP Scan"],
        "c2_beacon": ["Cobalt Strike", "Empire C2"],
        "lateral_move": ["Pass-the-Hash", "RDP Pivot"],
        "exfiltration": ["DNS Tunneling", "HTTPS Exfil"]
    }
    
    def __init__(self, anomaly_model, feature_names: List[str], 
                 background_data: np.ndarray):
        self.shap_explainer = SHAPExplainer(anomaly_model, feature_names, background_data)
        self.attention_explainer = AttentionExplainer()
        
        # Calculate normal distribution for counterfactuals
        self.normal_dist = {
            name: (float(background_data[:, i].mean()), float(background_data[:, i].std()))
            for i, name in enumerate(feature_names)
        }
        self.counterfactual_gen = CounterfactualGenerator(feature_names, self.normal_dist)
        self.feature_names = feature_names
    
    def explain(self, alert_id: str, sample: np.ndarray, 
                anomaly_score: float, predicted_attack_type: str = None,
                attention_weights: np.ndarray = None) -> ThreatExplanation:
        """Generate complete threat explanation"""
        
        # Determine verdict
        if anomaly_score > 0.9:
            verdict = "CRITICAL"
        elif anomaly_score > 0.7:
            verdict = "HIGH"
        elif anomaly_score > 0.5:
            verdict = "MEDIUM"
        else:
            verdict = "LOW"
        
        # Get SHAP factors
        top_factors = self.shap_explainer.explain(sample, top_k=5)
        
        # Generate counterfactual
        counterfactual = self.counterfactual_gen.generate(sample, top_factors)
        
        # Find similar known attacks
        similar_attacks = None
        if predicted_attack_type and predicted_attack_type in self.KNOWN_ATTACK_PATTERNS:
            similar_attacks = self.KNOWN_ATTACK_PATTERNS[predicted_attack_type]
        
        # Process attention weights if available
        attention_highlights = None
        if attention_weights is not None:
            attention_highlights = self.attention_explainer.explain(attention_weights)
        
        return ThreatExplanation(
            alert_id=alert_id,
            verdict=verdict,
            confidence=min(anomaly_score, 0.99),
            top_factors=top_factors,
            counterfactual=counterfactual,
            similar_attacks=similar_attacks,
            attention_highlights=attention_highlights
        )


logger.info("XAI module initialized")
