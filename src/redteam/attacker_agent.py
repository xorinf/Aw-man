"""
AI Red Team - LLM-powered Attacker Simulation
Multi-agent attack generation for adversarial training
"""
"""
SENTINEL AI Red Team Module
===========================

Simulates cyber attacks for training and validation.
Implements different attacker personas with distinct behaviors.

Personas:
    APTAgent: Stealthy, slow, goal-oriented (e.g., APT29)
    OpportunisticAgent: Noisy, fast, brute-force (e.g., Script Kiddie)

Author: xorinf
Version: 1.0.0
"""
import random
import numpy as np
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from loguru import logger

try:
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Using rule-based simulation.")


class AttackStage(Enum):
    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    COMMAND_AND_CONTROL = "command_and_control"
    EXFILTRATION = "exfiltration"


@dataclass
class AttackAction:
    """Single attack action with simulated traffic"""
    stage: AttackStage
    technique: str
    mitre_id: str
    description: str
    simulated_features: np.ndarray
    timestamp: float = 0.0
    source_ip: str = ""
    target_ip: str = ""


@dataclass
class AttackCampaign:
    """Complete attack campaign with multiple stages"""
    campaign_id: str
    persona: str
    actions: List[AttackAction] = field(default_factory=list)
    duration_hours: float = 0.0
    target_network: str = ""


class AttackerPersona(ABC):
    """Base class for attacker personas"""
    
    name: str = "BaseAttacker"
    description: str = ""
    speed: str = "medium"  # slow, medium, fast
    stealth: str = "medium"  # low, medium, high
    
    @abstractmethod
    def generate_campaign(self, target_network: Dict) -> AttackCampaign:
        pass
    
    @abstractmethod
    def generate_action_features(self, stage: AttackStage) -> np.ndarray:
        """Generate realistic feature vector for this action"""
        pass


class APTAgent(AttackerPersona):
    """
    Advanced Persistent Threat actor.
    Slow, stealthy, multi-stage attacks over extended periods.
    """
    
    name = "APT29_Cozy_Bear"
    description = "Nation-state actor, slow and methodical"
    speed = "slow"
    stealth = "high"
    
    ATTACK_CHAIN = [
        (AttackStage.RECONNAISSANCE, "T1595", "Active Scanning"),
        (AttackStage.INITIAL_ACCESS, "T1566", "Phishing"),
        (AttackStage.EXECUTION, "T1059", "PowerShell Execution"),
        (AttackStage.PERSISTENCE, "T1053", "Scheduled Task"),
        (AttackStage.CREDENTIAL_ACCESS, "T1003", "OS Credential Dumping"),
        (AttackStage.DISCOVERY, "T1087", "Account Discovery"),
        (AttackStage.LATERAL_MOVEMENT, "T1021", "Remote Services"),
        (AttackStage.COLLECTION, "T1005", "Data from Local System"),
        (AttackStage.COMMAND_AND_CONTROL, "T1071", "Web Protocols"),
        (AttackStage.EXFILTRATION, "T1048", "Exfiltration Over C2"),
    ]
    
    def generate_campaign(self, target_network: Dict = None) -> AttackCampaign:
        campaign = AttackCampaign(
            campaign_id=f"APT-{random.randint(1000, 9999)}",
            persona=self.name,
            duration_hours=random.uniform(72, 336),  # 3-14 days
            target_network=str(target_network) if target_network else "10.0.0.0/8"
        )
        
        base_time = 0.0
        for stage, mitre_id, technique in self.ATTACK_CHAIN:
            # APT adds delays between stages
            base_time += random.uniform(1, 24) * 3600  # 1-24 hours between stages
            
            action = AttackAction(
                stage=stage,
                technique=technique,
                mitre_id=mitre_id,
                description=f"{self.name} executing {technique}",
                simulated_features=self.generate_action_features(stage),
                timestamp=base_time,
                source_ip=f"10.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}",
                target_ip=f"10.0.{random.randint(1,254)}.{random.randint(1,254)}"
            )
            campaign.actions.append(action)
        
        return campaign
    
    def generate_action_features(self, stage: AttackStage) -> np.ndarray:
        """Generate stealthy, low-volume features"""
        # 13 features matching NetworkFeatureExtractor
        base = np.zeros(13, dtype=np.float32)
        
        if stage == AttackStage.RECONNAISSANCE:
            base[0] = random.randint(40000, 65535)  # src_port (high)
            base[1] = random.choice([22, 80, 443, 445])  # dst_port
            base[2] = random.randint(100, 500)  # bytes_sent (low for stealth)
            base[3] = random.randint(50, 200)  # bytes_recv
            base[4] = random.randint(1, 5)  # packets_sent
            base[5] = random.randint(1, 3)  # packets_recv
            base[6] = random.uniform(0.1, 2)  # duration
            base[11] = random.uniform(0.5, 1.5)  # port_entropy (moderate)
            
        elif stage == AttackStage.COMMAND_AND_CONTROL:
            base[0] = random.randint(40000, 65535)
            base[1] = 443  # HTTPS blending
            base[2] = random.randint(200, 1000)  # Small beacon
            base[3] = random.randint(100, 500)
            base[4] = random.randint(2, 5)
            base[5] = random.randint(1, 3)
            base[6] = random.uniform(30, 90)  # Regular interval
            
        elif stage == AttackStage.EXFILTRATION:
            base[0] = random.randint(40000, 65535)
            base[1] = random.choice([443, 53])  # HTTPS or DNS
            base[2] = random.randint(10000, 50000)  # More data out
            base[3] = random.randint(100, 500)
            base[4] = random.randint(10, 50)
            base[5] = random.randint(5, 20)
            base[6] = random.uniform(10, 60)
            
        else:
            # Default internal activity
            base[0] = random.randint(40000, 65535)
            base[1] = random.choice([135, 445, 3389, 5985])  # Windows services
            base[2] = random.randint(500, 5000)
            base[3] = random.randint(200, 2000)
            base[4] = random.randint(5, 20)
            base[5] = random.randint(3, 15)
            base[6] = random.uniform(1, 10)
        
        # Derived features
        base[7] = 0  # protocol (TCP)
        base[8] = (base[2] + base[3]) / (base[4] + base[5] + 1)  # bytes_per_packet
        base[9] = (base[4] + base[5]) / (base[6] + 0.01)  # packet_rate
        base[10] = base[2] / (base[3] + 1)  # byte_ratio
        base[12] = random.randint(5, 50)  # connection_count
        
        return base


class OpportunisticAgent(AttackerPersona):
    """
    Script kiddie / opportunistic attacker.
    Fast, noisy, uses automated tools.
    """
    
    name = "OpportunisticHacker"
    description = "Automated tools, noisy, quick smash-and-grab"
    speed = "fast"
    stealth = "low"
    
    ATTACK_CHAIN = [
        (AttackStage.RECONNAISSANCE, "T1046", "Network Service Scanning"),
        (AttackStage.INITIAL_ACCESS, "T1190", "Exploit Public-Facing App"),
        (AttackStage.EXECUTION, "T1059", "Command and Scripting"),
        (AttackStage.EXFILTRATION, "T1567", "Exfil to Cloud Storage"),
    ]
    
    def generate_campaign(self, target_network: Dict = None) -> AttackCampaign:
        campaign = AttackCampaign(
            campaign_id=f"OPP-{random.randint(1000, 9999)}",
            persona=self.name,
            duration_hours=random.uniform(0.5, 4),  # Minutes to hours
            target_network=str(target_network) if target_network else "10.0.0.0/8"
        )
        
        base_time = 0.0
        for stage, mitre_id, technique in self.ATTACK_CHAIN:
            base_time += random.uniform(60, 600)  # Seconds to minutes
            
            action = AttackAction(
                stage=stage,
                technique=technique,
                mitre_id=mitre_id,
                description=f"{self.name} executing {technique}",
                simulated_features=self.generate_action_features(stage),
                timestamp=base_time
            )
            campaign.actions.append(action)
        
        return campaign
    
    def generate_action_features(self, stage: AttackStage) -> np.ndarray:
        """Generate noisy, high-volume features"""
        base = np.zeros(13, dtype=np.float32)
        
        if stage == AttackStage.RECONNAISSANCE:
            base[0] = random.randint(40000, 65535)
            base[1] = random.randint(1, 1024)  # Scanning many ports
            base[2] = random.randint(50, 200)
            base[3] = random.randint(50, 200)
            base[4] = random.randint(100, 1000)  # Many packets (scan)
            base[5] = random.randint(50, 500)
            base[6] = random.uniform(0.01, 1)  # Fast
            base[11] = random.uniform(3, 5)  # HIGH port entropy
            base[12] = random.randint(100, 500)  # Many connections
            
        else:
            base[0] = random.randint(40000, 65535)
            base[1] = random.choice([80, 443, 8080, 8443])
            base[2] = random.randint(5000, 50000)
            base[3] = random.randint(1000, 10000)
            base[4] = random.randint(50, 200)
            base[5] = random.randint(20, 100)
            base[6] = random.uniform(1, 30)
        
        # Derived
        base[7] = 0
        base[8] = (base[2] + base[3]) / (base[4] + base[5] + 1)
        base[9] = (base[4] + base[5]) / (base[6] + 0.01)
        base[10] = base[2] / (base[3] + 1)
        
        return base


class RedTeamSimulator:
    """
    Main simulator that generates attack traffic for model training.
    """
    
    PERSONAS = {
        "apt": APTAgent(),
        "opportunistic": OpportunisticAgent(),
    }
    
    def __init__(self, personas: List[str] = None):
        self.active_personas = []
        
        if personas is None:
            personas = list(self.PERSONAS.keys())
        
        for p in personas:
            if p in self.PERSONAS:
                self.active_personas.append(self.PERSONAS[p])
            else:
                logger.warning(f"Unknown persona: {p}")
    
    def generate_campaigns(self, num_campaigns: int = 10) -> List[AttackCampaign]:
        """Generate multiple attack campaigns"""
        campaigns = []
        
        for _ in range(num_campaigns):
            persona = random.choice(self.active_personas)
            campaign = persona.generate_campaign()
            campaigns.append(campaign)
            logger.info(f"Generated campaign {campaign.campaign_id} using {persona.name}")
        
        return campaigns
    
    def generate_attack_stream(self, duration_hours: float = 24
                               ) -> Generator[AttackAction, None, None]:
        """Generate continuous stream of attack actions"""
        current_time = 0.0
        end_time = duration_hours * 3600
        
        while current_time < end_time:
            persona = random.choice(self.active_personas)
            campaign = persona.generate_campaign()
            
            for action in campaign.actions:
                action.timestamp += current_time
                if action.timestamp < end_time:
                    yield action
            
            current_time += campaign.duration_hours * 3600 * random.uniform(0.5, 2)
    
    def get_training_data(self, num_samples: int = 1000
                          ) -> tuple[np.ndarray, np.ndarray]:
        """Generate labeled training data"""
        features = []
        labels = []
        
        for _ in range(num_samples):
            persona = random.choice(self.active_personas)
            stage = random.choice(list(AttackStage))
            
            feature_vec = persona.generate_action_features(stage)
            label = list(AttackStage).index(stage)
            
            features.append(feature_vec)
            labels.append(label)
        
        return np.array(features), np.array(labels)


logger.info("Red Team simulator initialized")
