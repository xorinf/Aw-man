"""
Sequence Models for Attack Behavior Prediction
LSTM and Transformer architectures for predicting attacker actions
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from loguru import logger


@dataclass
class SequencePrediction:
    """Result from sequence prediction"""
    predicted_action: int
    action_probabilities: np.ndarray
    attention_weights: Optional[np.ndarray] = None
    threat_stage: Optional[str] = None


class AttackLSTM(nn.Module):
    """
    LSTM model for predicting attacker behavior sequences.
    Maps: [event_1, event_2, ..., event_n] -> next_action
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor, return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional attention weights
        
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Class logits [batch, num_classes]
            attention_weights: Optional attention weights [batch, seq_len]
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        
        # Attention mechanism
        attn_scores = self.attention(lstm_out).squeeze(-1)  # [batch, seq_len]
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        # Classification
        logits = self.classifier(context)
        
        if return_attention:
            return logits, attn_weights
        return logits, None


class AttackTransformer(nn.Module):
    """
    Transformer model for attack sequence prediction.
    Better at capturing long-range dependencies in APT attacks.
    """
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, num_classes: int = 10, 
                 max_seq_len: int = 100, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            return_attention: Whether to return attention (not implemented for transformer)
            
        Returns:
            logits: Class logits [batch, num_classes]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use CLS token for classification
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        
        return logits, None


class BehaviorPredictor:
    """
    High-level interface for attack behavior prediction.
    Maps attack sequences to kill chain stages and next actions.
    """
    
    KILL_CHAIN_STAGES = [
        "Reconnaissance",
        "Weaponization", 
        "Delivery",
        "Exploitation",
        "Installation",
        "Command & Control",
        "Actions on Objectives"
    ]
    
    ACTION_TYPES = [
        "port_scan", "dns_enum", "exploit_attempt", "malware_download",
        "persistence_install", "c2_beacon", "lateral_move", "privilege_esc",
        "data_access", "exfiltration"
    ]
    
    def __init__(self, input_dim: int, model_type: str = "lstm", 
                 hidden_dim: int = 128, device: str = "cpu"):
        self.input_dim = input_dim
        self.device = torch.device(device)
        self.num_classes = len(self.ACTION_TYPES)
        
        if model_type == "lstm":
            self.model = AttackLSTM(input_dim, hidden_dim, num_classes=self.num_classes)
        else:
            self.model = AttackTransformer(input_dim, hidden_dim, num_classes=self.num_classes)
        
        self.model.to(self.device)
        self.trained = False
        
    def train(self, sequences: np.ndarray, labels: np.ndarray,
              epochs: int = 100, batch_size: int = 32, lr: float = 1e-3):
        """Train the model on labeled attack sequences"""
        logger.info(f"Training behavior predictor on {len(sequences)} sequences")
        
        X = torch.FloatTensor(sequences).to(self.device)
        y = torch.LongTensor(labels).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits, _ = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_y.size(0)
            
            scheduler.step()
            
            if (epoch + 1) % 20 == 0:
                acc = 100 * correct / total
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}, Acc: {acc:.2f}%")
        
        self.trained = True
        
    def predict(self, sequence: np.ndarray) -> SequencePrediction:
        """Predict next action from sequence"""
        if not self.trained:
            raise RuntimeError("Model not trained")
        
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            logits, attn_weights = self.model(x, return_attention=True)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_action = int(probs.argmax())
        
        # Determine kill chain stage based on predicted action
        stage_map = {
            0: 0, 1: 0,  # Recon
            2: 3, 3: 3,  # Exploitation
            4: 4,        # Installation
            5: 5,        # C2
            6: 6, 7: 6,  # Actions
            8: 6, 9: 6   # Exfil
        }
        threat_stage = self.KILL_CHAIN_STAGES[stage_map.get(predicted_action, 0)]
        
        return SequencePrediction(
            predicted_action=predicted_action,
            action_probabilities=probs,
            attention_weights=attn_weights.cpu().numpy()[0] if attn_weights is not None else None,
            threat_stage=threat_stage
        )
    
    def save(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "trained": self.trained
        }, path)
        
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.trained = checkpoint["trained"]
