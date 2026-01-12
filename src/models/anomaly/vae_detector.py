"""
Anomaly Detection Models - VAE and Isolation Forest
"""
import numpy as np
"""
SENTINEL Anomaly Detection Module
=================================

Implements hybrid unsupervised learning for zero-day threat detection.
Combines Variational Autoencoders (VAE) with Isolation Forests.

Models:
    VariationalAutoEncoder: Deep generative model for reconstruction error
    HybridAnomalyDetector: Ensemble of VAE and Isolation Forest

Author: xorinf
Version: 1.0.0
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class AnomalyResult:
    """Result from anomaly detection"""
    is_anomaly: bool
    anomaly_score: float
    reconstruction_error: Optional[float] = None
    isolation_score: Optional[float] = None
    feature_contributions: Optional[Dict[str, float]] = None


class VariationalAutoEncoder(nn.Module):
    """
    Variational Autoencoder for anomaly detection.
    Learns the distribution of normal behavior, flags high reconstruction errors.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dims: list = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        prev_dim = latent_dim
        for h_dim in hidden_dims_rev:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims_rev[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for backprop through sampling"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    def loss_function(self, x: torch.Tensor, recon_x: torch.Tensor, 
                      mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """VAE loss = Reconstruction + KL Divergence"""
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_loss
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Get per-sample reconstruction error (anomaly score)"""
        self.eval()
        with torch.no_grad():
            recon_x, _, _ = self(x)
            error = torch.mean((x - recon_x) ** 2, dim=1)
        return error


class HybridAnomalyDetector:
    """
    Combines VAE and Isolation Forest for robust anomaly detection.
    - VAE captures complex patterns in normal behavior
    - Isolation Forest catches point anomalies
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 16,
                 contamination: float = 0.01, threshold_percentile: float = 95):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.contamination = contamination
        self.threshold_percentile = threshold_percentile
        
        # Models
        self.vae = VariationalAutoEncoder(input_dim, latent_dim)
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200
        )
        
        # Thresholds (learned during training)
        self.vae_threshold: float = 0.0
        self.trained: bool = False
        
    def train(self, X: np.ndarray, epochs: int = 50, batch_size: int = 64,
              learning_rate: float = 1e-3) -> Dict[str, list]:
        """Train both models on normal data"""
        logger.info(f"Training hybrid anomaly detector on {len(X)} samples")
        
        # Train VAE
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)
        
        losses = []
        self.vae.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in loader:
                x = batch[0]
                optimizer.zero_grad()
                recon_x, mu, log_var = self.vae(x)
                loss = self.vae.loss_function(x, recon_x, mu, log_var)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Calculate VAE threshold
        self.vae.eval()
        recon_errors = self.vae.get_reconstruction_error(X_tensor).numpy()
        self.vae_threshold = np.percentile(recon_errors, self.threshold_percentile)
        logger.info(f"VAE threshold set to {self.vae_threshold:.4f}")
        
        # Train Isolation Forest
        self.isolation_forest.fit(X)
        logger.info("Isolation Forest trained")
        
        self.trained = True
        return {"vae_losses": losses, "vae_threshold": self.vae_threshold}
    
    def predict(self, x: np.ndarray, feature_names: list = None) -> AnomalyResult:
        """Predict if sample is anomaly"""
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        x_tensor = torch.FloatTensor(x.reshape(1, -1))
        
        # VAE reconstruction error
        recon_error = self.vae.get_reconstruction_error(x_tensor).item()
        
        # Isolation Forest score
        iso_score = -self.isolation_forest.score_samples(x.reshape(1, -1))[0]
        
        # Combined decision
        vae_anomaly = recon_error > self.vae_threshold
        iso_anomaly = self.isolation_forest.predict(x.reshape(1, -1))[0] == -1
        
        is_anomaly = vae_anomaly or iso_anomaly
        anomaly_score = (recon_error / self.vae_threshold + iso_score) / 2
        
        # Feature contributions (for explainability)
        contributions = None
        if feature_names and is_anomaly:
            self.vae.eval()
            with torch.no_grad():
                recon_x, _, _ = self.vae(x_tensor)
                diff = (x_tensor - recon_x).abs().squeeze().numpy()
                contributions = {
                    name: float(diff[i]) 
                    for i, name in enumerate(feature_names)
                }
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=float(anomaly_score),
            reconstruction_error=recon_error,
            isolation_score=iso_score,
            feature_contributions=contributions
        )
    
    def save(self, path: str):
        """Save model state"""
        import pickle
        state = {
            "vae_state": self.vae.state_dict(),
            "vae_threshold": self.vae_threshold,
            "isolation_forest": self.isolation_forest,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "trained": self.trained
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model state"""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.vae.load_state_dict(state["vae_state"])
        self.vae_threshold = state["vae_threshold"]
        self.isolation_forest = state["isolation_forest"]
        self.trained = state["trained"]
        logger.info(f"Model loaded from {path}")
