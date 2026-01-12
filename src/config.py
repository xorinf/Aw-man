"""
SENTINEL Configuration Module
=============================

Centralized configuration management using Pydantic Settings.
All settings can be overridden via environment variables or .env file.

Environment Variables:
    API_HOST: Host to bind the API server (default: 0.0.0.0)
    API_PORT: Port for the API server (default: 8000)
    DEBUG: Enable debug mode (default: False)
    REDIS_URL: Redis connection URL for caching
    KAFKA_BROKER: Kafka broker URL for message streaming
    DB_URL: PostgreSQL/TimescaleDB connection URL
    MODEL_PATH: Directory for saved ML models
    ANOMALY_THRESHOLD: Threshold for anomaly detection (0-1)
    OPENAI_API_KEY: Optional API key for LLM-powered red team

Usage:
    from src.config import settings
    print(settings.API_PORT)  # Access configuration

Author: xorinf
Version: 1.0.0
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    This class uses Pydantic's BaseSettings to automatically load
    configuration from environment variables, with sensible defaults
    for local development.
    
    Attributes:
        API_HOST: Host address for the FastAPI server
        API_PORT: Port number for the FastAPI server
        DEBUG: Enable debug mode for development
        REDIS_URL: Connection string for Redis cache
        KAFKA_BROKER: Kafka broker address for streaming
        DB_URL: Database connection URL
        MODEL_PATH: Path to saved ML model files
        ANOMALY_THRESHOLD: Detection threshold (higher = more strict)
        SEQUENCE_WINDOW: Window size for sequence models
        SHAP_SAMPLES: Number of samples for SHAP explanations
        OPENAI_API_KEY: API key for LLM features (optional)
    """
    
    # ===================
    # API Configuration
    # ===================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    
    # ===================
    # Data Stores
    # ===================
    REDIS_URL: str = "redis://localhost:6379"
    KAFKA_BROKER: str = "localhost:9092"
    KAFKA_TOPIC_RAW: str = "sentinel.raw.traffic"
    KAFKA_TOPIC_ALERTS: str = "sentinel.alerts"
    DB_URL: str = "postgresql://sentinel:sentinel@localhost:5432/sentinel_db"
    
    # ===================
    # ML Model Settings
    # ===================
    MODEL_PATH: str = "./models"
    ANOMALY_THRESHOLD: float = 0.85  # 0-1, higher = more strict
    SEQUENCE_WINDOW: int = 50  # Events per sequence
    
    # ===================
    # XAI Settings
    # ===================
    SHAP_SAMPLES: int = 100  # Background samples for SHAP
    
    # ===================
    # Red Team (Optional)
    # ===================
    OPENAI_API_KEY: Optional[str] = None  # For LLM-powered agents
    
    class Config:
        """Pydantic configuration for settings loading."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance - import this in other modules
settings = Settings()
