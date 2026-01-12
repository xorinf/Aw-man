"""
Configuration settings for SENTINEL
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # Kafka
    KAFKA_BROKER: str = "localhost:9092"
    KAFKA_TOPIC_RAW: str = "sentinel.raw.traffic"
    KAFKA_TOPIC_ALERTS: str = "sentinel.alerts"
    
    # Database
    DB_URL: str = "postgresql://sentinel:sentinel@localhost:5432/sentinel_db"
    
    # ML Model Settings
    MODEL_PATH: str = "./models"
    ANOMALY_THRESHOLD: float = 0.85
    SEQUENCE_WINDOW: int = 50
    
    # XAI Settings
    SHAP_SAMPLES: int = 100
    
    # Red Team
    OPENAI_API_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
