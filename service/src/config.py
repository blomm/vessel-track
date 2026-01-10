from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Vessel Track API"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_MAX_TOKENS: int = 1000
    OPENAI_TEMPERATURE: float = 0.3

    # Slack
    SLACK_WEBHOOK_URL: Optional[str] = None
    SLACK_NOTIFICATION_THRESHOLD: float = 0.80

    # Prediction Settings
    PREDICTION_MAX_DISTANCE_KM: float = 500.0
    PREDICTION_MIN_CONFIDENCE: float = 0.50
    PREDICTION_UPDATE_INTERVAL_SECONDS: int = 300  # 5 minutes

    # RAG Settings
    RAG_SIMILARITY_THRESHOLD: float = 0.30
    RAG_MAX_RESULTS: int = 5
    EMBEDDING_DIMENSION: int = 1536

    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30  # seconds

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_CONSUMER_GROUP: str = "vessel-track-workers"
    KAFKA_AUTO_OFFSET_RESET: str = "latest"
    KAFKA_ENABLE_AUTO_COMMIT: bool = True

    # Kafka Topics
    KAFKA_VESSEL_POSITIONS_TOPIC: str = "vessel-position-updates"
    KAFKA_PREDICTION_REQUESTS_TOPIC: str = "prediction-requests"
    KAFKA_PREDICTION_RESULTS_TOPIC: str = "prediction-results"

    # Kafka Performance
    KAFKA_COMPRESSION_TYPE: str = "gzip"
    KAFKA_BATCH_SIZE: int = 16384
    KAFKA_LINGER_MS: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
