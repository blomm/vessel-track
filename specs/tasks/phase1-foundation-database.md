# Phase 1: Foundation & Database

**Duration**: Days 1-4
**Goal**: Set up Python/FastAPI project structure, Docker environment, and database models

---

## 1.1. Project Structure Setup

Create complete `service/` directory at project root:

```bash
cd /Users/michaelblom/vessel-track
mkdir -p service/src/{api/routers,database,services,schemas,utils}
mkdir -p service/{tests/test_services,scripts}
```

### Directory Structure

```
service/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry
│   ├── config.py            # Configuration
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py    # DB connection
│   │   ├── models.py        # SQLAlchemy models
│   │   └── migrations/      # Alembic (created by alembic init)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py          # Dependencies
│   │   └── routers/
│   │       ├── __init__.py
│   │       ├── vessels.py
│   │       ├── terminals.py
│   │       ├── predictions.py
│   │       ├── admin.py
│   │       └── websocket.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── prediction_engine.py
│   │   ├── ai_service.py
│   │   ├── embedding_service.py
│   │   ├── rag_service.py
│   │   ├── learning_service.py
│   │   ├── slack_service.py
│   │   └── websocket_manager.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── vessel.py
│   │   ├── terminal.py
│   │   ├── prediction.py
│   │   └── websocket.py
│   └── utils/
│       ├── __init__.py
│       ├── geo.py           # Haversine, bearing
│       ├── logger.py
│       └── constants.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_services/
├── scripts/
│   ├── seed_terminals.py
│   ├── init_db.py
│   └── backfill_embeddings.py
├── alembic.ini              # Created by alembic init
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

---

## 1.2. Docker Setup

### Create `service/docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: vessel_track_db
    environment:
      POSTGRES_DB: vessel_track
      POSTGRES_USER: vessel_user
      POSTGRES_PASSWORD: vessel_pass_dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/01_init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U vessel_user -d vessel_track"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: vessel_track_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  redis_data:
```

### Create `service/scripts/init_db.sql`:

```sql
-- Enable pgvector extension for vector embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Optional: Enable PostGIS for advanced geographic calculations
CREATE EXTENSION IF NOT EXISTS postgis;
```

### Create `service/.gitignore`:

```
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
.env
.env.local
*.log
.pytest_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
.DS_Store
alembic/versions/*.py
!alembic/versions/__init__.py
```

---

## 1.3. Python Dependencies

### Create `service/pyproject.toml`:

```toml
[tool.poetry]
name = "vessel-track-service"
version = "0.1.0"
description = "AI-powered LNG vessel destination prediction"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.109.0"
uvicorn = {extras = ["standard"], version = "^0.27.0"}
sqlalchemy = {extras = ["asyncio"], version = "^2.0.25"}
asyncpg = "^0.29.0"
alembic = "^1.13.1"
pydantic = {extras = ["email"], version = "^2.5.3"}
pydantic-settings = "^2.1.0"
python-dotenv = "^1.0.0"
openai = "^1.10.0"
pgvector = "^0.2.4"
redis = "^5.0.1"
httpx = "^0.26.0"
websockets = "^12.0"
python-multipart = "^0.0.6"
geopy = "^2.4.1"
numpy = "^1.26.3"

[tool.poetry.dev-dependencies]
pytest = "^7.4.4"
pytest-asyncio = "^0.23.3"
pytest-cov = "^4.1.0"
black = "^24.1.1"
ruff = "^0.1.13"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### Install dependencies:

```bash
cd service
poetry install
```

---

## 1.4. Configuration

### Create `service/src/config.py`:

```python
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

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### Create `service/.env.example`:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://vessel_user:vessel_pass@localhost:5432/vessel_track

# Redis
REDIS_URL=redis://localhost:6379/0

# OpenAI API
OPENAI_API_KEY=sk-your-openai-api-key-here

# Slack Notifications (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Application
ENVIRONMENT=development
DEBUG=true
```

### Create actual `.env` file (not committed to git):

```bash
cp .env.example .env
# Edit .env with your actual credentials
```

---

## 1.5. Database Models

### Create `service/src/database/models.py`:

```python
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Index, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Vessel(Base):
    __tablename__ = "vessels"

    id = Column(String(50), primary_key=True)  # e.g., 'lng-001'
    name = Column(String(255), nullable=False, index=True)
    current_lat = Column(Float, nullable=False)
    current_lon = Column(Float, nullable=False)
    heading = Column(Float)  # 0-360 degrees
    speed = Column(Float)  # knots
    vessel_type = Column(String(50), default='lng_tanker')
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    mmsi = Column(String(20), unique=True, index=True, nullable=True)
    imo = Column(String(20), unique=True, index=True, nullable=True)
    status = Column(String(50))  # 'underway', 'at_anchor', 'moored'
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    journeys = relationship("VesselJourney", back_populates="vessel")
    predictions = relationship("Prediction", back_populates="vessel")
    behavior_events = relationship("BehaviorEvent", back_populates="vessel")


class Terminal(Base):
    __tablename__ = "terminals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    code = Column(String(10), unique=True, index=True)  # e.g., 'SABINE'
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    country = Column(String(100))
    region = Column(String(100))
    capacity_bcm_year = Column(Float)  # Billion cubic meters per year
    terminal_type = Column(String(50))  # 'export' or 'import'
    approach_zone_radius_km = Column(Float, default=50.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    destination_journeys = relationship("VesselJourney", foreign_keys="VesselJourney.destination_terminal_id", back_populates="destination")
    predictions = relationship("Prediction", back_populates="terminal")
    approach_behaviors = relationship("TerminalApproachBehavior", back_populates="terminal")


class VesselJourney(Base):
    __tablename__ = "vessel_journeys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vessel_id = Column(String(50), ForeignKey('vessels.id'), nullable=False, index=True)
    origin_terminal_id = Column(Integer, ForeignKey('terminals.id'), nullable=True)
    destination_terminal_id = Column(Integer, ForeignKey('terminals.id'), nullable=False, index=True)
    departure_time = Column(DateTime(timezone=True))
    arrival_time = Column(DateTime(timezone=True))
    duration_hours = Column(Float)
    avg_speed = Column(Float)
    distance_nm = Column(Float)  # Nautical miles
    route_polyline = Column(Text)  # JSON array of [lat, lon] points
    completed = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    vessel = relationship("Vessel", back_populates="journeys")
    origin = relationship("Terminal", foreign_keys=[origin_terminal_id])
    destination = relationship("Terminal", foreign_keys=[destination_terminal_id], back_populates="destination_journeys")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vessel_id = Column(String(50), ForeignKey('vessels.id'), nullable=False, index=True)
    terminal_id = Column(Integer, ForeignKey('terminals.id'), nullable=False, index=True)
    prediction_time = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    vessel_lat = Column(Float, nullable=False)
    vessel_lon = Column(Float, nullable=False)
    vessel_speed = Column(Float)
    vessel_heading = Column(Float)

    # Prediction metrics
    confidence_score = Column(Float)  # 0.0 - 1.0
    distance_to_terminal_km = Column(Float)
    eta_hours = Column(Float)
    predicted_arrival = Column(DateTime(timezone=True))

    # Score breakdown
    proximity_score = Column(Float)
    speed_score = Column(Float)
    heading_score = Column(Float)
    historical_similarity_score = Column(Float)
    ai_confidence_adjustment = Column(Float)  # -0.3 to +0.3
    ai_reasoning = Column(Text)  # GPT-4o explanation

    # Outcome tracking
    status = Column(String(50), default='active', index=True)  # active/confirmed/incorrect/expired
    actual_arrival_time = Column(DateTime(timezone=True), nullable=True)
    accuracy_score = Column(Float, nullable=True)

    slack_notification_sent = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    vessel = relationship("Vessel", back_populates="predictions")
    terminal = relationship("Terminal", back_populates="predictions")


class TerminalApproachBehavior(Base):
    __tablename__ = "terminal_approach_behaviors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    terminal_id = Column(Integer, ForeignKey('terminals.id'), nullable=False, index=True)
    vessel_id = Column(String(50), ForeignKey('vessels.id'), nullable=True, index=True)

    # Pattern data
    approach_distance_km = Column(Float)
    typical_speed_range_min = Column(Float)
    typical_speed_range_max = Column(Float)
    typical_heading_range_min = Column(Float)
    typical_heading_range_max = Column(Float)
    approach_angle_degrees = Column(Float)

    # Statistics
    observation_count = Column(Integer, default=1)
    last_observed = Column(DateTime(timezone=True), server_default=func.now())
    confidence = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    terminal = relationship("Terminal", back_populates="approach_behaviors")
    vessel = relationship("Vessel")


class BehaviorEvent(Base):
    __tablename__ = "behavior_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vessel_id = Column(String(50), ForeignKey('vessels.id'), nullable=False, index=True)
    event_time = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)

    # Event details
    event_type = Column(String(50), index=True)  # speed_change/course_change/stop/resume
    speed_before = Column(Float)
    speed_after = Column(Float)
    heading_before = Column(Float)
    heading_after = Column(Float)
    magnitude = Column(Float)

    # Context
    nearest_terminal_id = Column(Integer, ForeignKey('terminals.id'), nullable=True)
    distance_to_terminal_km = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    vessel = relationship("Vessel", back_populates="behavior_events")
    nearest_terminal = relationship("Terminal")


class VectorEmbedding(Base):
    __tablename__ = "vector_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content_type = Column(String(50), nullable=False, index=True)  # journey/prediction/behavior
    content_id = Column(Integer, nullable=False, index=True)
    text_content = Column(Text, nullable=False)
    embedding = Column(Vector(1536))  # OpenAI text-embedding-3-small dimension
    metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# Create indexes
Index('ix_vessels_location', Vessel.current_lat, Vessel.current_lon)
Index('ix_predictions_active', Prediction.vessel_id, Prediction.status)
Index('ix_journeys_completed', VesselJourney.destination_terminal_id, VesselJourney.completed)
Index('ix_behavior_events_vessel_time', BehaviorEvent.vessel_id, BehaviorEvent.event_time.desc())
Index('ix_content_lookup', VectorEmbedding.content_type, VectorEmbedding.content_id)
```

### Create `service/src/database/connection.py`:

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from src.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_db():
    """Dependency for getting database sessions"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

---

## 1.6. Database Migrations (Alembic)

### Initialize Alembic:

```bash
cd service
poetry run alembic init alembic
```

### Update `service/alembic.ini`:

Find the line with `sqlalchemy.url` and replace with:
```ini
# sqlalchemy.url = driver://user:pass@localhost/dbname
# Leave commented - we'll set it in env.py
```

### Update `service/alembic/env.py`:

```python
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.database.models import Base

# this is the Alembic Config object
config = context.config

# Set sqlalchemy.url from settings
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target_metadata for autogenerate
target_metadata = Base.metadata

# ... rest of default alembic env.py code
```

### Create initial migration:

```bash
poetry run alembic revision --autogenerate -m "Initial schema"
poetry run alembic upgrade head
```

---

## 1.7. Terminal Seed Data

### Create `service/scripts/seed_terminals.py`:

```python
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.connection import AsyncSessionLocal
from src.database.models import Terminal

TERMINALS = [
    # USA Export Terminals
    {
        "name": "Sabine Pass LNG", "code": "SABINE",
        "lat": 29.7294, "lon": -93.8767,
        "country": "USA", "region": "Gulf of Mexico",
        "capacity_bcm_year": 27.0, "terminal_type": "export",
        "approach_zone_radius_km": 50.0
    },
    {
        "name": "Cameron LNG", "code": "CAMERON",
        "lat": 29.7965, "lon": -93.3191,
        "country": "USA", "region": "Gulf of Mexico",
        "capacity_bcm_year": 15.0, "terminal_type": "export",
        "approach_zone_radius_km": 50.0
    },
    {
        "name": "Freeport LNG", "code": "FREEPORT",
        "lat": 28.9450, "lon": -95.3028,
        "country": "USA", "region": "Gulf of Mexico",
        "capacity_bcm_year": 15.0, "terminal_type": "export",
        "approach_zone_radius_km": 50.0
    },

    # Asia Import Terminals
    {
        "name": "Tokyo Gas Negishi", "code": "NEGISHI",
        "lat": 35.4222, "lon": 139.6456,
        "country": "Japan", "region": "East Asia",
        "capacity_bcm_year": 9.0, "terminal_type": "import",
        "approach_zone_radius_km": 40.0
    },
    {
        "name": "Incheon LNG Terminal", "code": "INCHEON",
        "lat": 37.4563, "lon": 126.7052,
        "country": "South Korea", "region": "East Asia",
        "capacity_bcm_year": 12.0, "terminal_type": "import",
        "approach_zone_radius_km": 40.0
    },
    {
        "name": "Guangdong Dapeng LNG", "code": "DAPENG",
        "lat": 22.6444, "lon": 114.4906,
        "country": "China", "region": "East Asia",
        "capacity_bcm_year": 8.0, "terminal_type": "import",
        "approach_zone_radius_km": 40.0
    },

    # Europe Import Terminals
    {
        "name": "Gate Terminal Rotterdam", "code": "GATE",
        "lat": 51.9497, "lon": 4.0342,
        "country": "Netherlands", "region": "Northwest Europe",
        "capacity_bcm_year": 12.0, "terminal_type": "import",
        "approach_zone_radius_km": 35.0
    },
    {
        "name": "Zeebrugge LNG Terminal", "code": "ZEEBRUGGE",
        "lat": 51.3356, "lon": 3.2006,
        "country": "Belgium", "region": "Northwest Europe",
        "capacity_bcm_year": 9.0, "terminal_type": "import",
        "approach_zone_radius_km": 35.0
    },
    {
        "name": "Montoir-de-Bretagne", "code": "MONTOIR",
        "lat": 47.3114, "lon": -2.1428,
        "country": "France", "region": "Northwest Europe",
        "capacity_bcm_year": 10.0, "terminal_type": "import",
        "approach_zone_radius_km": 35.0
    },

    # Middle East Export
    {
        "name": "Ras Laffan LNG", "code": "RASLAFFAN",
        "lat": 25.9231, "lon": 51.5453,
        "country": "Qatar", "region": "Middle East",
        "capacity_bcm_year": 77.0, "terminal_type": "export",
        "approach_zone_radius_km": 50.0
    },

    # Australia Export
    {
        "name": "Gorgon LNG", "code": "GORGON",
        "lat": -20.6167, "lon": 115.0500,
        "country": "Australia", "region": "Pacific",
        "capacity_bcm_year": 15.6, "terminal_type": "export",
        "approach_zone_radius_km": 50.0
    },
    {
        "name": "Queensland Curtis LNG", "code": "QCLNG",
        "lat": -23.8500, "lon": 151.2667,
        "country": "Australia", "region": "Pacific",
        "capacity_bcm_year": 8.5, "terminal_type": "export",
        "approach_zone_radius_km": 50.0
    },

    # South Asia Import
    {
        "name": "Dahej LNG Terminal", "code": "DAHEJ",
        "lat": 21.7000, "lon": 72.6000,
        "country": "India", "region": "South Asia",
        "capacity_bcm_year": 17.5, "terminal_type": "import",
        "approach_zone_radius_km": 45.0
    },

    # Latin America Import
    {
        "name": "Guanabara Bay LNG", "code": "GUANABARA",
        "lat": -22.9068, "lon": -43.1729,
        "country": "Brazil", "region": "South America",
        "capacity_bcm_year": 7.0, "terminal_type": "import",
        "approach_zone_radius_km": 40.0
    },

    # UK Import
    {
        "name": "South Hook LNG", "code": "SOUTHHOOK",
        "lat": 51.7103, "lon": -5.1636,
        "country": "United Kingdom", "region": "Northwest Europe",
        "capacity_bcm_year": 21.0, "terminal_type": "import",
        "approach_zone_radius_km": 35.0
    },
]

async def seed_terminals():
    async with AsyncSessionLocal() as session:
        for terminal_data in TERMINALS:
            # Check if already exists
            existing = await session.get(Terminal, terminal_data["code"])
            if not existing:
                terminal = Terminal(**terminal_data)
                session.add(terminal)
                print(f"Added terminal: {terminal_data['name']}")
            else:
                print(f"Terminal already exists: {terminal_data['name']}")

        await session.commit()
    print(f"✓ Seeded {len(TERMINALS)} terminals")

if __name__ == "__main__":
    asyncio.run(seed_terminals())
```

---

## Verification Checklist

- [ ] `service/` directory structure created
- [ ] Docker Compose configured with PostgreSQL + pgvector + Redis
- [ ] `docker-compose up -d` starts successfully
- [ ] Poetry dependencies installed
- [ ] Configuration (`config.py`) loads environment variables
- [ ] Database models defined (8 tables)
- [ ] Alembic initialized and migrations created
- [ ] `alembic upgrade head` runs successfully
- [ ] 15 LNG terminals seeded into database
- [ ] Can connect to PostgreSQL: `psql -h localhost -U vessel_user -d vessel_track`
- [ ] pgvector extension enabled: `SELECT * FROM pg_extension WHERE extname = 'vector';`

---

## Next Steps

Once this phase is complete, move to **Phase 2: Core Services** where we'll implement:
- Geographic utility functions (Haversine distance, bearing calculations)
- Traditional prediction engine (proximity, speed, heading scoring)
- Embedding service for RAG
- RAG service for similarity search
