# Implementation Tasks - LNG Vessel Destination Prediction System

This folder contains the complete implementation plan broken down into 6 phases.

## Overview

Build an AI-powered LNG vessel destination prediction system with:
- **Backend**: Python + FastAPI + PostgreSQL + pgvector + OpenAI GPT-4o
- **Frontend**: Next.js 15 + TypeScript + Mapbox GL
- **Real-time**: WebSocket for live updates
- **AI/RAG**: GPT-4o for analysis + vector embeddings for historical pattern matching
- **Learning**: Continuous improvement from prediction outcomes

## Implementation Phases

### ✅ Phase 1: Foundation & Database (Days 1-4)
**File**: [phase1-foundation-database.md](./phase1-foundation-database.md)

- Set up Python/FastAPI project structure
- Configure Docker (PostgreSQL + pgvector + Redis)
- Create SQLAlchemy database models (8 tables)
- Initialize Alembic migrations
- Seed 15 LNG terminals worldwide

**Deliverables**:
- `service/` directory with complete structure
- Docker Compose running PostgreSQL + Redis
- Database with schema and terminals

---

### Phase 2: Core Services (Days 5-12)
**File**: [phase2-core-services.md](./phase2-core-services.md)

- Implement geographic utilities (Haversine, bearing calculations)
- Build traditional prediction engine (proximity, speed, heading scoring)
- Create embedding service (OpenAI text-embedding-3-small)
- Implement RAG service (pgvector similarity search)

**Deliverables**:
- Working traditional prediction algorithm
- Embeddings generated and stored
- RAG retrieval from vector database

---

### Phase 3: AI Integration (Days 13-17)
**File**: [phase3-ai-integration.md](./phase3-ai-integration.md)

- Build AI service with GPT-4o integration
- Design comprehensive prompts for analysis
- Complete end-to-end prediction pipeline (traditional + RAG + AI)
- Implement Slack notification service

**Deliverables**:
- GPT-4o provides confidence adjustments + reasoning
- Full prediction pipeline with AI enhancement
- Slack alerts for high-confidence predictions (>80%)

---

### Phase 4: Learning System (Days 18-21)
**File**: [phase4-learning-system.md](./phase4-learning-system.md)

- Implement learning service for prediction outcomes
- Calculate accuracy metrics
- Update approach behavior patterns
- Detect behavior events (speed/course changes)
- Set up background tasks for automatic learning

**Deliverables**:
- Feedback loop: predictions → outcomes → RAG updates
- Continuous improvement from actual results
- Automated outcome detection

---

### Phase 5: API & WebSocket (Days 22-26)
**File**: [phase5-api-websocket.md](./phase5-api-websocket.md)

- Create REST API endpoints (vessels, terminals, predictions, admin)
- Implement WebSocket for real-time updates
- Build WebSocket manager for connection handling
- Add admin endpoints (metrics, health checks)

**Deliverables**:
- Complete REST API with Swagger docs
- WebSocket real-time vessel updates
- Admin dashboard metrics

---

### Phase 6: Frontend Integration (Days 27-28)
**File**: [phase6-frontend-integration.md](./phase6-frontend-integration.md)

- Create Next.js API client
- Implement WebSocket client for real-time updates
- Update map component with predictions display
- Show AI reasoning in vessel popups

**Deliverables**:
- Frontend connected to backend
- Real-time vessel tracking
- Predictions with AI explanations visible on map

---

### ✅ Phase 7: Kafka Event-Driven Architecture (Days 29-33)
**File**: [phase7-kafka-integration.md](./phase7-kafka-integration.md)

- Set up Kafka and Zookeeper infrastructure (Docker Compose)
- Create Kafka topics with proper configuration
- Implement event schemas (Pydantic models)
- Build Kafka producer service for publishing events
- Build Kafka consumer base class
- Implement prediction worker consumer (async processing)
- Update API endpoints to publish events (event-driven)
- Implement fan-out pattern for prediction results

**Deliverables**:
- Event-driven prediction processing
- API response time <50ms (vs 5-10s synchronous)
- Kafka streams for vessel updates and predictions
- Horizontal scalability support (multiple workers)
- Complete event audit trail

---

## How to Use These Tasks

### Sequential Approach (Recommended)
Work through phases 1-7 in order. Each phase builds on the previous one.

```bash
# Start with Phase 1
cd /Users/michaelblom/vessel-track
# Follow instructions in phase1-foundation-database.md

# Once Phase 1 complete, move to Phase 2
# Continue sequentially through Phase 7
```

### Parallel Development (Advanced)
If you have multiple developers:
- **Backend Team**: Phases 1-5, 7 (event-driven architecture)
- **Frontend Team**: Phase 6 (can start once Phase 5 API is ready)

### Verification
Each phase includes:
- **Verification Checklist**: Items to check before moving to next phase
- **Testing Section**: Manual tests to run
- **Next Steps**: What comes next

## Project Structure

```
vessel-track/
├── app/                         # Next.js frontend
│   ├── components/              # Map, UI components
│   ├── lib/                     # API client, WebSocket
│   └── app/                     # Pages
│
└── service/                     # Python/FastAPI backend
    ├── src/
    │   ├── api/                 # REST & WebSocket endpoints
    │   ├── services/            # Prediction engine, AI, RAG, learning
    │   ├── database/            # SQLAlchemy models & migrations
    │   ├── schemas/             # Pydantic models
    │   └── utils/               # Geographic calculations
    ├── tests/                   # Unit & integration tests
    ├── scripts/                 # Database seeding, initialization
    └── docker-compose.yml       # PostgreSQL + Redis
```

## Key Technologies

### Backend
- **FastAPI**: Modern async Python web framework
- **SQLAlchemy**: ORM with async support
- **PostgreSQL + pgvector**: Vector database for RAG
- **OpenAI GPT-4o**: AI analysis and confidence adjustment
- **OpenAI text-embedding-3-small**: Vector embeddings (1536 dimensions)
- **Redis**: Caching and rate limiting
- **WebSockets**: Real-time updates
- **Apache Kafka**: Event streaming for async processing
- **confluent-kafka-python**: Kafka client library

### Frontend
- **Next.js 15**: React framework with App Router
- **TypeScript**: Type-safe development
- **Mapbox GL JS**: Interactive maps
- **WebSocket API**: Real-time backend connection

## Critical Files by Phase

### Phase 1
- `service/src/database/models.py` (8 tables)
- `service/docker-compose.yml` (PostgreSQL + pgvector)
- `service/scripts/seed_terminals.py` (15 terminals)

### Phase 2
- `service/src/utils/geo.py` (Haversine, bearing)
- `service/src/services/prediction_engine.py` (traditional algorithm)
- `service/src/services/embedding_service.py` (OpenAI embeddings)
- `service/src/services/rag_service.py` (vector similarity)

### Phase 3
- `service/src/services/ai_service.py` (GPT-4o integration)
- Prediction pipeline integration (all services combined)

### Phase 4
- `service/src/services/learning_service.py` (feedback loop)
- `service/src/services/outcome_detector.py` (automatic learning)

### Phase 5
- `service/src/main.py` (FastAPI app)
- `service/src/api/routers/` (all endpoint files)
- `service/src/services/websocket_manager.py` (real-time)

### Phase 6
- `app/lib/api-client.ts` (fetch vessels, predictions)
- `app/lib/websocket.ts` (WebSocket client)
- `app/app/page.tsx` (main page with real-time updates)

### Phase 7
- `service/docker-compose.yml` (add Kafka + Zookeeper)
- `service/src/config.py` (Kafka settings)
- `service/src/schemas/events.py` (event Pydantic models)
- `service/src/services/kafka_producer.py` (event publishing)
- `service/src/services/kafka_consumer.py` (base consumer class)
- `service/src/services/consumers/prediction_worker.py` (async prediction processing)
- `service/scripts/create_kafka_topics.sh` (topic creation script)

## Environment Variables

### Backend (`service/.env`)
```bash
DATABASE_URL=postgresql+asyncpg://vessel_user:vessel_pass@localhost:5432/vessel_track
REDIS_URL=redis://localhost:6379/0
OPENAI_API_KEY=sk-your-key-here
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK
```

### Frontend (`app/.env.local`)
```bash
NEXT_PUBLIC_MAPBOX_TOKEN=your-mapbox-token
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/vessels
```

## Success Metrics

- **Accuracy**: >85% of predictions confirmed within 4 hours of ETA
- **Confidence Calibration**: 90% confidence = 90% accuracy rate
- **AI Value**: Predictions with AI adjustment >10% more accurate than base
- **RAG Improvement**: Accuracy increases over time as data accumulates
- **Latency**: Predictions generated in <5 seconds including AI analysis

## Getting Help

If stuck on a phase:
1. Review the main plan: `specs/plans/eager-strolling-lake.md`
2. Check verification checklist in current phase
3. Review testing section for manual verification
4. Check backend logs: `docker-compose logs -f`
5. Check API docs: http://localhost:8000/docs

## Notes

- Each phase is designed to be completable independently
- All code examples are production-ready
- Security best practices are followed
- Error handling and logging included
- Type safety enforced (Python type hints + TypeScript)

---

**Start with Phase 1 and work sequentially for best results!**
