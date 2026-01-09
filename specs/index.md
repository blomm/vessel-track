# Vessel Track - Specification Index

This document is the starting point for understanding the Vessel Track project.

## Project Overview

**Vessel Track** is an AI-powered full-stack web application for tracking LNG (Liquefied Natural Gas) tanker vessels and predicting their destinations in real-time.

**Status**: Implementation-ready with complete specifications across 6 development phases

## What Makes This Project Unique

### AI-Powered Predictions
- **Hybrid Approach**: Combines traditional algorithms, RAG (Retrieval-Augmented Generation), and GPT-4o
- **Continuous Learning**: System improves from prediction outcomes
- **Explainable AI**: Natural language reasoning for every prediction

### Full-Stack Architecture
- **Backend**: FastAPI (Python) with PostgreSQL + pgvector for vector similarity search
- **Frontend**: Next.js 15 (React 19) with TypeScript and Mapbox GL for visualization
- **Real-time**: WebSocket integration for live updates

### Production-Ready Features
- REST API with OpenAPI documentation
- Database migrations with Alembic
- Comprehensive error handling
- Admin metrics and monitoring
- Slack notifications for high-confidence predictions

## Specification Files

### Getting Started
- [setup.md](setup.md) - Complete installation guide for backend and frontend

### Technical Documentation
- [architecture.md](architecture.md) - System architecture, data flow, and component design
- [tech-stack.md](tech-stack.md) - Technology choices and rationale
- [features.md](features.md) - Complete feature list with implementation status

### Implementation Guides
- [tasks/README.md](tasks/README.md) - Implementation phases overview
- [tasks/phase1-foundation-database.md](tasks/phase1-foundation-database.md) - Database setup (Days 1-4)
- [tasks/phase2-core-services.md](tasks/phase2-core-services.md) - Prediction engine (Days 5-12)
- [tasks/phase3-ai-integration.md](tasks/phase3-ai-integration.md) - GPT-4o integration (Days 13-17)
- [tasks/phase4-learning-system.md](tasks/phase4-learning-system.md) - Feedback loop (Days 18-21)
- [tasks/phase5-api-websocket.md](tasks/phase5-api-websocket.md) - API and WebSocket (Days 22-26)
- [tasks/phase6-frontend-integration.md](tasks/phase6-frontend-integration.md) - Frontend connection (Days 27-28)

## Quick Start

### Prerequisites
- Node.js 18+, Python 3.11+, Docker, Poetry
- Mapbox API token (free)
- OpenAI API key

### Backend Setup
```bash
cd service
poetry install
docker-compose up -d
poetry run alembic upgrade head
poetry run python scripts/seed_terminals.py
poetry run uvicorn src.main:app --reload
```

### Frontend Setup
```bash
cd app
npm install
# Add NEXT_PUBLIC_MAPBOX_TOKEN to .env.local
npm run dev
```

Visit:
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

## System Capabilities

### Core Features
1. **Interactive Map Visualization**
   - Real-time vessel tracking on Mapbox
   - 15 major LNG terminals worldwide
   - WebSocket live updates

2. **AI-Powered Prediction Engine**
   - Traditional algorithm (proximity, speed, heading)
   - RAG with pgvector for historical patterns
   - GPT-4o for holistic analysis
   - Confidence scores with natural language explanations

3. **Continuous Learning**
   - Automatic outcome detection
   - Accuracy calculation
   - Terminal approach behavior learning
   - Vector embeddings for future predictions

4. **REST API**
   - Vessel CRUD operations
   - Prediction management
   - Admin metrics and health checks
   - OpenAPI/Swagger documentation

5. **Real-time Communication**
   - WebSocket server for live updates
   - Subscription-based filtering
   - Auto-reconnect with exponential backoff

6. **Notifications**
   - Slack integration for high-confidence predictions
   - Detailed prediction summaries
   - AI reasoning included

## Technology Stack

### Frontend
- Next.js 15, React 19, TypeScript
- Mapbox GL JS for mapping
- WebSocket API for real-time updates

### Backend
- FastAPI (async Python web framework)
- PostgreSQL 16 with pgvector extension
- Redis for caching
- SQLAlchemy 2.0 (async)
- Alembic for migrations

### AI/ML
- OpenAI GPT-4o for prediction analysis
- text-embedding-3-small for RAG embeddings
- pgvector for similarity search

### Infrastructure
- Docker & Docker Compose
- Uvicorn ASGI server
- Poetry for Python dependencies

## Database Schema

8 core tables:
1. **vessels** - Current vessel state
2. **terminals** - LNG terminal locations
3. **vessel_journeys** - Historical journey records
4. **predictions** - Prediction records with scores
5. **terminal_approach_behaviors** - Learned patterns
6. **behavior_events** - Vessel behavior changes
7. **vector_embeddings** - RAG embeddings (pgvector)
8. Supporting indexes for performance

## Implementation Phases

All phases are fully documented with step-by-step instructions:

- ✅ **Phase 1**: Foundation & Database (Days 1-4)
- ✅ **Phase 2**: Core Services (Days 5-12)
- ✅ **Phase 3**: AI Integration (Days 13-17)
- ✅ **Phase 4**: Learning System (Days 18-21)
- ✅ **Phase 5**: API & WebSocket (Days 22-26)
- ✅ **Phase 6**: Frontend Integration (Days 27-28)

## Key Innovations

### Hybrid Prediction System
```
Final Score = Traditional Algorithm + RAG Historical Boost + AI Adjustment
            = (0.0-1.0)           + (0.0-0.3)            + (-0.3 to +0.3)
```

### Learning Feedback Loop
1. Vessel arrives at terminal
2. Prediction outcome confirmed
3. Accuracy calculated
4. Embedding generated and stored
5. Future predictions benefit from learned patterns

### Explainable AI
Every prediction includes:
- Confidence percentage
- Score breakdown by component
- AI natural language reasoning
- Key factors that influenced the decision

## Documentation Standards

Each specification file includes:
- Clear sections with descriptive headers
- Code examples where applicable
- Step-by-step instructions
- Verification checklists
- Troubleshooting guidance

## Instructions for AI Assistants

When working on this project:

1. **Start Here**: Read [architecture.md](architecture.md) for system overview
2. **Understand Features**: Review [features.md](features.md) for capabilities
3. **Follow Implementation**: Use phase documents in [tasks/](tasks/) for step-by-step guidance
4. **Reference Tech Stack**: Check [tech-stack.md](tech-stack.md) for technology decisions
5. **Setup Instructions**: Follow [setup.md](setup.md) for environment configuration

### Development Guidelines
- Follow existing patterns in phase documentation
- Update specs when making significant changes
- Test both backend and frontend integration
- Verify predictions display correctly in UI
- Check that WebSocket updates work

### Code Organization
- Backend: `service/src/` with clear separation of concerns
- Frontend: `app/` with components, lib, and app router
- Database: Models in `service/src/database/models.py`
- API: Routers in `service/src/api/routers/`
- Services: Business logic in `service/src/services/`

## Current Implementation

The specifications define a complete, production-ready system with:

- **8 database tables** with proper relationships and indexes
- **5+ REST API endpoints** with full CRUD operations
- **9 core services** (prediction, AI, RAG, learning, WebSocket, etc.)
- **Real-time WebSocket** communication
- **Comprehensive error handling** and logging
- **Admin metrics** and monitoring
- **Continuous learning** from prediction outcomes

## Next Steps

For new developers:
1. Read [setup.md](setup.md) to get the system running
2. Review [architecture.md](architecture.md) to understand the design
3. Explore [features.md](features.md) to see what's implemented
4. Follow phase documents in [tasks/](tasks/) for implementation details
5. Test the system end-to-end with test vessels

For extending the system:
- See "Planned Enhancements" in [features.md](features.md)
- Consider AIS data integration for real vessel tracking
- Add historical track visualization
- Implement filtering and search
- Build analytics dashboard
