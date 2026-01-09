# Technology Stack

## Frontend Stack

### Framework
- **Next.js 15** with App Router
- **React 19** for UI components
- **TypeScript** for type safety

### Styling
- **Tailwind CSS** for utility-first styling
- Dark theme support built-in

### Mapping & Visualization
- **Mapbox GL JS** (v3.x) - Vector-based mapping library
- **@types/mapbox-gl** - TypeScript definitions

**Why Mapbox?**
- High performance with vector tiles
- Excellent customization options
- Good support for real-time updates
- Beautiful built-in styles (using dark-v11 theme)
- Interactive markers and popups

### Real-time Communication
- **WebSocket API** (native browser WebSocket)
- Auto-reconnect with exponential backoff
- Subscription-based filtering

### Development Tools
- **ESLint** for code linting
- **TypeScript** compiler for type checking
- npm for package management

## Backend Stack

### Web Framework
- **FastAPI** (async Python web framework)
  - OpenAPI/Swagger documentation
  - Pydantic data validation
  - CORS middleware
  - WebSocket support

### Database
- **PostgreSQL 16** - Primary relational database
  - **pgvector** extension for vector similarity search
  - Async queries with asyncpg driver
  - Connection pooling (10 base + 20 overflow)

### ORM & Migrations
- **SQLAlchemy 2.0** (async mode)
  - Declarative models
  - Relationship mapping
  - Query building
- **Alembic** for database migrations
  - Version control for schema changes
  - Auto-generate migrations from models

### AI & Machine Learning
- **OpenAI API**
  - **GPT-4o** for intelligent prediction analysis
  - **text-embedding-3-small** for RAG embeddings (1536 dimensions)
- **pgvector** for vector similarity search
  - Cosine distance calculations
  - Efficient nearest-neighbor queries

### Caching
- **Redis 7** (Alpine)
  - Session caching
  - Prediction result caching (5 min TTL)
  - Rate limiting

### Data Validation
- **Pydantic** v2
  - Request/response schemas
  - Settings management
  - Data serialization

### HTTP Client
- **httpx** (async HTTP client)
  - Slack webhook integration
  - External API calls

### Utilities
- **geopy** - Geographic calculations (backup to custom Haversine)
- **numpy** - Numerical computations
- **python-dotenv** - Environment variable loading

### Testing
- **pytest** - Test framework
- **pytest-asyncio** - Async test support
- **pytest-cov** - Coverage reporting

### Code Quality
- **black** - Code formatting
- **ruff** - Fast Python linter

### Dependency Management
- **Poetry** - Python dependency management and packaging
  - Lock file for reproducible builds
  - Virtual environment management

## Infrastructure & DevOps

### Containerization
- **Docker** & **Docker Compose**
  - PostgreSQL with pgvector
  - Redis
  - Service orchestration

### Web Server
- **Uvicorn** - ASGI server
  - Auto-reload in development
  - Production-ready performance
  - WebSocket support

## External Services

### APIs
- **OpenAI API**
  - Model: GPT-4o (`gpt-4o`)
  - Embeddings: `text-embedding-3-small`
  - Temperature: 0.3 (deterministic)
  - Max tokens: 1000

### Notifications
- **Slack** (optional)
  - Incoming webhooks
  - Rich message formatting
  - Confidence-based alerts (≥80%)

## Environment Variables

### Frontend (`.env.local`)
```bash
NEXT_PUBLIC_MAPBOX_TOKEN=        # Mapbox API token
NEXT_PUBLIC_API_URL=             # Backend API URL (http://localhost:8000/api/v1)
NEXT_PUBLIC_WS_URL=              # WebSocket URL (ws://localhost:8000/ws/vessels)
```

### Backend (`service/.env`)
```bash
# Database
DATABASE_URL=                     # PostgreSQL connection string
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# Redis
REDIS_URL=redis://localhost:6379/0

# OpenAI
OPENAI_API_KEY=                   # OpenAI API key
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_MAX_TOKENS=1000
OPENAI_TEMPERATURE=0.3

# Slack (optional)
SLACK_WEBHOOK_URL=                # Slack incoming webhook URL
SLACK_NOTIFICATION_THRESHOLD=0.80

# Application
ENVIRONMENT=development
DEBUG=true
API_V1_PREFIX=/api/v1

# Prediction Settings
PREDICTION_MAX_DISTANCE_KM=500.0
PREDICTION_MIN_CONFIDENCE=0.50
PREDICTION_UPDATE_INTERVAL_SECONDS=300

# RAG Settings
RAG_SIMILARITY_THRESHOLD=0.30
RAG_MAX_RESULTS=5
EMBEDDING_DIMENSION=1536

# WebSocket
WS_HEARTBEAT_INTERVAL=30
```

## Project Structure

```
vessel-track/
├── app/                          # Next.js frontend
│   ├── app/                      # App router pages
│   ├── components/               # React components
│   ├── lib/                      # Utilities
│   │   ├── api-client.ts        # Backend API client
│   │   └── websocket.ts         # WebSocket client
│   ├── data/                     # Mock data (legacy)
│   ├── public/                   # Static assets
│   ├── .env.local               # Frontend environment variables
│   ├── package.json
│   ├── tsconfig.json
│   └── tailwind.config.ts
│
├── service/                      # FastAPI backend
│   ├── src/
│   │   ├── main.py              # FastAPI app entry
│   │   ├── config.py            # Configuration
│   │   ├── database/            # Database layer
│   │   ├── api/                 # API routes
│   │   ├── services/            # Business logic
│   │   ├── schemas/             # Pydantic schemas
│   │   └── utils/               # Utilities
│   ├── alembic/                 # Database migrations
│   ├── scripts/                 # Seed scripts
│   ├── tests/                   # Test suite
│   ├── docker-compose.yml       # Docker services
│   ├── Dockerfile
│   ├── pyproject.toml           # Poetry dependencies
│   └── .env                     # Backend environment variables
│
└── specs/                        # Documentation
    ├── index.md
    ├── architecture.md
    ├── features.md
    ├── tech-stack.md (this file)
    ├── setup.md
    └── tasks/                    # Implementation phases
        ├── phase1-foundation-database.md
        ├── phase2-core-services.md
        ├── phase3-ai-integration.md
        ├── phase4-learning-system.md
        ├── phase5-api-websocket.md
        └── phase6-frontend-integration.md
```

## Technology Choices Rationale

### Why FastAPI?
- **Async-first**: Native async/await support for concurrent requests
- **Type safety**: Pydantic integration for robust validation
- **Auto-documentation**: OpenAPI/Swagger UI out of the box
- **Performance**: Comparable to Node.js and Go
- **Modern**: Built on latest Python standards (3.11+)

### Why PostgreSQL + pgvector?
- **Relational + Vector**: Single database for structured and vector data
- **Proven reliability**: Battle-tested database
- **pgvector extension**: Efficient vector similarity search
- **ACID compliance**: Data integrity guarantees
- **Rich ecosystem**: Excellent tooling and community

### Why GPT-4o?
- **Reasoning capability**: Understands complex maritime context
- **Structured output**: JSON mode for reliable parsing
- **Nuanced analysis**: Catches edge cases algorithms miss
- **Natural language**: Provides human-readable explanations
- **Context window**: Large enough for comprehensive analysis

### Why RAG (Retrieval-Augmented Generation)?
- **Historical learning**: Learns from past prediction outcomes
- **Pattern recognition**: Finds similar situations in history
- **Contextual boost**: Improves predictions with relevant historical data
- **Continuous improvement**: Gets better with more data

### Why Next.js?
- **React 19**: Latest React features
- **App Router**: Modern routing and server components
- **TypeScript**: Type safety across the stack
- **Performance**: Automatic optimizations
- **Developer experience**: Hot reload, fast refresh

### Why Mapbox GL JS?
- **Vector tiles**: Smooth rendering and performance
- **Customization**: Full control over map styling
- **Real-time updates**: Efficient marker updates
- **Mobile support**: Responsive on all devices
- **Rich API**: Extensive features for maritime visualization

## Performance Characteristics

### Backend
- **Response time**: <100ms for most endpoints
- **Prediction latency**: 2-5 seconds (includes GPT-4o call)
- **WebSocket latency**: <50ms for broadcasts
- **Database queries**: <10ms (indexed queries)
- **Vector search**: <100ms (pgvector similarity)

### Frontend
- **Initial load**: <2 seconds
- **Map initialization**: <1 second
- **Marker updates**: 60 FPS smooth updates
- **WebSocket reconnect**: <3 seconds

### Scalability
- **Concurrent users**: 100s (current setup)
- **Vessels tracked**: 1000s (with clustering)
- **Predictions/hour**: 10,000+
- **Database**: Scalable to millions of records
