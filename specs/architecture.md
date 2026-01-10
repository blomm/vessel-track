# Architecture

## Overview
Vessel Track is a full-stack AI-powered application for tracking LNG tanker vessels and predicting their destinations. It combines a Next.js frontend with a FastAPI backend, leveraging GPT-4o for intelligent prediction analysis.

## System Architecture

### High-Level Architecture (Event-Driven)
```
┌─────────────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Map Display │  │  WebSocket   │  │  API Client  │              │
│  │   (Mapbox)   │  │   Client     │  │              │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└────────────────┬────────────────┬───────────────────────────────────┘
                 │ HTTP           │ WebSocket
                 ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Backend (FastAPI)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  REST API    │  │  WebSocket   │  │  Background  │              │
│  │  Endpoints   │  │   Manager    │  │    Tasks     │              │
│  └──────┬───────┘  └──────▲───────┘  └──────────────┘              │
│         │                  │                                         │
│         │ publish          │ subscribe                               │
│         ▼                  │                                         │
│  ┌─────────────────────────────────────────────────┐                │
│  │           Apache Kafka Event Broker              │                │
│  │  ┌──────────────────┐  ┌──────────────────┐    │                │
│  │  │ vessel-position- │  │ prediction-      │    │                │
│  │  │    updates       │  │   requests       │    │                │
│  │  └──────────────────┘  └──────────────────┘    │                │
│  │  ┌──────────────────┐                          │                │
│  │  │ prediction-      │                          │                │
│  │  │   results        │                          │                │
│  │  └──────────────────┘                          │                │
│  └─────────────────────────────────────────────────┘                │
│         │                  │                  │                     │
│         ▼                  ▼                  ▼                     │
│  ┌─────────────┐    ┌─────────────┐   ┌─────────────┐             │
│  │  Position   │    │ Prediction  │   │  Result     │             │
│  │  Consumer   │    │   Worker    │   │ Consumers   │             │
│  │             │    │             │   │ (DB/WS/     │             │
│  │             │    │ Prediction  │   │  Slack/     │             │
│  │             │    │  Engine     │   │  Learning)  │             │
│  │             │    │ RAG Service │   │             │             │
│  │             │    │ AI Service  │   │             │             │
│  └─────────────┘    └─────────────┘   └─────────────┘             │
└────────────────┬────────────────┬────────────────┬──────────────────┘
                 │                │                │
                 ▼                ▼                ▼
┌──────────────────────┐  ┌────────────────────────┐
│  PostgreSQL          │  │  Redis                 │
│  + pgvector          │  │  (Caching)             │
│  (Vector DB + Data)  │  │                        │
└──────────────────────┘  └────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│         OpenAI API                   │
│  - GPT-4o (Prediction Analysis)      │
│  - text-embedding-3-small (RAG)      │
└──────────────────────────────────────┘
```

### Key Architectural Patterns

**Event-Driven Architecture**: The system uses Apache Kafka as a central event broker to decouple services and enable asynchronous processing. This provides:
- **Low API latency**: API returns in <50ms, processing happens asynchronously
- **Horizontal scalability**: Multiple worker instances process events in parallel
- **Resilience**: Failed operations retry automatically
- **Observability**: Complete audit trail of all events

## Backend Architecture (FastAPI)

### Directory Structure
```
service/
├── src/
│   ├── main.py                    # FastAPI app entry
│   ├── config.py                  # Configuration management
│   ├── database/
│   │   ├── connection.py          # Async SQLAlchemy setup
│   │   └── models.py              # Database models (8 tables)
│   ├── api/
│   │   ├── deps.py                # Shared dependencies
│   │   └── routers/
│   │       ├── vessels.py         # Vessel CRUD endpoints
│   │       ├── terminals.py       # Terminal endpoints
│   │       ├── predictions.py     # Prediction endpoints
│   │       ├── admin.py           # Admin/metrics endpoints
│   │       └── websocket.py       # WebSocket endpoint
│   ├── services/
│   │   ├── prediction_engine.py   # Traditional algorithm
│   │   ├── ai_service.py          # GPT-4o integration
│   │   ├── rag_service.py         # Vector similarity search
│   │   ├── embedding_service.py   # OpenAI embeddings
│   │   ├── learning_service.py    # Continuous learning
│   │   ├── websocket_manager.py   # WebSocket management
│   │   └── slack_service.py       # Slack notifications
│   ├── schemas/                   # Pydantic schemas
│   └── utils/                     # Utilities (geo, logger)
├── alembic/                       # Database migrations
├── scripts/                       # Seed data scripts
└── tests/                         # Test suite
```

### Core Services

#### 1. Prediction Engine (Traditional)
**Purpose**: Algorithmic vessel destination prediction

**Components**:
- **Proximity Scoring**: Exponential decay based on distance
- **Speed Scoring**: Optimal range detection (8-15 knots for approach)
- **Heading Scoring**: Angular alignment with target terminal

**Formula**:
```
base_confidence = proximity_score × 0.4 + speed_score × 0.3 + heading_score × 0.3
```

#### 2. RAG Service (Retrieval-Augmented Generation)
**Purpose**: Historical pattern matching using vector similarity

**Technology**: pgvector with cosine similarity search

**Process**:
1. Generate embeddings of completed journeys
2. Query similar journeys for current vessel/terminal pair
3. Calculate historical similarity score (0.0-0.3 boost)

#### 3. AI Service (GPT-4o)
**Purpose**: Holistic analysis and confidence adjustment

**Input Context**:
- Vessel state (position, speed, heading)
- Terminal information
- Traditional algorithm scores
- Historical similar journeys (RAG results)
- Learned approach behaviors

**Output**:
- Confidence adjustment: -0.3 to +0.3
- Natural language reasoning
- Key factors influencing decision

**Final Score**:
```
final_confidence = base_confidence + rag_score + ai_adjustment
```

#### 4. Learning Service
**Purpose**: Continuous improvement from prediction outcomes

**Capabilities**:
- Accuracy calculation on confirmed predictions
- Embedding generation for completed predictions
- Terminal approach behavior updates
- Behavior event detection (speed/course changes)

#### 5. WebSocket Manager
**Purpose**: Real-time updates to connected clients

**Features**:
- Connection management
- Subscription-based filtering (per vessel or all)
- Broadcasts for vessel updates and new predictions

### Database Schema

#### Core Tables
1. **vessels**: Current vessel state and metadata
2. **terminals**: LNG terminal locations and characteristics
3. **vessel_journeys**: Historical journey records
4. **predictions**: Prediction records with all scores
5. **terminal_approach_behaviors**: Learned approach patterns
6. **behavior_events**: Vessel behavior change events
7. **vector_embeddings**: RAG embeddings (pgvector)

#### Key Relationships
- Vessels ↔ Predictions (one-to-many)
- Terminals ↔ Predictions (one-to-many)
- Vessels ↔ Journeys (one-to-many)
- Terminals ↔ Approach Behaviors (one-to-many)

## Frontend Architecture (Next.js)

### Component Structure

#### Pages
- **app/page.tsx**: Main application page
  - Fetches vessels and predictions from API
  - Establishes WebSocket connection
  - Manages real-time state updates
  - Renders map with predictions

#### Components

##### Map Component (`components/Map.tsx`)
**Purpose**: Interactive map visualization with predictions

**Type**: Client Component (`'use client'`)

**Props**:
```typescript
interface Vessel {
  id: string;
  name: string;
  current_lat: number;
  current_lon: number;
  heading: number;
  speed: number;
  predictions?: Prediction[];
}
```

**Features**:
- Vessel markers with real-time updates
- Terminal markers showing LNG facilities
- Prediction popups with:
  - Confidence percentage with color coding
  - Distance and ETA
  - AI reasoning explanation
  - Score breakdown
- Auto-fit bounds to show all vessels

#### API Client (`lib/api-client.ts`)
**Purpose**: HTTP client for backend API

**Endpoints**:
- `GET /api/v1/vessels` - List all vessels
- `GET /api/v1/vessels/{id}` - Get vessel with predictions
- `GET /api/v1/predictions/active` - List active predictions
- `POST /api/v1/predictions/analyze` - Trigger prediction
- `GET /api/v1/admin/metrics` - System metrics

#### WebSocket Client (`lib/websocket.ts`)
**Purpose**: Real-time updates from backend

**Features**:
- Auto-reconnect with exponential backoff
- Subscription management
- Message type handling:
  - `vessel_update`: Update vessel state
  - `prediction_created`: New prediction alert
  - `subscribed`: Confirmation message

## Data Flow

### Event-Driven Prediction Pipeline

**Synchronous Phase** (API Request):
```
1. API receives vessel position update (PUT /vessels/{id})
   ↓
2. Update vessel record in PostgreSQL
   ↓
3. Publish VesselPositionUpdateEvent to Kafka
   ↓
4. Return 202 Accepted to client (<50ms total)
```

**Asynchronous Processing Phase** (Event-Driven):
```
5. Position Consumer receives event
   ↓
6. Check if vessel within range of terminals
   ↓
7. Publish PredictionRequestEvent to Kafka
   ↓
8. Prediction Worker consumes request
   ↓
9. Traditional algorithm calculates base scores
   ↓
10. RAG service retrieves similar historical journeys (pgvector)
    ↓
11. AI service analyzes all context with GPT-4o (2-5s)
    ↓
12. Final confidence score calculated
    ↓
13. Save prediction to PostgreSQL
    ↓
14. Publish PredictionResultEvent to Kafka
    ↓
15. Multiple Consumers process result in parallel:
    - DB Writer: Persist to database (if needed)
    - WebSocket Broadcaster: Push to connected clients
    - Slack Notifier: Send alert if confidence ≥ 80%
    - Learning Service: Prepare for outcome tracking
    ↓
16. Frontend receives WebSocket update
    ↓
17. Map updates with new prediction
```

**Total Latency**:
- **API Response**: <50ms (immediate)
- **End-to-End** (position update → frontend update): 2-5 seconds (async)

### Learning Feedback Loop
```
1. Vessel arrives at terminal
   ↓
2. Outcome detector confirms arrival
   ↓
3. Learning service processes outcome
   ↓
4. Accuracy calculated
   ↓
5. Prediction + outcome embedded for RAG
   ↓
6. Terminal approach behaviors updated
   ↓
7. Future predictions benefit from learned patterns
```

## Event Streams

The system uses three core Kafka event streams for asynchronous processing:

### 1. vessel-position-updates
**Purpose**: Real-time vessel location and state changes

**Producers**: API endpoints, AIS data ingestion (future)

**Consumers**: Position analyzer (triggers predictions), WebSocket broadcaster, journey tracker

**Schema**: `VesselPositionUpdateEvent`
```json
{
  "event_id": "uuid",
  "event_type": "vessel.position.updated",
  "vessel_id": "lng-001",
  "data": {"lat": 29.76, "lon": -95.36, "speed": 12.5, "heading": 275.0}
}
```

### 2. prediction-requests
**Purpose**: Trigger asynchronous prediction analysis

**Producers**: Position consumer (when in range), scheduled jobs, manual API

**Consumers**: Prediction worker pool (horizontally scaled)

**Schema**: `PredictionRequestEvent`
```json
{
  "event_id": "uuid",
  "request_id": "req-12345",
  "vessel_id": "lng-001",
  "data": {"vessel_snapshot": {...}, "priority": "normal"}
}
```

### 3. prediction-results
**Purpose**: Completed predictions ready for consumption

**Producers**: Prediction workers (after AI analysis)

**Consumers**: DB writer, WebSocket broadcaster, Slack notifier, learning service

**Schema**: `PredictionResultEvent`
```json
{
  "event_id": "uuid",
  "request_id": "req-12345",
  "vessel_id": "lng-001",
  "data": {
    "prediction_id": "pred-789",
    "terminal_name": "Sabine Pass LNG",
    "confidence_score": 0.87,
    "ai_reasoning": "..."
  }
}
```

**Complete Event Stream Documentation**: See [Kafka Architecture](kafka-architecture.md) for detailed schemas, configurations, and processing patterns.

## Scalability Considerations

### Event-Driven Processing (Kafka)
- **Horizontal Scaling**: Add more consumer instances to process events in parallel
- **Partitioning**: Events distributed across partitions by vessel_id for load balancing
- **Consumer Groups**: Multiple instances share partition load automatically
- **Backpressure Handling**: Kafka queues events when consumers are busy
- **Replay Capability**: Reprocess historical events for debugging or retraining

### Database
- Connection pooling (10 connections + 20 overflow)
- Async queries for non-blocking I/O
- Indexes on frequently queried fields
- Vector indexes for pgvector similarity search

### API
- FastAPI async endpoints
- Event-driven processing (non-blocking, <50ms response)
- Response pagination for large datasets
- CORS configuration for multi-origin support

### WebSocket
- Connection limit management
- Subscription-based filtering (reduce bandwidth)
- Heartbeat mechanism for connection health

### Caching
- Redis for frequently accessed data
- Prediction result caching (5 min TTL)
- Terminal data caching (static)

### Worker Scaling
- **Prediction Workers**: Scale to 3-10 instances based on load
- **Consumer Parallelism**: Each partition processed by different instance
- **Independent Scaling**: Scale DB writers, WebSocket broadcasters separately

## Technology Stack Integration

### Backend Stack
- **FastAPI**: Async web framework
- **SQLAlchemy**: Async ORM
- **Alembic**: Database migrations
- **pgvector**: Vector similarity search
- **Pydantic**: Data validation
- **OpenAI**: GPT-4o + embeddings
- **confluent-kafka**: Kafka client library

### Frontend Stack
- **Next.js 15**: React framework
- **TypeScript**: Type safety
- **Mapbox GL JS**: Map visualization
- **WebSocket API**: Real-time updates

### Event Streaming
- **Apache Kafka**: Event broker for asynchronous processing
- **Zookeeper**: Kafka coordination service (or KRaft mode)
- **Event Schemas**: Pydantic models for type-safe event handling

### Infrastructure
- **PostgreSQL 16**: Primary database
- **Redis 7**: Caching layer
- **Kafka 3.x**: Event streaming platform
- **Docker**: Containerization
- **Poetry**: Python dependency management
