# Setup & Installation

This guide covers setup for both the backend (FastAPI/Python) and frontend (Next.js/TypeScript).

## Prerequisites

### System Requirements
- **Node.js** 18.x or higher
- **Python** 3.11 or higher
- **Docker** & **Docker Compose** (for PostgreSQL and Redis)
- **Poetry** (Python dependency management)
- **npm** (comes with Node.js)

### External Services
- **Mapbox account** (free tier available at [mapbox.com](https://mapbox.com))
- **OpenAI API key** (from [platform.openai.com](https://platform.openai.com))
- **Slack webhook** (optional, for notifications)

---

## Backend Setup (FastAPI)

### 1. Install Python Dependencies

```bash
cd service

# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

### 2. Start Docker Services

```bash
# Start PostgreSQL with pgvector and Redis
docker-compose up -d

# Verify services are running
docker-compose ps

# Check PostgreSQL logs
docker-compose logs postgres

# Check Redis logs
docker-compose logs redis
```

### 3. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your actual values
nano .env  # or use your preferred editor
```

**Required `.env` configuration:**

```bash
# Database
DATABASE_URL=postgresql+asyncpg://vessel_user:vessel_pass_dev@localhost:5432/vessel_track

# Redis
REDIS_URL=redis://localhost:6379/0

# OpenAI API (REQUIRED)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Slack (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Application
ENVIRONMENT=development
DEBUG=true
```

### 4. Initialize Database

```bash
# Run database migrations
poetry run alembic upgrade head

# Seed terminal data (15 LNG terminals worldwide)
poetry run python scripts/seed_terminals.py
```

### 5. Start Backend Server

```bash
# Development mode with auto-reload
poetry run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

**Backend will be available at:**
- API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

---

## Frontend Setup (Next.js)

### 1. Install Node Dependencies

```bash
cd app
npm install
```

### 2. Get Mapbox Token

1. Go to [mapbox.com/account](https://account.mapbox.com/)
2. Sign up for a free account
3. Navigate to your tokens page
4. Copy your default public token (or create a new one)

### 3. Configure Environment Variables

```bash
# Create environment file
touch .env.local

# Add your configuration
nano .env.local
```

**Required `.env.local` configuration:**

```bash
# Mapbox (REQUIRED)
NEXT_PUBLIC_MAPBOX_TOKEN=your_mapbox_token_here

# Backend API (adjust if using different port/host)
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/vessels
```

**Important**: The `.env.local` file is gitignored and should never be committed.

### 4. Run Development Server

```bash
npm run dev
```

**Frontend will be available at:**
- Application: http://localhost:3000

---

## Verification & Testing

### 1. Verify Backend

```bash
# Check health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy"}

# Check API docs
open http://localhost:8000/docs
```

### 2. Verify Database

```bash
# Connect to PostgreSQL
docker exec -it vessel_track_db psql -U vessel_user -d vessel_track

# Check pgvector extension
SELECT * FROM pg_extension WHERE extname = 'vector';

# Check terminals were seeded
SELECT COUNT(*) FROM terminals;
# Should return: 15

# Exit psql
\q
```

### 3. Create Test Vessel

```bash
# Create a test vessel via API
curl -X POST http://localhost:8000/api/v1/vessels \
  -H "Content-Type: application/json" \
  -d '{
    "id": "lng-test-001",
    "name": "Test Tanker 1",
    "current_lat": 29.7,
    "current_lon": -93.9,
    "heading": 90,
    "speed": 12.5,
    "vessel_type": "lng_tanker",
    "mmsi": "367123456",
    "imo": "IMO9123456"
  }'
```

### 4. Trigger AI Prediction

```bash
# Analyze vessel and generate predictions
curl -X POST http://localhost:8000/api/v1/predictions/analyze \
  -H "Content-Type: application/json" \
  -d '{"vessel_id": "lng-test-001"}'

# Check active predictions
curl http://localhost:8000/api/v1/predictions/active
```

### 5. Test Frontend Integration

1. Open http://localhost:3000
2. Verify map loads correctly
3. Check vessel marker appears on map
4. Click vessel marker to see prediction popup
5. Verify AI reasoning displays
6. Check WebSocket connection indicator (should show "● Live")

---

## Development Workflow

### Running Both Services

**Terminal 1 - Backend:**
```bash
cd service
docker-compose up -d  # Start DB and Redis
poetry run uvicorn src.main:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd app
npm run dev
```

### Making Code Changes

**Backend Changes:**
- Code changes auto-reload with `--reload` flag
- Database schema changes: `poetry run alembic revision --autogenerate -m "description"`
- Apply migrations: `poetry run alembic upgrade head`

**Frontend Changes:**
- Next.js hot-reloads automatically
- Type check: `npm run build`
- Lint: `npm run lint`

### Database Management

```bash
# Create new migration
cd service
poetry run alembic revision --autogenerate -m "Add new field"

# Apply migrations
poetry run alembic upgrade head

# Rollback one migration
poetry run alembic downgrade -1

# View migration history
poetry run alembic history
```

### Viewing Logs

```bash
# Backend logs (in uvicorn terminal)

# Database logs
docker-compose logs -f postgres

# Redis logs
docker-compose logs -f redis

# All Docker services
docker-compose logs -f
```

---

## Troubleshooting

### Backend Issues

**"ModuleNotFoundError" or import errors**
- Ensure you're in the Poetry shell: `poetry shell`
- Or prefix commands with `poetry run`
- Reinstall dependencies: `poetry install`

**Database connection errors**
- Check Docker containers are running: `docker-compose ps`
- Verify DATABASE_URL in `.env`
- Check PostgreSQL logs: `docker-compose logs postgres`

**OpenAI API errors**
- Verify OPENAI_API_KEY is set correctly in `.env`
- Check API key has credits: [platform.openai.com/usage](https://platform.openai.com/usage)
- Ensure model name is correct: `gpt-4o`

**pgvector extension not found**
- Verify using correct Docker image: `pgvector/pgvector:pg16`
- Check extension is enabled: `SELECT * FROM pg_extension WHERE extname = 'vector';`
- Recreate database: `docker-compose down -v && docker-compose up -d`

### Frontend Issues

**"Mapbox Token Required" Error**
- Ensure `.env.local` exists in the `app/` directory
- Verify token is correct and not expired
- Restart dev server after adding token: `npm run dev`

**Map Not Loading**
- Check browser console for errors (F12)
- Verify internet connection (Mapbox requires external resources)
- Check if Mapbox services are operational

**"Failed to fetch vessels" Error**
- Verify backend is running on http://localhost:8000
- Check NEXT_PUBLIC_API_URL in `.env.local`
- Verify CORS is configured correctly in backend
- Check browser network tab for failed requests

**WebSocket connection fails**
- Verify backend WebSocket endpoint is accessible
- Check NEXT_PUBLIC_WS_URL in `.env.local`
- Look for CORS or firewall issues
- Check backend logs for WebSocket errors

### Database Issues

**"relation does not exist" errors**
- Run migrations: `poetry run alembic upgrade head`
- Check migration status: `poetry run alembic current`

**Slow queries**
- Check indexes are created: `\d+ table_name` in psql
- Review query execution plan: `EXPLAIN ANALYZE query`
- Consider adding indexes for frequently queried fields

---

## Production Deployment Notes

### Environment-Specific Settings

**Production `.env` changes:**
```bash
ENVIRONMENT=production
DEBUG=false
DATABASE_URL=postgresql+asyncpg://user:password@production-host:5432/vessel_track
REDIS_URL=redis://production-redis-host:6379/0
```

### Security Considerations
- Use strong database passwords
- Keep OpenAI API keys secret
- Enable HTTPS for API and WebSocket
- Set appropriate CORS origins
- Use environment variables, never commit secrets
- Enable rate limiting in production

### Performance Tuning
- Increase database connection pool size
- Enable Redis caching
- Use production ASGI server (e.g., Gunicorn + Uvicorn workers)
- Enable Next.js production build: `npm run build && npm start`
- Consider CDN for static assets

---

## Next Steps

After successful setup:

1. **Explore the API**: Visit http://localhost:8000/docs and try the endpoints
2. **Create test vessels**: Use the POST /vessels endpoint
3. **Generate predictions**: Use the POST /predictions/analyze endpoint
4. **View predictions**: Open http://localhost:3000 and click vessel markers
5. **Read the documentation**:
   - [features.md](./features.md) for functionality overview
   - [architecture.md](./architecture.md) to understand the system
   - [tech-stack.md](./tech-stack.md) for technology details
6. **Review implementation phases**: See `specs/tasks/` for step-by-step implementation guide
