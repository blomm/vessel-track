# Phase 6: Frontend Integration

**Duration**: Days 27-28
**Goal**: Connect Next.js frontend to FastAPI backend with WebSocket real-time updates

---

## 6.1. API Client

### Create `app/lib/api-client.ts`:

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export interface Vessel {
  id: string;
  name: string;
  current_lat: number;
  current_lon: number;
  heading: number;
  speed: number;
  vessel_type: string;
  status?: string;
  last_updated: string;
}

export interface Prediction {
  id: number;
  vessel_id: string;
  vessel_name: string;
  terminal_id: number;
  terminal_name: string;
  confidence_score: number;
  distance_to_terminal_km: number;
  eta_hours?: number;
  predicted_arrival?: string;
  proximity_score: number;
  speed_score: number;
  heading_score: number;
  historical_similarity_score: number;
  ai_confidence_adjustment: number;
  ai_reasoning: string;
  status: string;
  prediction_time: string;
}

export async function fetchVessels(): Promise<Vessel[]> {
  const response = await fetch(`${API_BASE}/vessels`);
  if (!response.ok) {
    throw new Error(`Failed to fetch vessels: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchVessel(vesselId: string): Promise<Vessel> {
  const response = await fetch(`${API_BASE}/vessels/${vesselId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch vessel: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchActivePredictions(): Promise<Prediction[]> {
  const response = await fetch(`${API_BASE}/predictions/active`);
  if (!response.ok) {
    throw new Error(`Failed to fetch predictions: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchPrediction(predictionId: number): Promise<Prediction> {
  const response = await fetch(`${API_BASE}/predictions/${predictionId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch prediction: ${response.statusText}`);
  }
  return response.json();
}

export async function triggerPredictionAnalysis(vesselId: string) {
  const response = await fetch(`${API_BASE}/predictions/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ vessel_id: vesselId })
  });

  if (!response.ok) {
    throw new Error(`Failed to trigger prediction: ${response.statusText}`);
  }

  return response.json();
}

export async function fetchMetrics() {
  const response = await fetch(`${API_BASE}/admin/metrics`);
  if (!response.ok) {
    throw new Error(`Failed to fetch metrics: ${response.statusText}`);
  }
  return response.json();
}
```

---

## 6.2. WebSocket Client

### Create `app/lib/websocket.ts`:

```typescript
type MessageHandler = (message: any) => void;

export class VesselWebSocket {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 3000;
  private url: string;
  private onMessageHandler: MessageHandler;
  private isIntentionallyClosed = false;

  constructor(url: string, onMessage: MessageHandler) {
    this.url = url;
    this.onMessageHandler = onMessage;
  }

  connect() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }

    this.isIntentionallyClosed = false;
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('✓ WebSocket connected to backend');
      this.reconnectAttempts = 0;

      // Subscribe to all vessels
      this.send({ type: 'subscribe', vessel_ids: [] });
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.onMessageHandler(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);

      if (!this.isIntentionallyClosed) {
        this.attemptReconnect();
      }
    };
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * this.reconnectAttempts;

      console.log(
        `Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms...`
      );

      setTimeout(() => {
        this.connect();
      }, delay);
    } else {
      console.error('Max reconnection attempts reached');
    }
  }

  send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('Cannot send message: WebSocket not connected');
    }
  }

  subscribe(vesselIds: string[]) {
    this.send({ type: 'subscribe', vessel_ids: vesselIds });
  }

  disconnect() {
    this.isIntentionallyClosed = true;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
```

---

## 6.3. Update Environment Variables

### Update `app/.env.local`:

```bash
# Existing
NEXT_PUBLIC_MAPBOX_TOKEN=your_mapbox_token_here

# NEW: Backend API
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/vessels
```

---

## 6.4. Update Page Component

### Update `app/app/page.tsx`:

```typescript
'use client';

import { useState, useEffect } from 'react';
import VesselMap from '@/components/Map';
import { fetchVessels, fetchActivePredictions, Vessel, Prediction } from '@/lib/api-client';
import { VesselWebSocket } from '@/lib/websocket';

export default function Home() {
  const [vessels, setVessels] = useState<Vessel[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [wsConnected, setWsConnected] = useState(false);

  useEffect(() => {
    // Initial data load
    async function loadData() {
      try {
        setLoading(true);
        const [vesselsData, predictionsData] = await Promise.all([
          fetchVessels(),
          fetchActivePredictions()
        ]);

        setVessels(vesselsData);
        setPredictions(predictionsData);
        setError(null);
      } catch (err) {
        console.error('Failed to load data:', err);
        setError('Failed to load vessel data. Is the backend running?');
      } finally {
        setLoading(false);
      }
    }

    loadData();

    // WebSocket for real-time updates
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/vessels';

    const ws = new VesselWebSocket(wsUrl, (message) => {
      if (message.type === 'subscribed') {
        setWsConnected(true);
        console.log('Subscribed to vessel updates');
      } else if (message.type === 'vessel_update') {
        // Update vessel in state
        setVessels((prev) =>
          prev.map((v) =>
            v.id === message.data.id ? { ...v, ...message.data } : v
          )
        );
      } else if (message.type === 'prediction_created') {
        // Add new prediction
        setPredictions((prev) => [message.data, ...prev]);
      }
    });

    ws.connect();

    // Cleanup
    return () => {
      ws.disconnect();
    };
  }, []);

  // Find predictions for each vessel
  const vesselsWithPredictions = vessels.map((vessel) => ({
    ...vessel,
    predictions: predictions.filter((p) => p.vessel_id === vessel.id)
  }));

  return (
    <div className="flex flex-col h-screen">
      <header className="bg-gray-900 text-white p-4 shadow-lg">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">LNG Vessel Tracker</h1>
            <p className="text-sm text-gray-400">
              {loading ? (
                'Loading...'
              ) : error ? (
                <span className="text-red-400">{error}</span>
              ) : (
                <>
                  Tracking {vessels.length} vessels
                  {predictions.length > 0 &&
                    ` • ${predictions.length} active predictions`}
                  {wsConnected && (
                    <span className="ml-2 text-green-400">● Live</span>
                  )}
                </>
              )}
            </p>
          </div>
        </div>
      </header>

      <main className="flex-1 relative">
        {error ? (
          <div className="flex items-center justify-center h-full bg-gray-900 text-white">
            <div className="text-center p-8">
              <h2 className="text-xl font-bold mb-4">Connection Error</h2>
              <p className="mb-4">{error}</p>
              <p className="text-sm text-gray-400">
                Make sure the backend is running on http://localhost:8000
              </p>
            </div>
          </div>
        ) : (
          <VesselMap vessels={vesselsWithPredictions} />
        )}
      </main>
    </div>
  );
}
```

---

## 6.5. Update Map Component with Predictions

### Update `app/components/Map.tsx`:

Add prediction display to vessel popup:

```typescript
// Update Vessel interface to include predictions
export interface Vessel {
  id: string;
  name: string;
  lat: number;
  lon: number;
  heading: number;
  speed: number;
  type: 'lng_tanker';
  predictions?: Array<{
    id: number;
    terminal_name: string;
    confidence_score: number;
    distance_to_terminal_km: number;
    eta_hours?: number;
    ai_reasoning: string;
  }>;
}

// Update popup HTML generation
function createVesselPopupHTML(vessel: Vessel): string {
  const predictionsHTML = vessel.predictions && vessel.predictions.length > 0
    ? createPredictionsHTML(vessel.predictions)
    : '<p class="text-sm text-gray-400 mt-2">No active predictions</p>';

  return `
    <div class="p-3 min-w-[320px]">
      <h3 class="font-bold text-lg mb-2">${vessel.name}</h3>

      <div class="space-y-1 mb-3">
        <p class="text-sm">Type: LNG Tanker</p>
        <p class="text-sm">Speed: ${vessel.speed} knots</p>
        <p class="text-sm">Heading: ${vessel.heading}°</p>
        <p class="text-sm text-gray-500">
          Position: ${vessel.lat.toFixed(4)}, ${vessel.lon.toFixed(4)}
        </p>
      </div>

      ${predictionsHTML}
    </div>
  `;
}

function createPredictionsHTML(predictions: any[]): string {
  const topPrediction = predictions[0];

  const confidenceColor = getConfidenceColor(topPrediction.confidence_score);
  const confidencePercent = (topPrediction.confidence_score * 100).toFixed(0);

  return `
    <div class="border-t pt-3 mt-3">
      <h4 class="font-semibold text-sm mb-2">Predicted Destination</h4>

      <p class="text-base font-medium mb-2">${topPrediction.terminal_name}</p>

      <div class="flex items-center gap-2 mb-2">
        <span class="text-xs">Confidence:</span>
        <div class="flex-1 bg-gray-200 rounded-full h-2">
          <div
            class="h-2 rounded-full ${confidenceColor}"
            style="width: ${confidencePercent}%"
          ></div>
        </div>
        <span class="text-xs font-semibold">${confidencePercent}%</span>
      </div>

      <div class="space-y-1 text-xs text-gray-600">
        <p>Distance: ${topPrediction.distance_to_terminal_km.toFixed(1)} km</p>
        ${topPrediction.eta_hours
          ? `<p>ETA: ${topPrediction.eta_hours.toFixed(1)} hours</p>`
          : ''
        }
      </div>

      ${topPrediction.ai_reasoning
        ? `<div class="mt-2 p-2 bg-blue-50 rounded text-xs">
             <p class="font-semibold mb-1">AI Analysis:</p>
             <p class="text-gray-700">${topPrediction.ai_reasoning}</p>
           </div>`
        : ''
      }

      ${predictions.length > 1
        ? `<p class="text-xs text-gray-500 mt-2">
             +${predictions.length - 1} other prediction(s)
           </p>`
        : ''
      }
    </div>
  `;
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.80) return 'bg-green-500';
  if (confidence >= 0.60) return 'bg-yellow-500';
  return 'bg-red-500';
}
```

---

## 6.6. Add Terminal Markers (Optional)

### Create `app/components/TerminalMarkers.tsx`:

```typescript
'use client';

import { useEffect, useRef } from 'react';
import mapboxgl from 'mapbox-gl';

interface Terminal {
  id: number;
  name: string;
  lat: number;
  lon: number;
  country: string;
  terminal_type: string;
}

interface TerminalMarkersProps {
  map: mapboxgl.Map | null;
  mapLoaded: boolean;
  terminals: Terminal[];
}

export function useTerminalMarkers({ map, mapLoaded, terminals }: TerminalMarkersProps) {
  const terminalMarkers = useRef<Map<number, mapboxgl.Marker>>(new Map());

  useEffect(() => {
    if (!map || !mapLoaded || !terminals.length) return;

    // Clear existing markers
    terminalMarkers.current.forEach((marker) => marker.remove());
    terminalMarkers.current.clear();

    // Add terminal markers
    terminals.forEach((terminal) => {
      const el = document.createElement('div');
      el.className = 'terminal-marker';
      el.innerHTML = `
        <div class="relative">
          <div class="w-3 h-3 bg-orange-400 rounded-sm border border-white shadow-md"></div>
        </div>
      `;

      const marker = new mapboxgl.Marker({ element: el })
        .setLngLat([terminal.lon, terminal.lat])
        .setPopup(
          new mapboxgl.Popup({ offset: 15 }).setHTML(`
            <div class="p-2">
              <h4 class="font-semibold">${terminal.name}</h4>
              <p class="text-xs text-gray-600">${terminal.country}</p>
              <p class="text-xs">Type: ${terminal.terminal_type.toUpperCase()}</p>
            </div>
          `)
        )
        .addTo(map);

      terminalMarkers.current.set(terminal.id, marker);
    });

    return () => {
      terminalMarkers.current.forEach((marker) => marker.remove());
      terminalMarkers.current.clear();
    };
  }, [map, mapLoaded, terminals]);
}
```

Update Map.tsx to use terminal markers:

```typescript
import { useTerminalMarkers } from './TerminalMarkers';

// In VesselMap component, add:
const [terminals, setTerminals] = useState([]);

useEffect(() => {
  // Fetch terminals
  fetch(`${process.env.NEXT_PUBLIC_API_URL}/terminals`)
    .then(res => res.json())
    .then(setTerminals)
    .catch(console.error);
}, []);

// After map initialization:
useTerminalMarkers({ map: map.current, mapLoaded, terminals });
```

---

## Verification Checklist

- [ ] Backend API running on http://localhost:8000
- [ ] Frontend connects to backend successfully
- [ ] Vessels load from API
- [ ] Active predictions load from API
- [ ] WebSocket connection establishes
- [ ] Real-time vessel updates work
- [ ] Prediction updates broadcast to frontend
- [ ] Vessel popups show predictions
- [ ] AI reasoning displays in popups
- [ ] Confidence bars render correctly
- [ ] Terminal markers appear on map (if implemented)
- [ ] Error handling works when backend is down
- [ ] Live indicator shows WebSocket status

---

## Testing Full Stack

### 1. Start Backend:

```bash
cd service
docker-compose up -d  # Start PostgreSQL + Redis
poetry run uvicorn src.main:app --reload
```

### 2. Create Test Data:

```bash
# Create test vessels
curl -X POST http://localhost:8000/api/v1/vessels \
  -H "Content-Type: application/json" \
  -d '{
    "id": "lng-test-001",
    "name": "Test Tanker 1",
    "current_lat": 29.7,
    "current_lon": -93.9,
    "heading": 90,
    "speed": 12.5,
    "vessel_type": "lng_tanker"
  }'

# Trigger prediction
curl -X POST http://localhost:8000/api/v1/predictions/analyze \
  -H "Content-Type: application/json" \
  -d '{"vessel_id": "lng-test-001"}'
```

### 3. Start Frontend:

```bash
cd ../app
npm run dev
```

### 4. Test:

1. Open http://localhost:3000
2. Verify vessels appear on map
3. Click vessel marker → see prediction with AI reasoning
4. Check browser console for WebSocket connection
5. Watch for real-time updates

---

## Next Steps

Once this phase is complete, move to **Phase 7: Testing & Deployment** where we'll:
- Write comprehensive tests
- Create production Docker configuration
- Set up CI/CD
- Deploy to staging/production
- Monitor performance and accuracy
