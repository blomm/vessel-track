const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001/api/v1';

export interface Vessel {
  id: string;
  name: string;
  current_lat: number;
  current_lon: number;
  heading?: number;
  speed?: number;
  vessel_type: string;
  mmsi?: string;
  imo?: string;
  status?: string;
  last_updated: string;
  created_at: string;
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

export interface Terminal {
  id: number;
  name: string;
  code: string;
  lat: number;
  lon: number;
  country: string;
  region: string;
  terminal_type: string;
  capacity_bcm_year?: number;
  approach_zone_radius_km: number;
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

export async function fetchTerminals(): Promise<Terminal[]> {
  const response = await fetch(`${API_BASE}/terminals`);
  if (!response.ok) {
    throw new Error(`Failed to fetch terminals: ${response.statusText}`);
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
