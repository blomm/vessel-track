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
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001/ws/vessels';

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

  // Transform vessels for Map component (map API fields to component fields)
  const vesselsWithPredictions = vessels.map((vessel) => ({
    id: vessel.id,
    name: vessel.name,
    lat: vessel.current_lat,
    lon: vessel.current_lon,
    heading: vessel.heading || 0,
    speed: vessel.speed || 0,
    type: 'lng_tanker' as const,
    predictions: predictions
      .filter((p) => p.vessel_id === vessel.id)
      .map((p) => ({
        id: p.id,
        terminal_name: p.terminal_name,
        confidence_score: p.confidence_score,
        distance_to_terminal_km: p.distance_to_terminal_km,
        eta_hours: p.eta_hours,
        ai_reasoning: p.ai_reasoning
      }))
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
                  Tracking {vessels.length} vessel{vessels.length !== 1 ? 's' : ''}
                  {predictions.length > 0 &&
                    ` · ${predictions.length} active prediction${predictions.length !== 1 ? 's' : ''}`}
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
                Make sure the backend is running on http://localhost:8001
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
