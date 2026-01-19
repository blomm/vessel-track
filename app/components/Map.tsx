'use client';

import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || '';

export interface VesselPrediction {
  id: number;
  terminal_name: string;
  confidence_score: number;
  distance_to_terminal_km: number;
  eta_hours?: number;
  ai_reasoning: string;
}

export interface Vessel {
  id: string;
  name: string;
  lat: number;
  lon: number;
  heading: number;
  speed: number;
  type: 'lng_tanker';
  predictions?: VesselPrediction[];
}

interface MapProps {
  vessels?: Vessel[];
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.80) return 'bg-green-500';
  if (confidence >= 0.60) return 'bg-yellow-500';
  return 'bg-red-500';
}

function createPredictionsHTML(predictions: VesselPrediction[]): string {
  if (!predictions || predictions.length === 0) {
    return '<p class="text-sm text-gray-400 mt-3 pt-3 border-t">No active predictions</p>';
  }

  const topPrediction = predictions[0];
  const confidencePercent = (topPrediction.confidence_score * 100).toFixed(0);
  const confidenceColor = getConfidenceColor(topPrediction.confidence_score);

  return `
    <div class="border-t pt-3 mt-3">
      <h4 class="font-semibold text-sm mb-2">Predicted Destination</h4>

      <p class="text-base font-medium mb-2">${topPrediction.terminal_name}</p>

      <div class="flex items-center gap-2 mb-2">
        <span class="text-xs">Confidence:</span>
        <div class="flex-1 bg-gray-200 rounded-full h-2">
          <div class="h-2 rounded-full ${confidenceColor}" style="width: ${confidencePercent}%"></div>
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
        ? `<p class="text-xs text-gray-500 mt-2">+${predictions.length - 1} other prediction(s)</p>`
        : ''
      }
    </div>
  `;
}

function createVesselPopupHTML(vessel: Vessel): string {
  const predictionsHTML = createPredictionsHTML(vessel.predictions || []);

  return `
    <div class="p-3 min-w-[280px]">
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

export default function VesselMap({ vessels = [] }: MapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const markers = useRef<Map<string, mapboxgl.Marker>>(new Map());
  const [mapLoaded, setMapLoaded] = useState(false);

  // Initialize map
  useEffect(() => {
    if (map.current || !mapContainer.current) return;

    mapboxgl.accessToken = MAPBOX_TOKEN;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [0, 20],
      zoom: 2,
    });

    map.current.on('load', () => {
      setMapLoaded(true);
    });

    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

    return () => {
      map.current?.remove();
    };
  }, []);

  // Update vessel markers when vessels change
  useEffect(() => {
    if (!map.current || !mapLoaded) return;

    // Remove markers that no longer exist
    const currentVesselIds = new Set(vessels.map(v => v.id));
    markers.current.forEach((marker, id) => {
      if (!currentVesselIds.has(id)) {
        marker.remove();
        markers.current.delete(id);
      }
    });

    // Add or update markers
    vessels.forEach(vessel => {
      let marker = markers.current.get(vessel.id);

      if (!marker) {
        // Create new marker
        const el = document.createElement('div');
        el.className = 'vessel-marker';

        // Color based on prediction confidence
        const hasPrediction = vessel.predictions && vessel.predictions.length > 0;
        const topConfidence = hasPrediction ? vessel.predictions![0].confidence_score : 0;
        const markerColor = hasPrediction
          ? (topConfidence >= 0.8 ? 'bg-green-500' : topConfidence >= 0.6 ? 'bg-yellow-500' : 'bg-blue-500')
          : 'bg-blue-500';

        el.innerHTML = `
          <div class="relative">
            <div class="w-4 h-4 ${markerColor} rounded-full border-2 border-white shadow-lg"></div>
            <div class="absolute -top-6 left-1/2 -translate-x-1/2 bg-black/80 text-white text-xs px-2 py-1 rounded whitespace-nowrap opacity-0 hover:opacity-100 transition-opacity">
              ${vessel.name}
            </div>
          </div>
        `;

        marker = new mapboxgl.Marker({ element: el })
          .setLngLat([vessel.lon, vessel.lat])
          .setPopup(
            new mapboxgl.Popup({ offset: 25, maxWidth: '320px' })
              .setHTML(createVesselPopupHTML(vessel))
          )
          .addTo(map.current!);

        markers.current.set(vessel.id, marker);
      } else {
        // Update existing marker position and popup
        marker.setLngLat([vessel.lon, vessel.lat]);
        marker.getPopup()?.setHTML(createVesselPopupHTML(vessel));
      }
    });

    // Fit bounds to show all vessels if there are any
    if (vessels.length > 0) {
      const bounds = new mapboxgl.LngLatBounds();
      vessels.forEach(vessel => {
        bounds.extend([vessel.lon, vessel.lat]);
      });
      map.current.fitBounds(bounds, { padding: 50, maxZoom: 10 });
    }
  }, [vessels, mapLoaded]);

  if (!MAPBOX_TOKEN) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-900 text-white">
        <div className="text-center p-8">
          <h2 className="text-2xl font-bold mb-4">Mapbox Token Required</h2>
          <p className="mb-4">Please add your Mapbox token to continue.</p>
          <ol className="text-left list-decimal list-inside space-y-2">
            <li>Get a free token at <a href="https://account.mapbox.com/" className="text-blue-400 underline" target="_blank" rel="noopener noreferrer">mapbox.com</a></li>
            <li>Create a <code className="bg-gray-800 px-2 py-1 rounded">.env.local</code> file in the app directory</li>
            <li>Add: <code className="bg-gray-800 px-2 py-1 rounded">NEXT_PUBLIC_MAPBOX_TOKEN=your_token_here</code></li>
          </ol>
        </div>
      </div>
    );
  }

  return (
    <div ref={mapContainer} className="w-full h-full" />
  );
}
