'use client';

import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

// You'll need to add your Mapbox token here
// Get one for free at https://account.mapbox.com/
const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || '';

export interface Vessel {
  id: string;
  name: string;
  lat: number;
  lon: number;
  heading: number;
  speed: number;
  type: 'lng_tanker';
}

interface MapProps {
  vessels?: Vessel[];
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
      center: [0, 20], // Start with a global view
      zoom: 2,
    });

    map.current.on('load', () => {
      setMapLoaded(true);
    });

    // Add navigation controls
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
        el.innerHTML = `
          <div class="relative">
            <div class="w-4 h-4 bg-blue-500 rounded-full border-2 border-white shadow-lg"></div>
            <div class="absolute -top-6 left-1/2 -translate-x-1/2 bg-black/80 text-white text-xs px-2 py-1 rounded whitespace-nowrap opacity-0 hover:opacity-100 transition-opacity">
              ${vessel.name}
            </div>
          </div>
        `;

        marker = new mapboxgl.Marker({ element: el })
          .setLngLat([vessel.lon, vessel.lat])
          .setPopup(
            new mapboxgl.Popup({ offset: 25 }).setHTML(`
              <div class="p-2">
                <h3 class="font-bold">${vessel.name}</h3>
                <p class="text-sm">Type: LNG Tanker</p>
                <p class="text-sm">Speed: ${vessel.speed} knots</p>
                <p class="text-sm">Heading: ${vessel.heading}°</p>
                <p class="text-sm">Position: ${vessel.lat.toFixed(4)}, ${vessel.lon.toFixed(4)}</p>
              </div>
            `)
          )
          .addTo(map.current!);

        markers.current.set(vessel.id, marker);
      } else {
        // Update existing marker position
        marker.setLngLat([vessel.lon, vessel.lat]);
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
