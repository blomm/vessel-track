import { Vessel } from '@/components/Map';

// Mock LNG tanker vessels with realistic locations
export const mockVessels: Vessel[] = [
  {
    id: 'lng-001',
    name: 'Arctic Spirit',
    lat: 51.5074,
    lon: -0.1278,
    heading: 145,
    speed: 12.5,
    type: 'lng_tanker',
  },
  {
    id: 'lng-002',
    name: 'Pacific Energy',
    lat: 35.6762,
    lon: 139.6503,
    heading: 270,
    speed: 15.2,
    type: 'lng_tanker',
  },
  {
    id: 'lng-003',
    name: 'Gulf Carrier',
    lat: 25.2048,
    lon: 55.2708,
    heading: 90,
    speed: 10.8,
    type: 'lng_tanker',
  },
  {
    id: 'lng-004',
    name: 'Atlantic Navigator',
    lat: 40.7128,
    lon: -74.0060,
    heading: 45,
    speed: 14.3,
    type: 'lng_tanker',
  },
  {
    id: 'lng-005',
    name: 'Nordic Voyager',
    lat: 59.9139,
    lon: 10.7522,
    heading: 180,
    speed: 11.7,
    type: 'lng_tanker',
  },
  {
    id: 'lng-006',
    name: 'Southern Cross',
    lat: -33.8688,
    lon: 151.2093,
    heading: 225,
    speed: 13.9,
    type: 'lng_tanker',
  },
];
