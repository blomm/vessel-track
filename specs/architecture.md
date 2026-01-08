# Architecture

## Overview
Vessel Track is a Next.js application for tracking LNG tanker vessels on an interactive map.

## Component Architecture

### Pages
- **app/page.tsx**: Main application page
  - Renders header with vessel count
  - Hosts the Map component
  - Passes vessel data to Map

### Components

#### Map Component (`components/Map.tsx`)
**Purpose**: Primary map visualization component

**Type**: Client Component (`'use client'`)

**Props**:
```typescript
interface MapProps {
  vessels?: Vessel[];
}
```

**Responsibilities**:
- Initialize Mapbox GL map instance
- Manage map lifecycle (mount/unmount)
- Create and update vessel markers
- Handle marker interactions (hover, click)
- Auto-fit map bounds to show all vessels
- Display error state if Mapbox token missing

**Key Features**:
- Lazy initialization (only creates map once)
- Efficient marker management (reuses existing markers)
- Custom HTML markers with hover states
- Popup information on click
- Responsive to vessel prop changes

**State Management**:
- `mapLoaded`: Tracks when map is ready
- `map` ref: Holds Mapbox map instance
- `markers` ref: Map of vessel IDs to marker instances
- `mapContainer` ref: DOM reference for map container

### Data Layer

#### Vessel Interface
```typescript
interface Vessel {
  id: string;         // Unique identifier
  name: string;       // Vessel name
  lat: number;        // Latitude (-90 to 90)
  lon: number;        // Longitude (-180 to 180)
  heading: number;    // Direction (0-360 degrees)
  speed: number;      // Speed in knots
  type: 'lng_tanker'; // Vessel type
}
```

#### Mock Data (`data/mockVessels.ts`)
- Array of 6 sample LNG tanker vessels
- Realistic global distribution
- Placeholder for future API integration

## Data Flow

```
mockVessels.ts (static data)
    ↓
page.tsx (passes as props)
    ↓
Map.tsx (receives vessels prop)
    ↓
useEffect (watches vessels changes)
    ↓
Updates markers on map
```

## Future Architecture Considerations

### Real-time Data Integration
When moving to live AIS data:
1. Create API route handlers in `app/api/`
2. Implement data fetching hooks
3. Add WebSocket or polling for live updates
4. Consider state management (React Context or Zustand)

### Scalability
For handling many vessels:
- Implement clustering for dense areas
- Add virtualization for marker rendering
- Consider server-side filtering
- Add pagination or viewport-based loading

### Data Sources
Planned integration points:
- AIS data providers (MarineTraffic, VesselFinder)
- WebSocket feeds for real-time updates
- Historical data APIs
- Port and weather APIs
