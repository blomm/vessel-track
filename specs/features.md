# Features

## Current Implementation (v0.1)

### Map Visualization
- **Interactive Map**: Full-screen Mapbox GL map with dark theme
- **Navigation Controls**: Zoom in/out and compass controls in top-right
- **Pan & Zoom**: Click and drag to pan, scroll to zoom

### Vessel Display
- **Vessel Markers**: Blue circular markers for each LNG tanker
- **Hover Labels**: Vessel name appears on hover
- **Click Popups**: Detailed information popup on marker click
  - Vessel name
  - Type (LNG Tanker)
  - Current speed (knots)
  - Heading (degrees)
  - Position (lat/lon coordinates)

### Vessel Information
Each vessel includes:
- **ID**: Unique identifier
- **Name**: Vessel name
- **Position**: Latitude and longitude
- **Heading**: Direction of travel (0-360°)
- **Speed**: Velocity in knots
- **Type**: Currently only LNG tankers

### Auto-fit View
- Map automatically adjusts to show all vessels
- Intelligent padding and zoom limits

### Mock Data
Currently tracking 6 mock LNG tanker vessels in various global locations:
- Arctic Spirit (London area)
- Pacific Energy (Tokyo area)
- Gulf Carrier (Dubai area)
- Atlantic Navigator (New York area)
- Nordic Voyager (Oslo area)
- Southern Cross (Sydney area)

## Planned Features

### Phase 2
- Real-time vessel data integration (AIS data)
- Live position updates
- Vessel filtering and search
- Historical vessel tracks/trails
- Multiple vessel types

### Phase 3
- Vessel detail pages
- Route prediction
- Port information
- Weather overlays
- Fleet management

### Future Enhancements
- Multi-user support
- Custom alerts and notifications
- Data export capabilities
- Mobile responsive design improvements
- Performance metrics dashboard
