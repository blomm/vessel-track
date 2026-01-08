# Technology Stack

## Frontend Framework
- **Next.js 15** with App Router
- **React 19** for UI components
- **TypeScript** for type safety

## Styling
- **Tailwind CSS** for utility-first styling
- Dark theme support built-in

## Mapping & Visualization
- **Mapbox GL JS** - Vector-based mapping library
- **mapbox-gl** (v3.x) - Core mapping functionality
- **@types/mapbox-gl** - TypeScript definitions

### Why Mapbox?
- High performance with vector tiles
- Excellent customization options
- Good support for real-time updates
- Beautiful built-in styles (using dark-v11 theme)
- Interactive markers and popups

## Development Tools
- **ESLint** for code linting
- **TypeScript** compiler for type checking
- npm for package management

## Environment Variables
- `NEXT_PUBLIC_MAPBOX_TOKEN` - Required Mapbox API token (get free at mapbox.com)

## Project Structure
```
vessel-track/
├── app/              # Next.js application
│   ├── app/          # App router pages
│   ├── components/   # React components
│   └── data/         # Mock data and fixtures
└── specs/            # Project documentation
```
