# Vessel Track

A real-time LNG (Liquefied Natural Gas) tanker vessel tracking application built with Next.js and Mapbox GL.

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- 🗺️ Interactive Mapbox GL map with dark theme
- 🚢 Real-time vessel marker display
- 📍 Detailed vessel information popups (name, speed, heading, position)
- 🎯 Auto-fit view to show all vessels
- 📱 Responsive design

## Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Mapping**: Mapbox GL JS
- **Package Manager**: npm

## Getting Started

### Prerequisites

- Node.js 18.x or higher
- A free Mapbox account ([sign up here](https://account.mapbox.com/))

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/blomm/vessel-track.git
   cd vessel-track
   ```

2. Install dependencies:
   ```bash
   cd app
   npm install
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env.local
   ```

   Edit `.env.local` and add your Mapbox token:
   ```
   NEXT_PUBLIC_MAPBOX_TOKEN=your_mapbox_token_here
   ```

4. Run the development server:
   ```bash
   npm run dev
   ```

5. Open [http://localhost:3000](http://localhost:3000) in your browser

## Project Structure

```
vessel-track/
├── app/              # Next.js application
│   ├── app/          # App router pages and layouts
│   ├── components/   # React components (Map, etc.)
│   └── data/         # Mock vessel data
└── specs/            # Project documentation and specifications
```

## Current Status

**v0.1** - Basic implementation with mock data

Currently tracking 6 mock LNG tanker vessels across global locations. Future versions will integrate real AIS (Automatic Identification System) data for live vessel tracking.

## Documentation

Detailed documentation is available in the [specs/](specs/) directory:

- [Setup Guide](specs/setup.md) - Detailed installation instructions
- [Architecture](specs/architecture.md) - System design and component structure
- [Features](specs/features.md) - Current and planned features
- [Tech Stack](specs/tech-stack.md) - Technology decisions and rationale

## Roadmap

- [ ] Integrate real-time AIS data
- [ ] Add vessel filtering and search
- [ ] Implement vessel tracking history/trails
- [ ] Add weather overlays
- [ ] Mobile app version

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT

## Acknowledgments

Built with [Claude Code](https://claude.com/claude-code)
