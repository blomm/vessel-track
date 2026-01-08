# Setup & Installation

## Prerequisites
- Node.js 18.x or higher
- npm (comes with Node.js)
- A Mapbox account (free tier available)

## Initial Setup

### 1. Install Dependencies
```bash
cd app
npm install
```

### 2. Get Mapbox Token
1. Go to [mapbox.com](https://account.mapbox.com/)
2. Sign up for a free account
3. Navigate to your tokens page
4. Copy your default public token (or create a new one)

### 3. Configure Environment Variables
Create a `.env.local` file in the `app/` directory:

```bash
cd app
touch .env.local
```

Add your Mapbox token:
```
NEXT_PUBLIC_MAPBOX_TOKEN=your_mapbox_token_here
```

**Important**: The `.env.local` file is gitignored and should never be committed.

### 4. Run Development Server
```bash
npm run dev
```

The application will be available at [http://localhost:3000](http://localhost:3000)

## Development Workflow

### Running the App
```bash
cd app
npm run dev
```

### Building for Production
```bash
npm run build
npm start
```

### Linting
```bash
npm run lint
```

### Type Checking
TypeScript will check types during development and build.

## Project Structure
```
vessel-track/
├── app/                        # Next.js application
│   ├── app/                    # App router
│   │   ├── page.tsx           # Main page
│   │   ├── layout.tsx         # Root layout
│   │   └── globals.css        # Global styles
│   ├── components/            # React components
│   │   └── Map.tsx           # Map component
│   ├── data/                  # Data and fixtures
│   │   └── mockVessels.ts    # Mock vessel data
│   ├── public/                # Static assets
│   ├── .env.local            # Environment variables (create this)
│   ├── package.json          # Dependencies
│   ├── tsconfig.json         # TypeScript config
│   └── tailwind.config.ts    # Tailwind config
└── specs/                     # Documentation
    ├── index.md              # Overview
    ├── setup.md              # This file
    ├── architecture.md       # Architecture docs
    ├── features.md           # Feature specs
    └── tech-stack.md         # Technology choices
```

## Troubleshooting

### "Mapbox Token Required" Error
- Ensure `.env.local` exists in the `app/` directory
- Verify the token is correct
- Restart the dev server after adding the token

### Map Not Loading
- Check browser console for errors
- Verify internet connection (Mapbox requires external resources)
- Check if Mapbox services are operational

### Type Errors
- Run `npm run build` to see all TypeScript errors
- Ensure all dependencies are installed
- Check that `@types/mapbox-gl` is installed

## Next Steps
After setup:
1. View the app at localhost:3000
2. Explore the 6 mock vessels on the map
3. Click vessels to see detailed information
4. Read [features.md](./features.md) for functionality overview
5. Read [architecture.md](./architecture.md) to understand the codebase
