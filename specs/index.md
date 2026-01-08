# Vessel Track - Specification Index

This document is the starting point for understanding the Vessel Track project.

## Project Overview

**Vessel Track** is a web application for tracking LNG (Liquefied Natural Gas) tanker vessels on an interactive map in real-time.

**Current Status**: v0.1 - Basic implementation with mock data

## Specification Files

### Getting Started
- [setup.md](setup.md) - Installation and setup instructions

### Technical Documentation
- [architecture.md](architecture.md) - System architecture and component design
- [tech-stack.md](tech-stack.md) - Technology choices and dependencies
- [features.md](features.md) - Current and planned features

### Future Specifications (To Be Created)
- data_models.md - Vessel data structures and API contracts
- api_integration.md - AIS data provider integration
- testing_strategy.md - Testing approach and coverage
- deployment.md - Production deployment strategy

## Quick Start

1. Read [setup.md](setup.md) for installation steps
2. Get a free Mapbox token
3. Run `cd app && npm install && npm run dev`
4. Open http://localhost:3000

## Instructions for AI Assistants

1. Read [architecture.md](architecture.md) to understand the codebase structure
2. Review [features.md](features.md) to understand current and planned functionality
3. Check [tech-stack.md](tech-stack.md) for technology decisions
4. Follow existing patterns when adding new features
5. Update relevant spec files when making significant changes
6. Ask for clarification if requirements are unclear

## Current Implementation

- **Framework**: Next.js 15 with TypeScript
- **Map Library**: Mapbox GL JS
- **Data**: Mock vessel data (6 LNG tankers)
- **Features**: Interactive map with vessel markers and popups

## Next Steps

1. Integrate real AIS data source
2. Add real-time vessel position updates
3. Implement filtering and search
4. Add vessel tracking history
