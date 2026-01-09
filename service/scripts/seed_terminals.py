import asyncio
from sqlalchemy import select
from src.database.connection import AsyncSessionLocal
from src.database.models import Terminal

TERMINALS = [
    # USA Export Terminals
    {
        "name": "Sabine Pass LNG", "code": "SABINE",
        "lat": 29.7294, "lon": -93.8767,
        "country": "USA", "region": "Gulf of Mexico",
        "capacity_bcm_year": 27.0, "terminal_type": "export",
        "approach_zone_radius_km": 50.0
    },
    {
        "name": "Cameron LNG", "code": "CAMERON",
        "lat": 29.7965, "lon": -93.3191,
        "country": "USA", "region": "Gulf of Mexico",
        "capacity_bcm_year": 15.0, "terminal_type": "export",
        "approach_zone_radius_km": 50.0
    },
    {
        "name": "Freeport LNG", "code": "FREEPORT",
        "lat": 28.9450, "lon": -95.3028,
        "country": "USA", "region": "Gulf of Mexico",
        "capacity_bcm_year": 15.0, "terminal_type": "export",
        "approach_zone_radius_km": 50.0
    },

    # Asia Import Terminals
    {
        "name": "Tokyo Gas Negishi", "code": "NEGISHI",
        "lat": 35.4222, "lon": 139.6456,
        "country": "Japan", "region": "East Asia",
        "capacity_bcm_year": 9.0, "terminal_type": "import",
        "approach_zone_radius_km": 40.0
    },
    {
        "name": "Incheon LNG Terminal", "code": "INCHEON",
        "lat": 37.4563, "lon": 126.7052,
        "country": "South Korea", "region": "East Asia",
        "capacity_bcm_year": 12.0, "terminal_type": "import",
        "approach_zone_radius_km": 40.0
    },
    {
        "name": "Guangdong Dapeng LNG", "code": "DAPENG",
        "lat": 22.6444, "lon": 114.4906,
        "country": "China", "region": "East Asia",
        "capacity_bcm_year": 8.0, "terminal_type": "import",
        "approach_zone_radius_km": 40.0
    },

    # Europe Import Terminals
    {
        "name": "Gate Terminal Rotterdam", "code": "GATE",
        "lat": 51.9497, "lon": 4.0342,
        "country": "Netherlands", "region": "Northwest Europe",
        "capacity_bcm_year": 12.0, "terminal_type": "import",
        "approach_zone_radius_km": 35.0
    },
    {
        "name": "Zeebrugge LNG Terminal", "code": "ZEEBRUGGE",
        "lat": 51.3356, "lon": 3.2006,
        "country": "Belgium", "region": "Northwest Europe",
        "capacity_bcm_year": 9.0, "terminal_type": "import",
        "approach_zone_radius_km": 35.0
    },
    {
        "name": "Montoir-de-Bretagne", "code": "MONTOIR",
        "lat": 47.3114, "lon": -2.1428,
        "country": "France", "region": "Northwest Europe",
        "capacity_bcm_year": 10.0, "terminal_type": "import",
        "approach_zone_radius_km": 35.0
    },

    # Middle East Export
    {
        "name": "Ras Laffan LNG", "code": "RASLAFFAN",
        "lat": 25.9231, "lon": 51.5453,
        "country": "Qatar", "region": "Middle East",
        "capacity_bcm_year": 77.0, "terminal_type": "export",
        "approach_zone_radius_km": 50.0
    },

    # Australia Export
    {
        "name": "Gorgon LNG", "code": "GORGON",
        "lat": -20.6167, "lon": 115.0500,
        "country": "Australia", "region": "Pacific",
        "capacity_bcm_year": 15.6, "terminal_type": "export",
        "approach_zone_radius_km": 50.0
    },
    {
        "name": "Queensland Curtis LNG", "code": "QCLNG",
        "lat": -23.8500, "lon": 151.2667,
        "country": "Australia", "region": "Pacific",
        "capacity_bcm_year": 8.5, "terminal_type": "export",
        "approach_zone_radius_km": 50.0
    },

    # South Asia Import
    {
        "name": "Dahej LNG Terminal", "code": "DAHEJ",
        "lat": 21.7000, "lon": 72.6000,
        "country": "India", "region": "South Asia",
        "capacity_bcm_year": 17.5, "terminal_type": "import",
        "approach_zone_radius_km": 45.0
    },

    # Latin America Import
    {
        "name": "Guanabara Bay LNG", "code": "GUANABARA",
        "lat": -22.9068, "lon": -43.1729,
        "country": "Brazil", "region": "South America",
        "capacity_bcm_year": 7.0, "terminal_type": "import",
        "approach_zone_radius_km": 40.0
    },

    # UK Import
    {
        "name": "South Hook LNG", "code": "SOUTHHOOK",
        "lat": 51.7103, "lon": -5.1636,
        "country": "United Kingdom", "region": "Northwest Europe",
        "capacity_bcm_year": 21.0, "terminal_type": "import",
        "approach_zone_radius_km": 35.0
    },
]

async def seed_terminals():
    async with AsyncSessionLocal() as session:
        for terminal_data in TERMINALS:
            # Check if already exists by code
            result = await session.execute(
                select(Terminal).where(Terminal.code == terminal_data["code"])
            )
            existing = result.scalar_one_or_none()

            if not existing:
                terminal = Terminal(**terminal_data)
                session.add(terminal)
                print(f"Added terminal: {terminal_data['name']}")
            else:
                print(f"Terminal already exists: {terminal_data['name']}")

        await session.commit()
    print(f"✓ Seeded {len(TERMINALS)} terminals")

if __name__ == "__main__":
    asyncio.run(seed_terminals())
