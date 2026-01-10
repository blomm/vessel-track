from math import radians, cos, sin, asin, sqrt, atan2, degrees
from typing import Tuple

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points on Earth.
    Uses Haversine formula.

    Args:
        lat1, lon1: First point coordinates (decimal degrees)
        lat2, lon2: Second point coordinates (decimal degrees)

    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Earth's radius in kilometers
    km = 6371 * c
    return km


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate initial bearing from point 1 to point 2.

    Args:
        lat1, lon1: Start point coordinates (decimal degrees)
        lat2, lon2: End point coordinates (decimal degrees)

    Returns:
        Bearing in degrees (0-360), where 0° is North
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)

    initial_bearing = atan2(x, y)
    bearing = (degrees(initial_bearing) + 360) % 360

    return bearing


def angular_difference(angle1: float, angle2: float) -> float:
    """
    Calculate the smallest angle between two headings.

    Args:
        angle1, angle2: Angles in degrees (0-360)

    Returns:
        Smallest angular difference in degrees (0-180)
    """
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)


def nautical_miles_to_km(nm: float) -> float:
    """Convert nautical miles to kilometers"""
    return nm * 1.852


def km_to_nautical_miles(km: float) -> float:
    """Convert kilometers to nautical miles"""
    return km / 1.852
