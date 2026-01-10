import pytest
from src.utils.geo import haversine_distance, calculate_bearing, angular_difference

def test_haversine_distance():
    # New York to London: ~5570 km
    ny_lat, ny_lon = 40.7128, -74.0060
    london_lat, london_lon = 51.5074, -0.1278

    distance = haversine_distance(ny_lat, ny_lon, london_lat, london_lon)
    assert 5500 < distance < 5600  # Approximate

def test_calculate_bearing():
    # North bearing from equator
    bearing = calculate_bearing(0, 0, 1, 0)
    assert 0 <= bearing < 10  # Approximately north

def test_angular_difference():
    assert angular_difference(10, 350) == 20  # Wraps around 360
    assert angular_difference(90, 270) == 180
    assert angular_difference(45, 45) == 0
