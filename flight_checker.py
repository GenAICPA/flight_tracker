#!/usr/bin/env python3
"""
A simple, direct client to check for overhead flights using the OpenSky Network API.
"""

import json
import math
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode

# --- Configuration ---
HOME_LAT = 40.259468940733676
HOME_LON = -76.9473328015312
RADIUS_MILES = 3
RADIUS_KM = RADIUS_MILES * 1.60934
BASE_URL = "https://opensky-network.org/api"

# --- Helper Functions (from flight_tracker_mcp.py) ---

def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers"""
    R = 6371  # Earth's radius in km
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat/2) * math.sin(dlat/2) + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2) * math.sin(dlon/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def _get_bounding_box(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """Get bounding box for a given radius around a point"""
    # Rough approximation: 1 degree =  111 km
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
    
    return (
        lat - lat_delta,  # lamin
        lat + lat_delta,  # lamax
        lon - lon_delta,  # lomin
        lon + lon_delta   # lomax
    )

def format_flight_info(flight: dict) -> str:
    """Format flight information for display"""
    callsign = flight.get('callsign', 'Unknown')
    icao24 = flight.get('icao24', 'Unknown')
    country = flight.get('origin_country', 'Unknown')
    distance = flight.get('distance_km', 'Unknown')
    
    altitude = flight.get('baro_altitude')
    altitude_str = f"{altitude * 3.28084:.0f} ft" if altitude else "Unknown"
    
    velocity = flight.get('velocity')
    velocity_str = f"{velocity * 2.23694:.0f} mph" if velocity else "Unknown"
    
    on_ground = flight.get('on_ground', False)
    status = "On Ground" if on_ground else "Airborne"
    
    return f"""
Flight: {callsign} ({icao24})
  Status: {status}
  Country: {country}
  Distance: {distance:.2f} km ({distance * 0.621371:.2f} miles)
  Altitude: {altitude_str}
  Velocity: {velocity_str}
    """.strip()

# --- Main Execution ---

def get_flights():
    """
    Fetches and displays flights from the OpenSky Network API.
    """
    print(f"Searching for flights within {RADIUS_MILES} miles ({RADIUS_KM:.2f} km) of your location...")
    
    lamin, lamax, lomin, lomax = _get_bounding_box(HOME_LAT, HOME_LON, RADIUS_KM)
    
    params = {
        'lamin': lamin,
        'lamax': lamax,
        'lomin': lomin,
        'lomax': lomax
    }
    
    url = f"{BASE_URL}/states/all?{urlencode(params)}"
    
    try:
        request = Request(url)
        request.add_header('User-Agent', 'FlightChecker/1.0')
        
        with urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        if not data or 'states' not in data or not data['states']:
            print("No flights found in the specified area.")
            return

        found_flights = []
        for state in data['states']:
            if len(state) < 17 or not state[5] or not state[6]:  # lat/lon required
                continue
            
            flight = {
                'icao24': state[0],
                'callsign': state[1].strip() if state[1] else 'N/A',
                'origin_country': state[2],
                'longitude': state[5],
                'latitude': state[6],
                'baro_altitude': state[7],
                'on_ground': state[8],
                'velocity': state[9],
            }
            
            distance_km = _haversine_distance(
                HOME_LAT, HOME_LON,
                flight['latitude'], flight['longitude']
            )
            
            if distance_km <= RADIUS_KM:
                flight['distance_km'] = distance_km
                found_flights.append(flight)
        
        if not found_flights:
            print("No flights currently within the radius.")
            return

        # Sort by distance
        found_flights.sort(key=lambda x: x['distance_km'])
        
        print(f"\nFound {len(found_flights)} flight(s):")
        for flight in found_flights:
            print("-" * 40)
            print(format_flight_info(flight))

    except (URLError, HTTPError) as e:
        print(f"API request failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    get_flights()
