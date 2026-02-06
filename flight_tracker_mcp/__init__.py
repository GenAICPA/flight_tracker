#!/usr/bin/env python3
"""
Enhanced Flight Tracker MCP Server
Supports multiple flight tracking operations using OpenSky Network API
"""

import argparse
import asyncio
import json
import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
from opensky_api import OpenSkyApi
from mcp.server.stdio import stdio_server
import mcp.types as types
from mcp.server import NotificationOptions, Server, InitializationOptions


class FlightTracker:
    """Enhanced flight tracking functionality using OpenSky Network and local SDR data"""
    
    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        local_url: Optional[str] = None,
        receiver_lat: Optional[float] = None,
        receiver_lon: Optional[float] = None
    ):
        self.api = OpenSkyApi(username=username, password=password)
        self.username = username
        self.password = password
        self.local_url = local_url
        self.receiver_lat = receiver_lat
        self.receiver_lon = receiver_lon

    async def _fetch_local_data(self) -> List[Dict[str, Any]]:
        """Fetch flight data from local dump1090/readsb receiver"""
        if not self.local_url:
            return []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.local_url) as response:
                    if response.status != 200:
                        return []
                    data = await response.json()

                    # Support standard dump1090/readsb aircraft.json format
                    aircraft_list = data.get("aircraft", [])
                    now = data.get("now", time.time())

                    flights = []
                    for a in aircraft_list:
                        icao24 = a.get("hex", "").lower()
                        if not icao24:
                            continue

                        # Convert altitude from feet to meters
                        alt_ft = a.get("alt_baro") or a.get("alt_geom")
                        alt_m = alt_ft * 0.3048 if alt_ft is not None else None

                        # Convert speed from knots to m/s
                        gs_kt = a.get("gs")
                        gs_mps = gs_kt * 0.514444 if gs_kt is not None else None

                        flights.append({
                            "icao24": icao24,
                            "callsign": a.get("flight", "").strip() or None,
                            "latitude": a.get("lat"),
                            "longitude": a.get("lon"),
                            "geo_altitude_m": alt_m,
                            "velocity_mps": gs_mps,
                            "heading": a.get("track"),
                            "last_contact": int(now - a.get("seen", 0)),
                            "source": "local",
                            # Additional fields from local radio
                            "registration": a.get("r"),
                            "type": a.get("t"),
                            "squawk": a.get("squawk"),
                        })
                    return flights
        except Exception:
            return []

    def _merge_flights(
        self,
        opensky_flights: List[Dict[str, Any]],
        local_flights: List[Dict[str, Any]],
        priority: str = "local"
    ) -> List[Dict[str, Any]]:
        """Merge flights from multiple sources, prioritizing one over the other"""
        merged = {f["icao24"]: f for f in opensky_flights}

        for lf in local_flights:
            icao = lf["icao24"]
            if priority == "local" or icao not in merged:
                # If local priority or not in OpenSky, use local
                # But we might want to keep some OpenSky info if it's missing in local
                if icao in merged and priority == "local":
                    # Keep origin country from OpenSky if available
                    lf["origin_country"] = merged[icao].get("origin_country")
                merged[icao] = lf
            else:
                # OpenSky priority and already in merged, do nothing unless we want to augment
                pass

        return list(merged.values())

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        R = 6371  # Earth's radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _get_bounding_box(
        self, lat: float, lon: float, radius_km: float
    ) -> Tuple[float, float, float, float]:
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
        return (
            lat - lat_delta,
            lat + lat_delta,
            lon - lon_delta,
            lon + lon_delta,
        )

    async def get_overhead_flights(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10,
        source: str = "opensky"
    ) -> List[Dict[str, Any]]:
        opensky_flights = []
        local_flights = []
        
        if source in ["opensky", "merged"]:
            lamin, lamax, lomin, lomax = self._get_bounding_box(lat, lon, radius_km)
            # OpenSky API call is blocking, but we'll treat it as such for now
            # In a production environment, we might use run_in_executor
            states = self.api.get_states(bbox=(lamin, lamax, lomin, lomax))
            if states and states.states:
                for s in states.states:
                    if s.latitude is None or s.longitude is None:
                        continue
                    distance = self._haversine_distance(lat, lon, s.latitude, s.longitude)
                    if distance > radius_km:
                        continue
                    opensky_flights.append({
                        "icao24": s.icao24,
                        "callsign": s.callsign.strip() if s.callsign else None,
                        "origin_country": s.origin_country,
                        "latitude": s.latitude,
                        "longitude": s.longitude,
                        "geo_altitude_m": s.geo_altitude,
                        "velocity_mps": s.velocity,
                        "heading": s.true_track,
                        "distance_km": distance,
                        "last_contact": s.last_contact,
                        "source": "opensky"
                    })

        if source in ["local", "merged"]:
            all_local = await self._fetch_local_data()
            for f in all_local:
                if f["latitude"] is None or f["longitude"] is None:
                    continue
                distance = self._haversine_distance(lat, lon, f["latitude"], f["longitude"])
                if distance <= radius_km:
                    f["distance_km"] = distance
                    local_flights.append(f)

        if source == "merged":
            flights = self._merge_flights(opensky_flights, local_flights)
        elif source == "local":
            flights = local_flights
        else:
            flights = opensky_flights
            
        flights.sort(key=lambda x: x.get("distance_km", float('inf')))
        return flights

    def get_my_states(self) -> List[Dict[str, Any]]:
        """Get states from user's own receivers (requires auth)"""
        if not self.username or not self.password:
            raise ValueError("Authentication required for get_my_states")
        
        states = self.api.get_my_states()
        if not states or not states.states:
            return []
        
        return [{
            "icao24": s.icao24,
            "callsign": s.callsign.strip() if s.callsign else None,
            "sensors": s.sensors,
            "latitude": s.latitude,
            "longitude": s.longitude,
            "geo_altitude_m": s.geo_altitude,
            "velocity_mps": s.velocity,
            "heading": s.true_track,
            "last_contact": s.last_contact,
        } for s in states.states]

    async def get_states_in_bbox(
        self, 
        min_lat: float, 
        max_lat: float, 
        min_lon: float, 
        max_lon: float,
        source: str = "opensky"
    ) -> List[Dict[str, Any]]:
        opensky_flights = []
        local_flights = []
        
        if source in ["opensky", "merged"]:
            states = self.api.get_states(bbox=(min_lat, max_lat, min_lon, max_lon))
            if states and states.states:
                opensky_flights = [{
                    "icao24": s.icao24,
                    "callsign": s.callsign.strip() if s.callsign else None,
                    "origin_country": s.origin_country,
                    "latitude": s.latitude,
                    "longitude": s.longitude,
                    "geo_altitude_m": s.geo_altitude,
                    "velocity_mps": s.velocity,
                    "heading": s.true_track,
                    "last_contact": s.last_contact,
                    "source": "opensky"
                } for s in states.states]

        if source in ["local", "merged"]:
            all_local = await self._fetch_local_data()
            local_flights = [
                f for f in all_local
                if f["latitude"] is not None and f["longitude"] is not None and
                min_lat <= f["latitude"] <= max_lat and
                min_lon <= f["longitude"] <= max_lon
            ]

        if source == "merged":
            return self._merge_flights(opensky_flights, local_flights)
        elif source == "local":
            return local_flights
        return opensky_flights

    def get_flights_in_interval(
        self, 
        start_time: int, 
        end_time: int
    ) -> List[Dict[str, Any]]:
        flights = self.api.get_flights_from_interval(start_time, end_time)
        return [flight.__dict__ for flight in flights]

    def get_flights_by_aircraft(
        self, 
        icao24: str, 
        start_time: int, 
        end_time: int
    ) -> List[Dict[str, Any]]:
        flights = self.api.get_flights_by_aircraft(icao24, start_time, end_time)
        return [flight.__dict__ for flight in flights]

    def get_airport_arrivals(
        self, 
        airport: str, 
        start_time: int, 
        end_time: int
    ) -> List[Dict[str, Any]]:
        arrivals = self.api.get_arrivals_by_airport(airport, start_time, end_time)
        return [arrival.__dict__ for arrival in arrivals]

    def get_airport_departures(
        self, 
        airport: str, 
        start_time: int, 
        end_time: int
    ) -> List[Dict[str, Any]]:
        departures = self.api.get_departures_by_airport(airport, start_time, end_time)
        return [departure.__dict__ for departure in departures]

    def get_aircraft_track(
        self, 
        icao24: str, 
        timestamp: Optional[int] = None
    ) -> Dict[str, Any]:
        time_param = timestamp if timestamp is not None else 0
        track = self.api.get_track_by_aircraft(icao24, time_param)
        return {
            "icao24": track.icao24,
            "callsign": track.callsign,
            "start_time": track.startTime,
            "end_time": track.endTime,
            "path": [{
                "timestamp": p[0],
                "latitude": p[1],
                "longitude": p[2],
                "altitude": p[3],
                "heading": p[4],
                "on_ground": bool(p[5])
            } for p in track.path]
        }


# Create server instance
server = Server("flight-tracker")
tracker: Optional[FlightTracker] = None


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List all available flight tracking tools"""
    return [
        types.Tool(
            name="get_overhead_flights",
            description="Get flights currently overhead a given location within specified radius",
            inputSchema={
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "Center latitude"},
                    "longitude": {"type": "number", "description": "Center longitude"},
                    "radius_km": {
                        "type": "number", 
                        "description": "Search radius in km (default: 10)",
                        "default": 10
                    },
                    "source": {
                        "type": "string",
                        "enum": ["opensky", "local", "merged"],
                        "description": "Data source to use (default: opensky)",
                        "default": "opensky"
                    },
                },
                "required": ["latitude", "longitude"],
            },
        ),
        types.Tool(
            name="get_my_states",
            description="Get states from your own receivers (requires authentication)",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_states_in_bbox",
            description="Get state vectors within a geographic bounding box",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_lat": {"type": "number", "description": "Minimum latitude"},
                    "max_lat": {"type": "number", "description": "Maximum latitude"},
                    "min_lon": {"type": "number", "description": "Minimum longitude"},
                    "max_lon": {"type": "number", "description": "Maximum longitude"},
                    "source": {
                        "type": "string",
                        "enum": ["opensky", "local", "merged"],
                        "description": "Data source to use (default: opensky)",
                        "default": "opensky"
                    },
                },
                "required": ["min_lat", "max_lat", "min_lon", "max_lon"],
            },
        ),
        types.Tool(
            name="get_flights_in_interval",
            description="Get flights active between two timestamps (max 2 hour interval)",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_time": {"type": "integer", "description": "Start Unix timestamp"},
                    "end_time": {"type": "integer", "description": "End Unix timestamp"},
                },
                "required": ["start_time", "end_time"],
            },
        ),
        types.Tool(
            name="get_flights_by_aircraft",
            description="Get flights for specific aircraft within time interval (max 30 days)",
            inputSchema={
                "type": "object",
                "properties": {
                    "icao24": {"type": "string", "description": "ICAO 24-bit aircraft address"},
                    "start_time": {"type": "integer", "description": "Start Unix timestamp"},
                    "end_time": {"type": "integer", "description": "End Unix timestamp"},
                },
                "required": ["icao24", "start_time", "end_time"],
            },
        ),
        types.Tool(
            name="get_airport_arrivals",
            description="Get arrivals at specified airport within time interval (max 7 days)",
            inputSchema={
                "type": "object",
                "properties": {
                    "airport": {"type": "string", "description": "ICAO airport code"},
                    "start_time": {"type": "integer", "description": "Start Unix timestamp"},
                    "end_time": {"type": "integer", "description": "End Unix timestamp"},
                },
                "required": ["airport", "start_time", "end_time"],
            },
        ),
        types.Tool(
            name="get_airport_departures",
            description="Get departures from specified airport within time interval (max 7 days)",
            inputSchema={
                "type": "object",
                "properties": {
                    "airport": {"type": "string", "description": "ICAO airport code"},
                    "start_time": {"type": "integer", "description": "Start Unix timestamp"},
                    "end_time": {"type": "integer", "description": "End Unix timestamp"},
                },
                "required": ["airport", "start_time", "end_time"],
            },
        ),
        types.Tool(
            name="get_aircraft_track",
            description="Get flight track for specific aircraft (live or historical)",
            inputSchema={
                "type": "object",
                "properties": {
                    "icao24": {"type": "string", "description": "ICAO 24-bit aircraft address"},
                    "timestamp": {
                        "type": "integer", 
                        "description": "Timestamp for historical track (0 for live)",
                        "default": 0
                    },
                },
                "required": ["icao24"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    """Handle tool calls with proper error handling"""
    if not arguments:
        raise ValueError("Tool arguments are required.")

    try:
        # Overhead flights by location
        if name == "get_overhead_flights":
            lat = arguments["latitude"]
            lon = arguments["longitude"]
            radius = arguments.get("radius_km", 10)
            source = arguments.get("source", "opensky")
            flights = await tracker.get_overhead_flights(lat, lon, radius, source)
            text = json.dumps(flights, indent=2) if flights else "No flights found"
            
        # User's receiver states (requires auth)
        elif name == "get_my_states":
            states = tracker.get_my_states()
            text = json.dumps(states, indent=2) if states else "No states available"
        
        # States in bounding box
        elif name == "get_states_in_bbox":
            source = arguments.get("source", "opensky")
            states = await tracker.get_states_in_bbox(
                arguments["min_lat"],
                arguments["max_lat"],
                arguments["min_lon"],
                arguments["max_lon"],
                source
            )
            text = json.dumps(states, indent=2) if states else "No states in area"
        
        # Flights in time interval
        elif name == "get_flights_in_interval":
            flights = tracker.get_flights_in_interval(
                arguments["start_time"],
                arguments["end_time"],
            )
            text = json.dumps(flights, indent=2) if flights else "No flights found"
        
        # Aircraft-specific flights
        elif name == "get_flights_by_aircraft":
            flights = tracker.get_flights_by_aircraft(
                arguments["icao24"],
                arguments["start_time"],
                arguments["end_time"],
            )
            text = json.dumps(flights, indent=2) if flights else "No flights for aircraft"
        
        # Airport arrivals
        elif name == "get_airport_arrivals":
            arrivals = tracker.get_airport_arrivals(
                arguments["airport"],
                arguments["start_time"],
                arguments["end_time"],
            )
            text = json.dumps(arrivals, indent=2) if arrivals else "No arrivals found"
        
        # Airport departures
        elif name == "get_airport_departures":
            departures = tracker.get_airport_departures(
                arguments["airport"],
                arguments["start_time"],
                arguments["end_time"],
            )
            text = json.dumps(departures, indent=2) if departures else "No departures found"
        
        # Aircraft track
        elif name == "get_aircraft_track":
            track = tracker.get_aircraft_track(
                arguments["icao24"],
                arguments.get("timestamp", 0),
            )
            text = json.dumps(track, indent=2)
        
        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
        
        return [types.TextContent(type="text", text=text)]
    
    except Exception as e:
        return [types.TextContent(
            type="text", 
            text=f"Error: {str(e)}"
        )]


async def main(
    local_url: Optional[str] = None,
    receiver_lat: Optional[float] = None,
    receiver_lon: Optional[float] = None
):
    """Run the enhanced MCP server"""
    global tracker
    tracker = FlightTracker(
        local_url=local_url,
        receiver_lat=receiver_lat,
        receiver_lon=receiver_lon
    )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(
                notification_options=NotificationOptions(tools_changed=True)
            ),
        )


def cli_main():
    """Entry point for the CLI command."""
    parser = argparse.ArgumentParser(description="Flight Tracker MCP Server")
    parser.add_argument("--local-url", help="URL for dump1090/readsb aircraft.json")
    parser.add_argument("--receiver-lat", type=float, help="Latitude of the local receiver")
    parser.add_argument("--receiver-lon", type=float, help="Longitude of the local receiver")

    args = parser.parse_args()

    asyncio.run(main(
        local_url=args.local_url,
        receiver_lat=args.receiver_lat,
        receiver_lon=args.receiver_lon
    ))

if __name__ == "__main__":
    cli_main()
