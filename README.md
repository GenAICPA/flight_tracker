# Flight Tracker MCP Server

A Model Context Protocol (MCP) server that provides comprehensive flight tracking capabilities using the OpenSky Network API.

## Features

- **Real-time Flight Tracking**: Get live flight data for aircraft overhead any location
- **Geographic Searches**: Query flights within bounding boxes or radius searches
- **Historical Data**: Access flight histories for specific aircraft or airports
- **Airport Operations**: Track arrivals and departures at specific airports
- **Flight Paths**: Get detailed flight tracks with position history
- **Authentication Support**: Optional authentication for enhanced data access

## Installation

```bash
pip install flight-tracker-mcp
```

## Usage

### Basic MCP Server Setup

```python
from flight_tracker_mcp import main
import asyncio

# Run the MCP server
if __name__ == "__main__":
    asyncio.run(main())
```

### Available Tools

#### 1. Get Overhead Flights
Get flights currently overhead a specific location:

```json
{
  "name": "get_overhead_flights",
  "arguments": {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "radius_km": 10
  }
}
```

#### 2. Get States in Bounding Box
Query flights within a geographic area:

```json
{
  "name": "get_states_in_bbox",
  "arguments": {
    "min_lat": 40.0,
    "max_lat": 41.0,
    "min_lon": -75.0,
    "max_lon": -73.0
  }
}
```

#### 3. Get Airport Arrivals
Track arrivals at a specific airport:

```json
{
  "name": "get_airport_arrivals",
  "arguments": {
    "airport": "KJFK",
    "start_time": 1640995200,
    "end_time": 1641081600
  }
}
```

#### 4. Get Aircraft Track
Get the flight path of a specific aircraft:

```json
{
  "name": "get_aircraft_track",
  "arguments": {
    "icao24": "a1b2c3",
    "timestamp": 0
  }
}
```

### Authentication (Optional)

For enhanced data access, you can provide OpenSky Network credentials:

```python
from flight_tracker_mcp import FlightTracker

# With authentication
tracker = FlightTracker(username="your_username", password="your_password")

# Without authentication (limited to public data)
tracker = FlightTracker()
```

## MCP Integration

This server integrates with MCP-compatible clients like Claude Desktop. Add to your MCP configuration:

```json
{
  "mcpServers": {
    "flight-tracker": {
      "command": "python",
      "args": ["-m", "flight_tracker_mcp"]
    }
  }
}
```

## API Reference

All tools return JSON-formatted flight data with the following common fields:

- `icao24`: Aircraft identifier
- `callsign`: Flight callsign
- `latitude`/`longitude`: Position coordinates
- `geo_altitude_m`: Altitude in meters
- `velocity_mps`: Speed in meters per second
- `heading`: True track heading
- `last_contact`: Timestamp of last position update

## Requirements

- Python >= 3.8
- OpenSky Network API access (free registration recommended)
- MCP-compatible client

## Data Source

This package uses the [OpenSky Network](https://opensky-network.org/) API, which provides real-time and historical flight data from a global network of ADS-B receivers.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please open an issue on the GitHub repository.
