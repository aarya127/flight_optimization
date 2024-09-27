import requests
import pandas as pd

# OpenSky API base URL
opensky_url = "https://opensky-network.org/api/states/all"

# Send a GET request to retrieve the data
response = requests.get(opensky_url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Extract the 'states' field which contains flight data
    flight_data = data.get('states', [])

    print(f"Retrieved {len(flight_data)} flights")

    # Flattening nested fields from the OpenSky API response
    flights = []

    for flight in flight_data:
        flat_flight = {
            'icao24': flight[0] if flight[0] is not None else 'Unknown',  # ICAO 24-bit transponder code
            'callsign': flight[1] if flight[1] is not None else 'Unknown',  # Call sign (can be None)
            'origin_country': flight[2] if flight[2] is not None else 'Unknown',  # Country of origin
            'time_position': flight[3] if flight[3] is not None else -1,  # Unix timestamp of the last position report
            'last_contact': flight[4] if flight[4] is not None else -1,  # Unix timestamp of the last update
            'longitude': flight[5] if flight[5] is not None else None,  # Longitude in degrees (can be None)
            'latitude': flight[6] if flight[6] is not None else None,  # Latitude in degrees (can be None)
            'baro_altitude': flight[7] if flight[7] is not None else None,  # Barometric altitude in meters
            'on_ground': flight[8] if flight[8] is not None else False,  # Boolean flag indicating if the plane is on the ground
            'velocity': flight[9] if flight[9] is not None else None,  # Velocity over ground in meters/second
            'heading': flight[10] if flight[10] is not None else None,  # Aircraft heading in degrees
            'vertical_rate': flight[11] if flight[11] is not None else None,  # Vertical rate in meters/second
            'sensors': str(flight[12]) if flight[12] is not None else '[]',  # List of sensor IDs converted to string
            'geo_altitude': flight[13] if flight[13] is not None else None,  # Geometric altitude in meters
            'squawk': flight[14] if flight[14] is not None else 'None',  # Transponder squawk code
            'spi': flight[15] if flight[15] is not None else False,  # Special Purpose Indicator (boolean)
            'position_source': flight[16] if flight[16] is not None else -1  # Position source (0=ADS-B, 1=ASTERIX, 2=MLAT)
        }
        flights.append(flat_flight)

    # Create a pandas DataFrame
    df = pd.DataFrame(flights)

    # Export the DataFrame to a CSV file
    df.to_csv('/Users/aaryas127/Downloads/flightheading.csv', index=False)

    print("Data has been exported to opensky_flight_data.csv")

else:
    print(f"Error: {response.status_code}")
