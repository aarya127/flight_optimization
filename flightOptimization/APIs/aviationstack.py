import requests

# Your API key
API_KEY = '3f371fc1feae7ecc9522d01e72a4c0ee'

# Base URL for the Aviationstack API
base_url = 'http://api.aviationstack.com/v1/flights'

# Parameters for the API request (e.g., fetch flights)
params = {
    'access_key': API_KEY,
    'limit': 100  # Example to limit the number of returned flights to 10
}

# Send the GET request
response = requests.get(base_url, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    
    # Loop through the data to handle missing or unknown types
    for flight in data.get('data', []):
        flight_date = flight.get('flight_date', 'Unknown Date')
        flight_status = flight.get('flight_status', 'Unknown Status')
        
        # Example of handling nested data
        departure_airport = flight.get('departure', {}).get('airport', 'Unknown Airport')
        arrival_airport = flight.get('arrival', {}).get('airport', 'Unknown Airport')
        
        print(f"Flight Date: {flight_date}")
        print(f"Flight Status: {flight_status}")
        print(f"Departure Airport: {departure_airport}")
        print(f"Arrival Airport: {arrival_airport}")
        print("-" * 50)
else:
    print(f"Error: {response.status_code}")
