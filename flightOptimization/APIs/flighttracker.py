import requests

url = "https://aerodatabox.p.rapidapi.com/industry/faa-ladd/N123A/status"

headers = {
	"x-rapidapi-key": "Sign Up for Key",
	"x-rapidapi-host": "aerodatabox.p.rapidapi.com"
}

response = requests.get(url, headers=headers)

print(response.json())