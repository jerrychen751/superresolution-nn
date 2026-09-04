import requests
import numpy as np

# Your token
TOKEN = "YOUR_TOKEN_HERE"

# Endpoint (JHTDB public API)
url = "https://turbulence.pha.jhu.edu/service/turbulence.asmx/GetVelocity"

# Example: one point (start small!)
points = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)

# Convert points to flattened string (API requirement)
point_list = ",".join(map(str, points.flatten()))

# Parameters
params = {
    "authToken": TOKEN,
    "dataset": "channel5200",
    "time": 0.0,
    "spatialInterpolation": "Lag4",
    "temporalInterpolation": "None",
    "points": point_list
}

# Send request
response = requests.get(url, params=params)

# Raw response (XML)
print(response.text)