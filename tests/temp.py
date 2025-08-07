import requests

url = "https://e7d32f20-b66f-4e82-9398-cd5ccefa77b1.eu-central-1-0.aws.cloud.qdrant.io/collections/food_recipes"
headers = {
    "api-key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.d-dHP8BfSJEpTx-gRyJQJ2rfg563Pwy4ng66FXc4k0g"
}

response = requests.get(url, headers=headers)
print(response.status_code)
print(response.json())
