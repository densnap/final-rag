import os
import requests
endpoint = os.getenv("AZURE_OPENAI_URL")  # Full URL including deployment path and ?api-version

# Define request headers
headers = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY"),
    "x-service-line": os.getenv("AZURE_OPENAI_SERVICE_LINE"),
    "x-brand": os.getenv("AZURE_OPENAI_BRAND"),
    "x-project": os.getenv("AZURE_OPENAI_PROJECT"),
    'api-version': os.getenv("AZURE_OPENAI_API_VERSION")
}

input_text = "SHIVAM SANAP"
payload = {
    "input": input_text
}

response = requests.post(endpoint, headers=headers, json=payload)

# Handle the response
if response.status_code == 200:
    embedding = response.json()["data"][0]["embedding"]
    print("✅ Embedding generated successfully!")
    print(embedding)
else:
    print(f"❌ Error {response.status_code}")
    print(response.json())