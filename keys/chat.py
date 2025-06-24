import os
import requests
from dotenv import load_dotenv

# Load .env variables

   

# Endpoint must include the deployment and `chat/completions?api-version=...`
endpoint = "https://ai-api-dev.dentsu.com/openai/deployments/GPT4o128k/chat/completions?api-version=2025-03-01-preview"
print("Endpoint being used:", endpoint)

headers  = {
    "x-service-line": "functions",
    "x-brand": "dentsu",
    "x-project": "aiassistant",
    "api-version": "v15",
    "Ocp-Apim-Subscription-Key": "43c73c423158406d825dfc2884a0bfea"
}

user_question = "What are the benefits of using Azure OpenAI for AI assistants?"

payload = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_question}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0
}

try:
    response = requests.post(endpoint, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    print("Full response:", data)  # Debug: see the full API response

    if "choices" in data and data["choices"]:
        answer = data["choices"][0].get("message", {}).get("content")
        if answer:
            print("✅ AI Assistant Response:")
            print(answer)
        else:
            print("❌ No answer found in response:", data)
    else:
        print("❌ Unexpected response format:", data)
except requests.exceptions.HTTPError as http_err:
    print(f"❌ HTTP error occurred: {http_err}")
    print("Raw response:", response.text)
except Exception as e:
    print("❌ Failed to parse JSON or other error:", e)
    print("Raw response:", getattr(response, "text", "No response"))
