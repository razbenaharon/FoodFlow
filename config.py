"""
Centralized configuration for Azure OpenAI and Qdrant.
Prefer environment variables over hardcoding credentials.

If using a `.env` file, ensure it's not committed and contains:
  API_KEY=...
"""
from dotenv import load_dotenv
import os

# Load environment variables from .env if present
load_dotenv()

# === Azure OpenAI configuration ===
AZURE_OPENAI_API_KEY = os.getenv("API_KEY")

# Consider updating to the latest stable API version when applicable
# AZURE_OPENAI_API_VERSION = "2024-02-15-preview"
AZURE_OPENAI_API_VERSION = "2023-05-15"
AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"

# Deployment names (must exist in your Azure OpenAI resource)
CHAT_DEPLOYMENT_NAME = "team4-gpt4o"
EMBEDDING_DEPLOYMENT_NAME = "team4-embedding"

# === Qdrant configuration ===
# TIP: Store these in environment variables instead of hardcoding.
QDRANT_URL = "https://e7d32f20-b66f-4e82-9398-cd5ccefa77b1.eu-central-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.d-dHP8BfSJEpTx-gRyJQJ2rfg563Pwy4ng66FXc4k0g"
