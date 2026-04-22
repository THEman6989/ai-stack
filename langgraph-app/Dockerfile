FROM python:3.11-slim
WORKDIR /app

# 1. Requirements installieren
COPY langgraph-app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Die API-Bridge kopieren (Dein Custom Fork)
COPY langchain-bridge-repo/langchain_fastapi_chat_completion /app/langchain_fastapi_chat_completion

# 3. NEU: Den Deep Researcher kopieren
COPY local-deep-researcher-repo /app/local_deep_researcher

# 4. Deinen eigenen Code kopieren
COPY langgraph-app/ /app/

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8123"]
