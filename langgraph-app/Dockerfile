FROM python:3.11-slim
WORKDIR /app

# Kopiere die Requirements und installiere sie
COPY langgraph-app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Jetzt kommt der wichtige Teil:
# Wir kopieren NUR den internen Paket-Ordner aus dem Submodule
COPY langchain-bridge-repo/langchain_fastapi_chat_completion /app/langchain_fastapi_chat_completion

# Kopiere den Rest deiner App
COPY langgraph-app/ /app/

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8123"]
