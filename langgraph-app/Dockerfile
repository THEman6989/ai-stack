FROM python:3.11-slim
WORKDIR /app

# Install Node.js/npm (for MCP) AND the GUI tools (Xvfb, VNC, Fluxbox, etc.)
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    curl \
    xvfb \
    x11vnc \
    fluxbox \
    xterm \
    python3-tk \
    python3-dev \
    scrot \
    xdotool \
    && rm -rf /var/lib/apt/lists/*

# Set Environment Variables for the Virtual Display
ENV DISPLAY=:0
ENV SCREEN_WIDTH=1280
ENV SCREEN_HEIGHT=720
ENV SCREEN_DEPTH=24

# 1. Install Requirements
COPY langgraph-app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the API-Bridge (Your Custom Fork)
COPY langchain-bridge-repo/langchain_fastapi_chat_completion /app/langchain_fastapi_chat_completion

# 3. Copy the Deep Researcher
COPY local-deep-researcher-repo /app/local_deep_researcher

# 4. Copy your own code
COPY langgraph-app/ /app/

# Remove any old X11 locks, start Xvfb, Fluxbox, VNC Server, and finally the FastAPI server
CMD ["sh", "-c", "rm -f /tmp/.X0-lock && Xvfb :0 -screen 0 1280x720x24 & fluxbox & x11vnc -display :0 -forever -nopw -listen 0.0.0.0 & uvicorn server:app --host 0.0.0.0 --port 8123"]