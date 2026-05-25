FROM python:3.11-slim
WORKDIR /app

# Install Node.js/npm (for MCP) AND the GUI tools (Xvfb, VNC, Fluxbox, etc.)
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    curl \
    ca-certificates \
    chromium \
    ffmpeg \
    xvfb \
    x11vnc \
    fluxbox \
    xterm \
    python3-tk \
    python3-dev \
    scrot \
    xdotool \
    sshpass \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Set Environment Variables for the Virtual Display
ENV DISPLAY=:0
ENV SCREEN_WIDTH=1280
ENV SCREEN_HEIGHT=720
ENV SCREEN_DEPTH=24
ENV OFFICECLI_CHROMIUM_PATH=/usr/bin/chromium
ENV OFFICE_OUTPUT_DIR=/workspace/office-output

# Install OfficeCLI as a deterministic container binary. Pin OFFICECLI_VERSION
# to a release tag for fully reproducible builds; default tracks latest.
ARG OFFICECLI_VERSION=latest
RUN set -eux; \
    arch="$(uname -m)"; \
    case "$arch" in \
      x86_64) asset="officecli-linux-x64" ;; \
      aarch64|arm64) asset="officecli-linux-arm64" ;; \
      *) echo "Unsupported OfficeCLI architecture: $arch" >&2; exit 1 ;; \
    esac; \
    if [ "$OFFICECLI_VERSION" = "latest" ]; then \
      base="https://github.com/iOfficeAI/OfficeCLI/releases/latest/download"; \
    else \
      base="https://github.com/iOfficeAI/OfficeCLI/releases/download/$OFFICECLI_VERSION"; \
    fi; \
    curl -fsSL -o /usr/local/bin/officecli "$base/$asset"; \
    chmod +x /usr/local/bin/officecli; \
    officecli --version

# 1. Install Requirements
COPY langgraph-app/requirements.txt .
RUN pip install uv && uv pip install --system -r requirements.txt && uv pip install --system --no-deps langgraph-cua

# 2. Copy optional local packages used by the legacy bridge and future phases
COPY langchain-bridge-repo/langchain_fastapi_chat_completion /app/langchain_fastapi_chat_completion

# 3. Copy the Deep Researcher
COPY local-deep-researcher-repo /app/local_deep_researcher

# 4. Copy your own code
COPY langgraph-app/ /app/

# Remove old X11 locks, start the GUI helpers, then run the native LangGraph API.
CMD ["sh", "-c", "python /workspace/langgraph-app/patches/patch_langchain_openai_disable_streaming.py && python /workspace/langgraph-app/patches/patch_langchain_openai_responses_tool_streaming.py && rm -f /tmp/.X0-lock && Xvfb :0 -screen 0 1280x720x24 & fluxbox & x11vnc -display :0 -forever -nopw -listen 0.0.0.0 & cd /workspace/langgraph-app && langgraph dev --config langgraph.json --host 0.0.0.0 --port 2024 --no-browser"]
