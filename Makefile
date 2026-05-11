PYTHON ?= python
COMPOSE ?= docker compose
STREAMING ?= prompt
SUBMODULES ?= prompt
BUILD ?= prompt
START ?= prompt
PROFILES ?= prompt
UPDATE_STREAMING ?= prompt
UPDATE_SUBMODULES ?= yes
UPDATE_BUILD ?= yes
UPDATE_START ?= yes
UPDATE_PROFILES ?= prompt

.PHONY: help install install-fullstreaming install-hybrid install-nonstreaming install-chat install-chat-fullstreaming install-chat-nonstreaming configure profiles streaming fullstreaming full-streaming hybrid-streaming nonstreaming chat-completions chat-fullstreaming chat-nonstreaming model-management owner-model-management media-vision openwebui update update-no-start status up up-fullstreaming up-chat-fullstreaming down logs submodules build bridge-smoke hermes-smoke media-smoke openwebui-smoke

help:
	@printf '%s\n' \
		'AlphaRavis Makefile' \
		'' \
		'Install / configure:' \
		'  make install STREAMING=prompt|responses-full|chat-full START=prompt|yes|no BUILD=prompt|yes|no PROFILES=prompt|none|openwebui' \
		'  make update                  # git pull, choose runtime profile, update submodules, build, start' \
		'  make install-fullstreaming   # Responses full streaming, init submodules, build, start' \
		'  make install-chat-fullstreaming # Chat Completions full streaming, init submodules, build, start' \
		'  make profiles                # show every runtime profile and the .env values it writes' \
		'  make streaming STREAMING=full # only update .env runtime/streaming settings' \
		'' \
		'Runtime:' \
		'  make up                      # docker compose up -d --build' \
		'  make up-fullstreaming        # set full streaming, then recreate langgraph-api/api-bridge' \
		'  make up-chat-fullstreaming   # set Chat Completions streaming, then recreate langgraph-api/api-bridge' \
		'  make down | make logs | make status' \
		'' \
		'Smoke checks:' \
		'  make bridge-smoke | make hermes-smoke | make media-smoke | make openwebui-smoke'

install:
	$(PYTHON) scripts/alpharavis_setup.py install --streaming-mode "$(STREAMING)" --submodules "$(SUBMODULES)" --build "$(BUILD)" --start "$(START)" --profiles "$(PROFILES)"

install-fullstreaming:
	$(PYTHON) scripts/alpharavis_setup.py install --streaming-mode full --submodules yes --build yes --start yes --profiles "$(PROFILES)"

install-hybrid:
	$(PYTHON) scripts/alpharavis_setup.py install --streaming-mode hybrid --submodules yes --build yes --start yes --profiles "$(PROFILES)"

install-nonstreaming:
	$(PYTHON) scripts/alpharavis_setup.py install --streaming-mode nonstreaming --submodules yes --build yes --start yes --profiles "$(PROFILES)"

install-chat:
	$(PYTHON) scripts/alpharavis_setup.py install --streaming-mode chat --submodules yes --build yes --start yes --profiles "$(PROFILES)"

install-chat-fullstreaming:
	$(PYTHON) scripts/alpharavis_setup.py install --streaming-mode chat-full --submodules yes --build yes --start yes --profiles "$(PROFILES)"

install-chat-nonstreaming:
	$(PYTHON) scripts/alpharavis_setup.py install --streaming-mode chat-nonstreaming --submodules yes --build yes --start yes --profiles "$(PROFILES)"

configure:
	$(PYTHON) scripts/alpharavis_setup.py configure

profiles:
	$(PYTHON) scripts/alpharavis_setup.py profiles

streaming:
	$(PYTHON) scripts/alpharavis_setup.py streaming --streaming-mode "$(STREAMING)"

fullstreaming:
	$(PYTHON) scripts/alpharavis_setup.py streaming --streaming-mode full

full-streaming: fullstreaming

hybrid-streaming:
	$(PYTHON) scripts/alpharavis_setup.py streaming --streaming-mode hybrid

nonstreaming:
	$(PYTHON) scripts/alpharavis_setup.py streaming --streaming-mode nonstreaming

chat-completions:
	$(PYTHON) scripts/alpharavis_setup.py streaming --streaming-mode chat

chat-fullstreaming:
	$(PYTHON) scripts/alpharavis_setup.py streaming --streaming-mode chat-full

chat-nonstreaming:
	$(PYTHON) scripts/alpharavis_setup.py streaming --streaming-mode chat-nonstreaming

model-management:
	$(PYTHON) scripts/alpharavis_setup.py model-management

owner-model-management:
	$(PYTHON) scripts/alpharavis_setup.py model-management

media-vision:
	$(PYTHON) scripts/alpharavis_setup.py media-vision

openwebui:
	$(PYTHON) scripts/alpharavis_setup.py openwebui

update:
	$(PYTHON) scripts/alpharavis_setup.py update --streaming-mode "$(UPDATE_STREAMING)" --submodules "$(UPDATE_SUBMODULES)" --build "$(UPDATE_BUILD)" --start "$(UPDATE_START)" --profiles "$(UPDATE_PROFILES)"

update-no-start:
	$(PYTHON) scripts/alpharavis_setup.py update --streaming-mode "$(UPDATE_STREAMING)" --submodules "$(UPDATE_SUBMODULES)" --build yes --start no --profiles "$(UPDATE_PROFILES)"

status:
	$(PYTHON) scripts/alpharavis_setup.py status

up:
	$(COMPOSE) up -d --build

up-fullstreaming:
	$(PYTHON) scripts/alpharavis_setup.py streaming --streaming-mode full
	$(COMPOSE) up -d --build --force-recreate langgraph-api api-bridge

up-chat-fullstreaming:
	$(PYTHON) scripts/alpharavis_setup.py streaming --streaming-mode chat-full
	$(COMPOSE) up -d --build --force-recreate langgraph-api api-bridge

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=120

submodules:
	git submodule update --init --recursive --remote

build:
	$(COMPOSE) build langgraph-api api-bridge hermes-agent media-gallery

bridge-smoke:
	$(PYTHON) scripts/alpharavis_setup.py bridge-smoke

hermes-smoke:
	$(PYTHON) scripts/alpharavis_setup.py hermes-smoke

media-smoke:
	$(PYTHON) scripts/alpharavis_setup.py media-smoke

openwebui-smoke:
	$(PYTHON) scripts/alpharavis_setup.py openwebui-smoke
