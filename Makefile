PYTHON ?= python
COMPOSE ?= docker compose

.PHONY: install configure model-management owner-model-management media-vision openwebui update status up down logs submodules build bridge-smoke hermes-smoke media-smoke openwebui-smoke

install:
	$(PYTHON) scripts/alpharavis_setup.py install

configure:
	$(PYTHON) scripts/alpharavis_setup.py configure

model-management:
	$(PYTHON) scripts/alpharavis_setup.py model-management

owner-model-management:
	$(PYTHON) scripts/alpharavis_setup.py model-management

media-vision:
	$(PYTHON) scripts/alpharavis_setup.py media-vision

openwebui:
	$(PYTHON) scripts/alpharavis_setup.py openwebui

update:
	$(PYTHON) scripts/alpharavis_setup.py update

status:
	$(PYTHON) scripts/alpharavis_setup.py status

up:
	$(COMPOSE) up -d --build

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
