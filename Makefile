PYTHON ?= python
COMPOSE ?= docker compose

.PHONY: install configure update status up down logs submodules build bridge-smoke hermes-smoke

install:
	$(PYTHON) scripts/alpharavis_setup.py install

configure:
	$(PYTHON) scripts/alpharavis_setup.py configure

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
	$(COMPOSE) build langgraph-api api-bridge hermes-agent

bridge-smoke:
	$(PYTHON) scripts/alpharavis_setup.py bridge-smoke

hermes-smoke:
	$(PYTHON) scripts/alpharavis_setup.py hermes-smoke
