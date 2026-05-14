PYTHON ?= python
COMPOSE ?= docker compose
STREAMING ?= prompt
SUBMODULES ?= prompt
BUILD ?= prompt
START ?= prompt
PROFILES ?= prompt
ENABLED ?= keep
FPS ?=
MAX_FRAMES ?=
UPDATE_STREAMING ?= prompt
UPDATE_SUBMODULES ?= yes
UPDATE_BUILD ?= yes
UPDATE_START ?= yes
UPDATE_PROFILES ?= prompt
VISION_ENABLED ?=
VISION_URL ?=
VISION_BASE_URL ?=
VISION_MODEL ?=
VISION_FALLBACK ?=
TAILSCALE_HOST ?=
TAILSCALE_SUDO ?= auto
TAILSCALE_DASHBOARD ?= true
TAILSCALE_AUTO ?= apply
TAILSCALE_EXTRA ?=
CONFIG_HOST ?= 127.0.0.1
CONFIG_PORT ?= 8765
VISION_ARGS := --vision-enabled "$(VISION_ENABLED)" --vision-url "$(VISION_URL)" --vision-base-url "$(VISION_BASE_URL)" --vision-model "$(VISION_MODEL)" --vision-fallback "$(VISION_FALLBACK)"
VISION_CONFIG_SET := $(strip $(VISION_ENABLED)$(VISION_URL)$(VISION_BASE_URL)$(VISION_MODEL)$(VISION_FALLBACK))
TAILSCALE_DASHBOARD_ARG := $(if $(filter false no 0,$(TAILSCALE_DASHBOARD)),--exclude-dashboard,)
TAILSCALE_SUDO_ARG := $(if $(filter true yes 1 always,$(TAILSCALE_SUDO)),--sudo,$(if $(filter false no 0 never off,$(TAILSCALE_SUDO)),--sudo-mode never,--sudo-mode auto))

.PHONY: help install install-fullstreaming install-hybrid install-nonstreaming install-chat install-chat-fullstreaming install-chat-nonstreaming config configure profiles streaming fullstreaming full-streaming hybrid-streaming nonstreaming chat-completions chat-fullstreaming chat-nonstreaming model-management owner-model-management media-vision vision-embedding video-analysis openwebui update update-no-start status up up-fullstreaming up-chat-fullstreaming service-dashboard dashboard test-ui tailscale-prep tailscale-auto tailscale-plan tailscale-overrides tailscale-routes-apply tailscale-routes-disable tailscale-apply tailscale-disable disable-tailscale tailscale-status down logs submodules build bridge-smoke hermes-smoke media-smoke openwebui-smoke

help:
	@printf '%s\n' \
		'AlphaRavis Makefile' \
		'' \
		'Install / configure:' \
		'  make config                  # open the local browser UI for editing .env from .env(exaple) defaults' \
		'  make install STREAMING=prompt|responses-full|chat-full START=prompt|yes|no BUILD=prompt|yes|no PROFILES=prompt|none|openwebui # defaults to Tailscale HTTPS mode' \
		'  make update                  # git pull, choose runtime profile, update submodules, build, start, default Tailscale HTTPS mode' \
		'  make install VISION_ENABLED=true VISION_URL=http://host:port/v1 VISION_MODEL=model-name' \
		'  make update VISION_URL=http://host:port/v1 VISION_MODEL=model-name' \
		'  make up VISION_URL=http://host:port/v1 VISION_MODEL=model-name # write .env, then start stack' \
		'  make install-fullstreaming   # Responses full streaming, init submodules, build, start' \
		'  make install-chat-fullstreaming # Chat Completions full streaming, init submodules, build, start' \
		'  make profiles                # show every runtime profile and the .env values it writes' \
		'  make streaming STREAMING=full # only update .env runtime/streaming settings' \
		'  make media-vision VISION_ENABLED=true VISION_URL=http://host:port/v1 VISION_MODEL=model-name' \
		'  make video-analysis ENABLED=true FPS=1 MAX_FRAMES=100' \
		'' \
		'Runtime:' \
		'  make up                      # docker compose up -d --build, including service-dashboard and bridge-test-ui' \
		'  make up-fullstreaming        # set full streaming, then recreate langgraph-api/api-bridge/test UI' \
		'  make up-chat-fullstreaming   # set Chat Completions streaming, then recreate langgraph-api/api-bridge/test UI' \
		'  make service-dashboard       # start only the AlphaRavis service dashboard on port 8090' \
		'  make test-ui                 # start/rebuild only the Bridge test UI on port 8140' \
		'  make down | make logs | make status' \
		'' \
		'Tailscale Serve HTTPS:' \
		'  install/update/up run tailscale-apply by default and bind Docker app ports to 127.0.0.1' \
		'  make install TAILSCALE_AUTO=off # disable Tailscale Serve routes and bind Docker app ports to 0.0.0.0 for LAN HTTP' \
		'  make tailscale-plan TAILSCALE_HOST=device.tailnet.ts.net # includes service-dashboard by default' \
		'  make tailscale-overrides TAILSCALE_HOST=device.tailnet.ts.net' \
		'  make tailscale-apply TAILSCALE_HOST=device.tailnet.ts.net # enable localhost Docker binds, recreate stack, apply HTTPS routes' \
		'  make tailscale-disable       # disable HTTPS routes, switch Docker app ports back to 0.0.0.0, recreate stack' \
		'  make tailscale-plan TAILSCALE_DASHBOARD=false # opt out of dashboard route' \
		'  make tailscale-status' \
		'' \
		'Smoke checks:' \
		'  make bridge-smoke | make hermes-smoke | make media-smoke | make openwebui-smoke'

install:
	$(MAKE) tailscale-prep
	$(PYTHON) scripts/alpharavis_setup.py install --streaming-mode "$(STREAMING)" --submodules "$(SUBMODULES)" --build "$(BUILD)" --start "$(START)" --profiles "$(PROFILES)" $(VISION_ARGS)
	$(MAKE) tailscale-auto

install-fullstreaming:
	$(MAKE) tailscale-prep
	$(PYTHON) scripts/alpharavis_setup.py install --streaming-mode full --submodules yes --build yes --start yes --profiles "$(PROFILES)" $(VISION_ARGS)
	$(MAKE) tailscale-auto

install-hybrid:
	$(MAKE) tailscale-prep
	$(PYTHON) scripts/alpharavis_setup.py install --streaming-mode hybrid --submodules yes --build yes --start yes --profiles "$(PROFILES)" $(VISION_ARGS)
	$(MAKE) tailscale-auto

install-nonstreaming:
	$(MAKE) tailscale-prep
	$(PYTHON) scripts/alpharavis_setup.py install --streaming-mode nonstreaming --submodules yes --build yes --start yes --profiles "$(PROFILES)" $(VISION_ARGS)
	$(MAKE) tailscale-auto

install-chat:
	$(MAKE) tailscale-prep
	$(PYTHON) scripts/alpharavis_setup.py install --streaming-mode chat --submodules yes --build yes --start yes --profiles "$(PROFILES)" $(VISION_ARGS)
	$(MAKE) tailscale-auto

install-chat-fullstreaming:
	$(MAKE) tailscale-prep
	$(PYTHON) scripts/alpharavis_setup.py install --streaming-mode chat-full --submodules yes --build yes --start yes --profiles "$(PROFILES)" $(VISION_ARGS)
	$(MAKE) tailscale-auto

install-chat-nonstreaming:
	$(MAKE) tailscale-prep
	$(PYTHON) scripts/alpharavis_setup.py install --streaming-mode chat-nonstreaming --submodules yes --build yes --start yes --profiles "$(PROFILES)" $(VISION_ARGS)
	$(MAKE) tailscale-auto

config:
	$(PYTHON) scripts/alpharavis_config_server.py --host "$(CONFIG_HOST)" --port "$(CONFIG_PORT)"

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
	$(PYTHON) scripts/alpharavis_setup.py media-vision $(VISION_ARGS)

vision-embedding: media-vision

video-analysis:
	$(PYTHON) scripts/alpharavis_setup.py video-analysis --enabled "$(ENABLED)" --fps "$(FPS)" --max-frames "$(MAX_FRAMES)"

openwebui:
	$(PYTHON) scripts/alpharavis_setup.py openwebui

update:
	$(MAKE) tailscale-prep
	$(PYTHON) scripts/alpharavis_setup.py update --streaming-mode "$(UPDATE_STREAMING)" --submodules "$(UPDATE_SUBMODULES)" --build "$(UPDATE_BUILD)" --start "$(UPDATE_START)" --profiles "$(UPDATE_PROFILES)" $(VISION_ARGS)
	$(MAKE) tailscale-auto

update-no-start:
	$(MAKE) tailscale-prep
	$(PYTHON) scripts/alpharavis_setup.py update --streaming-mode "$(UPDATE_STREAMING)" --submodules "$(UPDATE_SUBMODULES)" --build yes --start no --profiles "$(UPDATE_PROFILES)" $(VISION_ARGS)
	$(MAKE) tailscale-auto

status:
	$(PYTHON) scripts/alpharavis_setup.py status

up:
	$(MAKE) tailscale-prep
ifneq ($(VISION_CONFIG_SET),)
	$(PYTHON) scripts/alpharavis_setup.py media-vision $(VISION_ARGS)
endif
	$(COMPOSE) up -d --build
	$(MAKE) tailscale-auto

up-fullstreaming:
	$(MAKE) tailscale-prep
ifneq ($(VISION_CONFIG_SET),)
	$(PYTHON) scripts/alpharavis_setup.py media-vision $(VISION_ARGS)
endif
	$(PYTHON) scripts/alpharavis_setup.py streaming --streaming-mode full
	$(COMPOSE) up -d --build --force-recreate langgraph-api api-bridge bridge-test-ui
	$(MAKE) tailscale-auto

up-chat-fullstreaming:
	$(MAKE) tailscale-prep
ifneq ($(VISION_CONFIG_SET),)
	$(PYTHON) scripts/alpharavis_setup.py media-vision $(VISION_ARGS)
endif
	$(PYTHON) scripts/alpharavis_setup.py streaming --streaming-mode chat-full
	$(COMPOSE) up -d --build --force-recreate langgraph-api api-bridge bridge-test-ui
	$(MAKE) tailscale-auto

service-dashboard:
	$(COMPOSE) up -d service-dashboard

dashboard: service-dashboard

test-ui:
	$(COMPOSE) up -d --build bridge-test-ui

tailscale-prep:
	@case "$(TAILSCALE_AUTO)" in \
		off|false|no|0|disable|disabled|lan) \
			$(PYTHON) tailscale_https_routes.py disable $(TAILSCALE_DASHBOARD_ARG) $(TAILSCALE_SUDO_ARG) $(if $(TAILSCALE_HOST),--tailscale-host "$(TAILSCALE_HOST)",) $(TAILSCALE_EXTRA); \
			$(PYTHON) scripts/alpharavis_setup.py network-mode --mode lan; \
			;; \
		keep|skip|none) \
			printf '%s\n' 'Keeping current Tailscale/Docker bind mode because TAILSCALE_AUTO=$(TAILSCALE_AUTO)'; \
			;; \
		*) \
			$(PYTHON) scripts/alpharavis_setup.py network-mode --mode tailscale; \
			;; \
	esac

tailscale-auto:
	@case "$(TAILSCALE_AUTO)" in \
		off|false|no|0|disable|disabled|lan) \
			printf '%s\n' 'Tailscale Serve HTTPS disabled for this run; Docker app ports use LAN HTTP binding.'; \
			;; \
		keep|skip|none) \
			printf '%s\n' 'Skipping automatic Tailscale Serve HTTPS setup because TAILSCALE_AUTO=$(TAILSCALE_AUTO)'; \
			;; \
		*) \
			$(MAKE) tailscale-routes-apply; \
			;; \
	esac

tailscale-plan:
	$(PYTHON) tailscale_https_routes.py plan $(TAILSCALE_DASHBOARD_ARG) $(TAILSCALE_SUDO_ARG) $(if $(TAILSCALE_HOST),--tailscale-host "$(TAILSCALE_HOST)",) $(TAILSCALE_EXTRA)

tailscale-overrides:
	$(PYTHON) tailscale_https_routes.py write-overrides $(TAILSCALE_DASHBOARD_ARG) $(TAILSCALE_SUDO_ARG) $(if $(TAILSCALE_HOST),--tailscale-host "$(TAILSCALE_HOST)",) $(TAILSCALE_EXTRA)

tailscale-routes-apply:
	$(PYTHON) tailscale_https_routes.py apply $(TAILSCALE_DASHBOARD_ARG) $(TAILSCALE_SUDO_ARG) $(if $(TAILSCALE_HOST),--tailscale-host "$(TAILSCALE_HOST)",) $(TAILSCALE_EXTRA)
	$(COMPOSE) up -d service-dashboard

tailscale-routes-disable:
	$(PYTHON) tailscale_https_routes.py disable $(TAILSCALE_DASHBOARD_ARG) $(TAILSCALE_SUDO_ARG) $(if $(TAILSCALE_HOST),--tailscale-host "$(TAILSCALE_HOST)",) $(TAILSCALE_EXTRA)

tailscale-apply:
	$(PYTHON) scripts/alpharavis_setup.py network-mode --mode tailscale
	$(COMPOSE) up -d
	$(MAKE) tailscale-routes-apply

tailscale-disable:
	$(MAKE) tailscale-routes-disable
	$(PYTHON) scripts/alpharavis_setup.py network-mode --mode lan
	$(COMPOSE) up -d

disable-tailscale: tailscale-disable

tailscale-status:
	$(PYTHON) tailscale_https_routes.py status $(TAILSCALE_SUDO_ARG)

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=120

submodules:
	git submodule update --init --recursive --remote

build:
	$(COMPOSE) build langgraph-api api-bridge bridge-test-ui hermes-agent media-gallery

bridge-smoke:
	$(PYTHON) scripts/alpharavis_setup.py bridge-smoke

hermes-smoke:
	$(PYTHON) scripts/alpharavis_setup.py hermes-smoke

media-smoke:
	$(PYTHON) scripts/alpharavis_setup.py media-smoke

openwebui-smoke:
	$(PYTHON) scripts/alpharavis_setup.py openwebui-smoke
