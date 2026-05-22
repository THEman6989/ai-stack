SHELL := /usr/bin/env bash

PROJECT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
FIRMWARE_DIR := $(PROJECT_DIR)firmware/nodemcu-v3

PORT ?= /dev/ttyUSB0
BAUD ?= 9600
FIRMWARE ?= true
FIREWALL ?= true
FIREWALL_LAN ?= 192.168.178.0/24
LLAMA_FIREWALL_PORTS ?= 8001
SUDO ?= sudo

INSTALL_TARGETS := check
ifeq ($(FIRMWARE),true)
INSTALL_TARGETS += firmware-upload
endif
INSTALL_TARGETS += manager-install
ifeq ($(FIREWALL),true)
INSTALL_TARGETS += firewall-allow
endif
INSTALL_TARGETS += status

.PHONY: help install check test manager-install manager-restart firewall-allow \
	firewall-api-lan firewall-llama-lan firewall-lan firewall-wlan \
	firmware-build firmware-upload firmware-flash firmware-monitor flash upload \
	monitor status

help:
	@printf '%s\n' 'Ubuntu Llama Manager'
	@printf '%s\n' ''
	@printf '%s\n' 'Common targets:'
	@printf '%s\n' '  make install          Check, flash firmware, install systemd units, open firewall'
	@printf '%s\n' '  make check            Validate config and run unit tests'
	@printf '%s\n' '  make manager-install  Run sudo ./install.sh apply'
	@printf '%s\n' '  make firewall-lan     Allow API and selected Llama ports from LAN/WLAN only'
	@printf '%s\n' '  make firewall-wlan    Same as firewall-lan'
	@printf '%s\n' '  make firewall-llama-lan'
	@printf '%s\n' '                        Allow Llama port(s) from LAN/WLAN only, default: 8001'
	@printf '%s\n' '  make flash            Build and upload NodeMCU firmware'
	@printf '%s\n' '  make monitor          Open serial monitor'
	@printf '%s\n' '  make status           Show systemd status'
	@printf '%s\n' ''
	@printf '%s\n' 'Useful variables:'
	@printf '%s\n' '  PORT=/dev/ttyUSB0     Serial port for firmware upload/monitor'
	@printf '%s\n' '  BAUD=9600             Serial monitor baud rate'
	@printf '%s\n' '  FIRMWARE=false        Skip firmware upload during make install'
	@printf '%s\n' '  FIREWALL=false        Skip UFW allow during make install'
	@printf '%s\n' '  FIREWALL_LAN=192.168.178.0/24'
	@printf '%s\n' '                        Local LAN/WLAN subnet allowed by firewall targets'
	@printf '%s\n' '  LLAMA_FIREWALL_PORTS="8001"'
	@printf '%s\n' '                        Llama ports opened by firewall-llama-lan'
	@printf '%s\n' ''
	@printf '%s\n' 'Examples:'
	@printf '%s\n' '  make install'
	@printf '%s\n' '  make install PORT=/dev/ttyUSB0'
	@printf '%s\n' '  make install FIRMWARE=false'
	@printf '%s\n' '  make firewall-llama-lan'
	@printf '%s\n' '  make firewall-llama-lan LLAMA_FIREWALL_PORTS="8033 8001"'

install: $(INSTALL_TARGETS)

check:
	./install.sh check
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py' -v

test:
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py' -v

manager-install:
	$(SUDO) ./install.sh apply

manager-restart:
	$(SUDO) ./install.sh restart

firewall-allow:
	$(SUDO) ./install.sh firewall-allow

firewall-api-lan:
	$(SUDO) env API_FIREWALL_ALLOW_FROM="$(FIREWALL_LAN)" ./install.sh firewall-allow

firewall-llama-lan:
	@for port in $(LLAMA_FIREWALL_PORTS); do \
		printf 'Allowing LAN %s to llama port %s/tcp\n' "$(FIREWALL_LAN)" "$$port"; \
		$(SUDO) ufw allow from "$(FIREWALL_LAN)" to any port "$$port" proto tcp comment "ubuntu-llama-manager llama $$port LAN"; \
	done
	$(SUDO) ufw reload
	$(SUDO) ufw status verbose

firewall-lan: firewall-api-lan firewall-llama-lan

firewall-wlan: firewall-lan

firmware-build:
	cd "$(FIRMWARE_DIR)" && ./build.sh

firmware-upload:
	cd "$(FIRMWARE_DIR)" && PORT="$(PORT)" ./upload.sh

firmware-flash: firmware-upload

flash: firmware-upload

upload: firmware-upload

firmware-monitor:
	cd "$(FIRMWARE_DIR)" && PORT="$(PORT)" BAUD="$(BAUD)" ./monitor.sh

monitor: firmware-monitor

status:
	./install.sh status
