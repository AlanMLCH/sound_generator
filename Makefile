PY=python

# Set GPU=0 to run on CPU (no --gpus flag)
GPU ?= 1
ifeq ($(GPU),1)
  GPU_ARGS=--gpus all
else
  GPU_ARGS=
endif

TRAIN_IMAGE=musicgen-train
API_IMAGE=musicgen-api
FRONT_IMAGE=musicgen-front

.PHONY: help docker-build up down sh ingest preprocess fe tune train evaluate test api front all

help:
	@echo "Targets:"
	@echo "  docker-build   Build all images"
	@echo "  up             Start API and Front services"
	@echo "  down           Stop all"
	@echo "  sh             Open shell in trainer (GPU=$(GPU))"
	@echo "  ingest         Data ingestion (download/scan MIDIs + instrument map)"
	@echo "  preprocess     Clean & normalize MIDI -> tokens"
	@echo "  fe             Feature engineering (conditioning, tokens)"
	@echo "  tune           Hyperparameter tuning (grid)"
	@echo "  train          Train best model"
	@echo "  evaluate       Evaluate model"
	@echo "  test           Final testing"
	@echo "  api            Run API only"
	@echo "  front          Run Front only"
	@echo "  all            Full pipeline"

docker-build:
	docker compose build

up:
	docker compose up -d api front

down:
	docker compose down

# Interactive shell in the trainer image
sh:
	docker run --rm -it $(GPU_ARGS) \
	  --shm-size=8g \
	  -e PYTHONUNBUFFERED=1 \
	  -v $(PWD):/app \
	  -w /app \
	  $(TRAIN_IMAGE):latest bash

# ---- Pipeline steps via docker run (works even if compose lacks --gpus) ----

ingest:
	docker run --rm -t $(GPU_ARGS) \
	  --shm-size=8g \
	  -e PYTHONUNBUFFERED=1 \
	  -v $(PWD):/app \
	  -w /app \
	  $(TRAIN_IMAGE):latest \
	  bash -lc "$(PY) -m src.cli.ingest"

preprocess:
	docker run --rm -t $(GPU_ARGS) \
	  --shm-size=8g \
	  -e PYTHONUNBUFFERED=1 \
	  -v $(PWD):/app \
	  -w /app \
	  $(TRAIN_IMAGE):latest \
	  bash -lc "$(PY) -m src.cli.featurize --stage preprocess"

fe:
	docker run --rm -t $(GPU_ARGS) \
	  --shm-size=8g \
	  -e PYTHONUNBUFFERED=1 \
	  -v $(PWD):/app \
	  -w /app \
	  $(TRAIN_IMAGE):latest \
	  bash -lc "$(PY) -m src.cli.featurize --stage features"

tune:
	docker run --rm -t $(GPU_ARGS) \
	  --shm-size=8g \
	  -e PYTHONUNBUFFERED=1 \
	  -v $(PWD):/app \
	  -w /app \
	  $(TRAIN_IMAGE):latest \
	  bash -lc "$(PY) -m src.cli.tune"

train:
	docker run --rm -t $(GPU_ARGS) \
	  --shm-size=8g \
	  -e PYTHONUNBUFFERED=1 \
	  -v $(PWD):/app \
	  -w /app \
	  $(TRAIN_IMAGE):latest \
	  bash -lc "$(PY) -m src.cli.train"

evaluate:
	docker run --rm -t $(GPU_ARGS) \
	  --shm-size=8g \
	  -e PYTHONUNBUFFERED=1 \
	  -v $(PWD):/app \
	  -w /app \
	  $(TRAIN_IMAGE):latest \
	  bash -lc "$(PY) -m src.cli.evaluate"

test:
	docker run --rm -t $(GPU_ARGS) \
	  --shm-size=8g \
	  -e PYTHONUNBUFFERED=1 \
	  -v $(PWD):/app \
	  -w /app \
	  $(TRAIN_IMAGE):latest \
	  bash -lc "$(PY) -m src.cli.test"

api:
	docker compose up -d api

front:
	docker compose up -d front

all: ingest preprocess fe tune train evaluate test
