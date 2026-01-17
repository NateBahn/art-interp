.PHONY: help install install-dev test lint format typecheck check clean run-server run-web

help:
	@echo "Available commands:"
	@echo "  make install      - Install package in editable mode"
	@echo "  make install-dev  - Install with all dev dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linter"
	@echo "  make format       - Auto-format code"
	@echo "  make typecheck    - Run type checker"
	@echo "  make check        - Run lint + typecheck + test"
	@echo "  make run-server   - Start API server"
	@echo "  make run-web      - Start web frontend"
	@echo "  make clean        - Remove build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[all]"
	cd web && npm install

test:
	pytest tests/ -v

lint:
	ruff check src/ server/ tests/

format:
	ruff format src/ server/ tests/
	ruff check --fix src/ server/ tests/

typecheck:
	mypy src/ server/

check: lint typecheck test

run-server:
	uvicorn server.main:app --reload --port 8001

run-web:
	cd web && npm run dev

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
