clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf

env:
	poetry env activate
	poetry install --no-root

format:
	poetry run ruff check src/coref_onnx/*.py --fix
	poetry run ruff format src/coref_onnx/*.py
	poetry run isort src/coref_onnx/*.py
	poetry run black src/coref_onnx/*.py

test:
	poetry run pytest -s tests/

help:
	@echo "Available commands:"
	@echo "  clean            - Remove temporary files"
	@echo "  env          	  - Create virtualenv and install dependencies"
	@echo "  format           - Format code"
	@echo "  test             - Run test suite"
