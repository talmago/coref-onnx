clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf

env:
	poetry env activate
	poetry install --no-root

format:
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run isort .
	poetry run black .

test:
	poetry run pytest -s tests/

help:
	@echo "Available commands:"
	@echo "  clean            - Remove temporary files"
	@echo "  env          	  - Create virtualenv and install dependencies"
	@echo "  format           - Format code"
	@echo "  test             - Run test suite"
