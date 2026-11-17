LIB = eotorch
# Pinned, as new ruff versions enable more lint rules
RUFF = uvx ruff@0.17.4

check: lint typecheck test doctest

build: typecheck test
	python -m build

lint:
	$(RUFF) check $(LIB) tests
	$(RUFF) format --check $(LIB) tests

format:
	$(RUFF) format $(LIB) tests

test:
	pytest --disable-warnings

typecheck:
	mypy $(LIB)/ --config-file pyproject.toml

doctest:
	pytest --doctest-modules $(LIB)

coverage: 
	pytest --cov-report html --cov=$(LIB) tests/

docs: FORCE
	mkdocs build

clean:
	python -c "import shutil; shutil.rmtree('dist', ignore_errors=True)"
	python -c "import shutil; shutil.rmtree('htmlcov', ignore_errors=True)"
	python -c "import os; os.remove('.coverage') if os.path.exists('.coverage') else None"
	python -c "import shutil; shutil.rmtree('site', ignore_errors=True)"

FORCE: