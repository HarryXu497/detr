test:
	.venv/bin/python -m pytest -q

typecheck:
	.venv/bin/python -m pyright

check: test typecheck