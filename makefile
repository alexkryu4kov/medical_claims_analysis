CODE = slides *.py


.PHONY: pretty
pretty: ## Prettifying all files
	poetry run isort $(CODE)
	poetry run black $(CODE)