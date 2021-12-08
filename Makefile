make clean:
	black -l 100 ensembles/
	isort --atomic -l 100 --trailing-comma --remove-redundant-aliases --multi-line 3 ensembles/

make notebooks:
	jupytext --to notebook nbs/*.py

make tests:
	pytest -n auto -v tests/

make environment_update:
	conda env export > environment.yml