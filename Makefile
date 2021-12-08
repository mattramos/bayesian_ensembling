make clean:
	black py38 -l 100 src/
	isort --atomic -l 100 --trailing-comma --remove-redundant-aliases --multi-line 3 src/

make notebooks:
	jupytext --to notebook nbs/*.py

make tests:
	pytest -n auto -v tests/

make environment_update:
	conda env export > environment.yml