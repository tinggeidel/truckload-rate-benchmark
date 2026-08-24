.PHONY: help install demo test clean

help:
	@echo "make install   install dependencies"
	@echo "make demo      generate synthetic data and run the full pipeline"
	@echo "make test      run the test suite (offline, no data required)"
	@echo "make clean     remove generated data, keep the reference tables"

install:
	python -m pip install -r requirements.txt

demo:
	python run_pipeline.py

test:
	python -m pytest -q

clean:
	python -c "import shutil,pathlib; d=pathlib.Path('data'); [shutil.rmtree(p) if p.is_dir() else p.unlink() for p in d.iterdir() if p.name != 'reference']"
