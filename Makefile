DATA_DIR = ./data
ZIP_FILE = $(DATA_DIR)/datasets.zip
VENV_DIR = ./.venv/
DATASET_DIR = $(DATA_DIR)/dataset
DATASET_URL = http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip

all: collect 
	
collect: unzip run_script clean

setup:
	pip install -r requirements.txt

run1k:
	python main.py --dataset_size 1000 --results_dir "./results/1000Samples/" --model "roformer"
	python main.py --dataset_size 1000 --results_dir "./results/1000Samples/" --model "longformer"
	python main.py --dataset_size 1000 --results_dir "./results/1000Samples/" --model "bigbird"
	python main.py --dataset_size 1000 --results_dir "./results/1000Samples/" --model "legalbert"
	python main.py --dataset_size 1000 --results_dir "./results/1000Samples/" --model "modernbert"

run2k:
	python main.py --dataset_size 2500 --results_dir "./results/2500Samples/" --model "roformer"
	python main.py --dataset_size 2500 --results_dir "./results/2500Samples/" --model "longformer"
	python main.py --dataset_size 2500 --results_dir "./results/2500Samples/" --model "bigbird"
	python main.py --dataset_size 2500 --results_dir "./results/2500Samples/" --model "legalbert"
	python main.py --dataset_size 2500 --results_dir "./results/2500Samples/" --model "modernbert"

run5k:
	python main.py --dataset_size 5000 --results_dir "./results/5000Samples/" --model "roformer"
	python main.py --dataset_size 5000 --results_dir "./results/5000Samples/" --model "longformer"
	python main.py --dataset_size 5000 --results_dir "./results/5000Samples/" --model "bigbird"
	python main.py --dataset_size 5000 --results_dir "./results/5000Samples/" --model "legalbert"
	python main.py --dataset_size 5000 --results_dir "./results/5000Samples/" --model "modernbert"

run10k:
	python main.py --dataset_size 10000 --results_dir "./results/10000Samples/" --model "roformer"
	python main.py --dataset_size 10000 --results_dir "./results/10000Samples/" --model "longformer"
	python main.py --dataset_size 10000 --results_dir "./results/10000Samples/" --model "bigbird"
	python main.py --dataset_size 10000 --results_dir "./results/10000Samples/" --model "legalbert"
	python main.py --dataset_size 10000 --results_dir "./results/10000Samples/" --model "modernbert"

eval1k:
	python validation.py --dir "./results/1000Samples/" --model "roformer"
	python validation.py --dir "./results/1000Samples/" --model "longformer"
	python validation.py --dir "./results/1000Samples/" --model "bigbird"
	python validation.py --dir "./results/1000Samples/" --model "legalbert"
	python validation.py --dir "./results/1000Samples/" --model "modernbert"

eval2k:
	python validation.py --dir "./results/2500Samples/" --model "roformer"
	python validation.py --dir "./results/2500Samples/" --model "longformer"
	python validation.py --dir "./results/2500Samples/" --model "bigbird"
	python validation.py --dir "./results/2500Samples/" --model "legalbert"
	python validation.py --dir "./results/2500Samples/" --model "modernbert"

eval5k:
	python validation.py --dir "./results/5000Samples/" --model "roformer"
	python validation.py --dir "./results/5000Samples/" --model "longformer"
	python validation.py --dir "./results/5000Samples/" --model "bigbird"
	python validation.py --dir "./results/5000Samples/" --model "legalbert"
	python validation.py --dir "./results/5000Samples/" --model "modernbert"

eval10k:
	python validation.py --dir "./results/10000Samples/" --model "roformer"
	python validation.py --dir "./results/10000Samples/" --model "longformer"
	python validation.py --dir "./results/10000Samples/" --model "bigbird"
	python validation.py --dir "./results/10000Samples/" --model "legalbert"
	python validation.py --dir "./results/10000Samples/" --model "modernbert"

test: setup
	python -m unittest discover -s tests

unzip: $(ZIP_FILE)
	unzip -o $(ZIP_FILE) -d $(DATA_DIR)/dataset

run_script: unzip
	python ./include/data_collection.py

clean:
	rm -rf $(DATASET_DIR)
	rm -rf results

$(ZIP_FILE):
	curl -o $(ZIP_FILE) $(DATASET_URL)

.PHONY: unzip run_script clean all
