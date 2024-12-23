DATA_DIR = ./data
ZIP_FILE = $(DATA_DIR)/datasets.zip
VENV_DIR = ./.venv/
DATASET_DIR = $(DATA_DIR)/dataset
DATASET_URL = http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip

all: collect 
	
collect: unzip run_script clean

setup:
	pip install -r requirements.txt


test: setup
	python -m unittest discover -s tests

unzip: $(ZIP_FILE)
	unzip -o $(ZIP_FILE) -d $(DATA_DIR)/dataset

run_script: unzip
	python ./include/data_collection.py

clean:
	rm -rf $(DATASET_DIR)

$(ZIP_FILE):
	curl -o $(ZIP_FILE) $(DATASET_URL)

.PHONY: unzip run_script clean all
