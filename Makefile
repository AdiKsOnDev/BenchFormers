DATA_DIR = ./data
ZIP_FILE = $(DATA_DIR)/datasets.zip
DATASET_DIR = $(DATA_DIR)/dataset
DATASET_URL = http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip

all: collect 
	
# Target to unzip datasets.zip into the ./data/dataset/ directory
unzip: $(ZIP_FILE)
	unzip -o $(ZIP_FILE) -d $(DATA_DIR)/dataset

# Target to run the Python script
run_script: unzip
	python ./include/data_collection.py

clean:
	rm -rf $(DATASET_DIR)

$(ZIP_FILE):
	curl -o $(ZIP_FILE) $(DATASET_URL)

collect: unzip run_script clean

.PHONY: unzip run_script clean all
