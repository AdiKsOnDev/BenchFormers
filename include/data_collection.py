import os
import json
import logging
import pandas as pd

include_logger = logging.getLogger('include')

def collect_data(folders, output_csv="./data/dataset.csv"):
    """
    Iterates through JSON files in a folder, extracts 'main_body' and 'type'
    attributes, and adds them to a pandas DataFrame, which gets saved in a CSV

    Args:
        folders (str): Path to the folder containing JSON files.
        output_csv (str): Path to the output CSV file

    Returns:
        pd.DataFrame: A DataFrame with columns 'main_body' and 'type'.
    """
    data = []

    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith('.json'):
                file_path = os.path.join(folder, filename)
                try:
                    with open(file_path, 'r') as file:
                        json_data = json.load(file)

                        main_body = json_data.get('main_body', None)
                        doc_type = json_data.get('type', None)

                        data.append({'text': main_body, 'label': doc_type})
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    include_logger.error(f"Error processing {filename}", exc_info=True)

    df = pd.DataFrame(data, columns=['text', 'label'])
    df.to_csv(output_csv)
    include_logger.debug(f"DataFrame saved in {output_csv}")

    return df


if __name__ == "__main__":
    collect_data(["./data/dataset/dataset/train/"], 'train.csv')
    collect_data(["./data/dataset/dataset/test/"], 'test.csv')
    collect_data(["./data/dataset/dataset/test"], 'validation.csv')
