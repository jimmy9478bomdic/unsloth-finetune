import os
import json
from huggingface_hub import HfApi
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

def verbose_print(message):
    print(f"[INFO] {message}")

def parse_markdown(file_path):
    verbose_print(f"Parsing Markdown file {file_path}")
    with open(file_path, 'r') as file:
        content = file.read()
    sections = content.split('\n## ')
    parsed_sections = [section.replace('\n', ' ') for section in sections]
    return parsed_sections

def create_dataset():
    dataset = []

    # Scrape GitHub repository

    markdown_files = [
        "./ijms-25-00574-v2/ijms-25-00574-v2.md"
        ]
    for md_file in markdown_files:
        sections = parse_markdown(md_file)
        for section in sections:
            dataset.append({
                'source': '論文',
                'file': md_file,
                'label': '疲勞',
                'content': section
            })

    
    return dataset


def load_dataset_locally(file_path):

    if os.path.exists(file_path):
        verbose_print(f"Loading existing dataset from {file_path}")
        with open(file_path, 'r') as file:
            return json.load(file)
    verbose_print(f"No existing dataset found at {file_path}")
    return []

def save_dataset_locally(dataset, output_file):
    verbose_print(f"Saving dataset to {output_file}")
    with open(output_file, 'w') as file:
        json.dump(dataset, file, indent=4)
    verbose_print("Dataset saved successfully")

def upload_to_huggingface(dataset, repo_id):
    token = os.getenv("HF_TOKEN")
    verbose_print(f"Uploading dataset to Hugging Face with repository ID {repo_id}")
    hf_api = HfApi()
    hf_api.create_repo(repo_id, token=token, repo_type="dataset", private=False)

    # Create a DatasetDict and push to hub
    dataset_dict = DatasetDict({"train": Dataset.from_list(dataset)})
    dataset_dict.push_to_hub(repo_id, token=token)
    verbose_print(f"Dataset uploaded to Hugging Face with repository ID {repo_id}")

output_file = 'test_dataset.json'
repo_id = 'jimmy9478/test_dataset'

verbose_print("Starting dataset creation process")
existing_dataset = load_dataset_locally(output_file)
new_dataset = create_dataset()
combined_dataset = existing_dataset + new_dataset
save_dataset_locally(combined_dataset, output_file)
upload_to_huggingface(combined_dataset, repo_id)
verbose_print("Dataset creation and upload process completed")
