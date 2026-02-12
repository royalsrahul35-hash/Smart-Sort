# dataset_downloader.py

import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Ensure the .kaggle directory exists
kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

if not os.path.exists(kaggle_json):
    raise FileNotFoundError("❌ kaggle.json not found in ~/.kaggle")

# Set permissions (Linux/Mac only; ignored on Windows)
os.chmod(kaggle_json, 0o600)

# Authenticate and download
api = KaggleApi()
api.authenticate()

# Download dataset
api.dataset_download_files(
    'muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten',
    path='data',
    unzip=True
)

print("✅ Dataset downloaded and unzipped into the 'data' folder.")
