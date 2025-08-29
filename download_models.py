#!/usr/bin/env python3
"""
Download Models Script for Fine-tuned FetalCLIP HC18
This script downloads the required model files from the source repository
"""

import os
import sys
import requests
from pathlib import Path
import hashlib

# Model file information
MODEL_FILES = {
    "fetalclip_weights.pt": {
        "size_mb": 1632,
        "description": "FetalCLIP model weights (1.6GB)",
        "url": "TO_BE_PROVIDED"  # This will be updated with actual URL
    },
    "ga_predictor_finetuned.pt": {
        "size_mb": 6,
        "description": "Fine-tuned GA predictor (6MB)",
        "url": "TO_BE_PROVIDED"  # This will be updated with actual URL
    }
}

def calculate_file_hash(file_path):
    """Calculate SHA256 hash of a file"""
    if not os.path.exists(file_path):
        return None
    
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def download_file(url, filename, expected_size_mb):
    """Download a file with progress bar"""
    try:
        print(f"📥 Downloading {filename} ({expected_size_mb}MB)...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress bar
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        bar_length = 30
                        filled_length = int(bar_length * downloaded // total_size)
                        bar = '█' * filled_length + '-' * (bar_length - filled_length)
                        print(f"\r[{bar}] {percent:.1f}%", end='', flush=True)
        
        print(f"\n✅ Downloaded {filename} successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {filename}: {e}")
        return False

def main():
    """Main download function"""
    print("🚀 Fine-tuned FetalCLIP HC18 - Model Downloader")
    print("=" * 50)
    
    # Check if models already exist
    existing_models = []
    missing_models = []
    
    for filename, info in MODEL_FILES.items():
        if os.path.exists(filename):
            existing_models.append(filename)
            print(f"✅ {filename} already exists")
        else:
            missing_models.append(filename)
            print(f"❌ {filename} missing")
    
    if not missing_models:
        print("\n🎉 All model files are already present!")
        return
    
    print(f"\n📋 Missing models: {len(missing_models)}")
    print(f"📋 Existing models: {len(existing_models)}")
    
    # Check if URLs are provided
    urls_provided = all(info["url"] != "TO_BE_PROVIDED" for info in MODEL_FILES.values())
    
    if not urls_provided:
        print("\n⚠️  Model download URLs not configured yet.")
        print("Please update the download_models.py script with the correct URLs.")
        print("\nAlternatively, you can manually copy the model files:")
        print("1. Copy fetalclip_weights.pt from the source directory")
        print("2. Copy ga_predictor_finetuned.pt from fine_tune_output/checkpoints/")
        return
    
    # Download missing models
    print(f"\n🚀 Starting download of {len(missing_models)} model files...")
    
    for filename in missing_models:
        info = MODEL_FILES[filename]
        success = download_file(info["url"], filename, info["size_mb"])
        
        if not success:
            print(f"❌ Failed to download {filename}")
            return
    
    print("\n🎉 All model files downloaded successfully!")
    print("\n🚀 Next steps:")
    print("   1. Run setup: python setup.py")
    print("   2. Start Streamlit: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
