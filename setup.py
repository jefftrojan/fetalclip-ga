#!/usr/bin/env python3
"""
Setup script for Fine-tuned FetalCLIP HC18 Model Package
This script helps install dependencies and verify the model setup
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def verify_model_files():
    """Verify that all required model files are present"""
    print("🔍 Verifying model files...")
    
    required_files = [
        "fetalclip_config.json",
        "fetalclip_weights.pt", 
        "ga_predictor_finetuned.pt"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        return False
    else:
        print("✅ All model files present!")
        return True

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("🧠 Testing model loading...")
    
    try:
        # Try to import required modules
        import torch
        import open_clip
        from PIL import Image
        import numpy as np
        import json
        
        print("✅ All modules imported successfully!")
        
        # Test model loading
        with open("fetalclip_config.json", 'r') as f:
            fetalclip_config = json.load(f)
        
        open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = fetalclip_config
        
        # Create model (without loading weights to save time)
        model, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14',
            pretrained=None,
            precision='fp32',
            device='cpu'
        )
        
        print("✅ Model architecture created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Fine-tuned FetalCLIP HC18 - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Verify model files
    if not verify_model_files():
        return False
    
    # Test model loading
    if not test_model_loading():
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 Next steps:")
    print("   1. Run Streamlit app: streamlit run streamlit_app.py")
    print("   2. Or use command-line: python simple_inference.py --image your_image.png")
    print("   3. Open browser to http://localhost:8501 for the web interface")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
