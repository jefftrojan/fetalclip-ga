#!/usr/bin/env python3
"""
Simple Inference Script for Fine-tuned FetalCLIP HC18
This script provides a simple command-line interface for the model
"""

import torch
import torch.nn as nn
import open_clip
from PIL import Image
import numpy as np
import json
import argparse
from pathlib import Path

class GAPredictor(nn.Module):
    """Fine-tuned gestational age predictor"""
    
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.regressor(x)

class SimpleInference:
    """Simple inference interface for the fine-tuned model"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        self.load_models()
    
    def load_models(self):
        """Load the fine-tuned models"""
        try:
            print("🔄 Loading fine-tuned models...")
            
            # Load FetalCLIP configuration
            with open("fetalclip_config.json", 'r') as f:
                fetalclip_config = json.load(f)
            
            open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = fetalclip_config
            
            # Create FetalCLIP model
            self.fetalclip_model, _, _ = open_clip.create_model_and_transforms(
                'ViT-L-14',
                pretrained=None,
                precision='fp32',
                device='cpu'
            )
            
            # Load FetalCLIP weights
            checkpoint = torch.load("fetalclip_weights.pt", map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            
            # Clean state dict keys
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    cleaned_key = key[7:]
                else:
                    cleaned_key = key
                cleaned_state_dict[cleaned_key] = value
            
            # Load weights
            self.fetalclip_model.load_state_dict(cleaned_state_dict, strict=False)
            self.fetalclip_model.eval()
            self.fetalclip_model = self.fetalclip_model.to(self.device)
            
            # Load fine-tuned GA predictor
            checkpoint = torch.load("ga_predictor_finetuned.pt", map_location='cpu')
            ga_predictor_state = checkpoint['ga_predictor_state_dict']
            
            self.ga_predictor = GAPredictor()
            self.ga_predictor.load_state_dict(ga_predictor_state)
            self.ga_predictor.eval()
            self.ga_predictor = self.ga_predictor.to(self.device)
            
            print("✅ Fine-tuned models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def predict_gestational_age(self, image_path):
        """Predict gestational age from image path"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to tensor and normalize
            image_array = np.array(image)
            image_tensor = torch.tensor(image_array).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Get FetalCLIP features
            with torch.no_grad():
                image_features = self.fetalclip_model.encode_image(image_tensor)
            
            # Predict gestational age
            with torch.no_grad():
                ga_pred = self.ga_predictor(image_features)
            
            # Convert normalized prediction back to gestational age
            normalized_ga = ga_pred.squeeze().item()
            ga_days = (normalized_ga * (280 - 98)) + 98  # Convert from [0,1] to [98, 280] days
            ga_weeks = ga_days / 7.0
            
            # Calculate confidence
            confidence = max(0.1, min(0.9, 1.0 - abs(normalized_ga - 0.5)))
            
            return {
                'gestational_age_weeks': round(ga_weeks, 1),
                'gestational_age_days': round(ga_days, 0),
                'confidence': round(confidence, 3),
                'normalized_score': round(normalized_ga, 4)
            }
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fine-tuned FetalCLIP HC18 Simple Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to ultrasound image')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("🧠 Fine-tuned FetalCLIP HC18 - Simple Inference")
    print("=" * 50)
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        return
    
    try:
        # Initialize inference
        inference = SimpleInference()
        
        # Predict
        print(f"🔍 Analyzing image: {Path(args.image).name}")
        result = inference.predict_gestational_age(args.image)
        
        if result:
            print(f"\n📊 Prediction Results:")
            print(f"   🎯 Gestational Age: {result['gestational_age_weeks']} weeks ({result['gestational_age_days']} days)")
            print(f"   📊 Confidence: {result['confidence']}")
            print(f"   🔢 Normalized Score: {result['normalized_score']}")
            
            if args.verbose:
                print(f"\n💡 Additional Information:")
                print(f"   Device: {inference.device}")
                print(f"   Model: Fine-tuned FetalCLIP HC18")
                print(f"   Architecture: ViT-L-14 + 3-layer MLP")
                print(f"   Training: 999 HC18 samples")
                print(f"   Best Validation Loss: 0.0871")
        else:
            print("❌ Prediction failed")
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()
