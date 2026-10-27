"""
Inference Engine for Calorie & Macronutrient Estimation from Meal Photos
Project: An Intelligent System to Estimate calorie content from meal photographs
Author: PALAGANI MONESHA (23ME1A54C7)
Supervisor: Dr. P. Sudhakar
Domain: Deep Learning & Computer Vision
"""

import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from model import FoodCalorieHybridCNN
from dataset import FOOD_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, load_nutrition_db


class CalorieInferenceEngine:
    """
    Production-ready inference pipeline for meal photo calorie estimation.
    Integrates the CNN Hybrid model with nutritional reference validation.
    """
    def __init__(self, model_path=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.num_classes = len(FOOD_CLASSES)
        self.nutrition_db = load_nutrition_db()
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "best_hybrid_cnn.pth")

        model = FoodCalorieHybridCNN(num_classes=self.num_classes, pretrained=False)
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"[InferenceEngine] Successfully loaded checkpoint: {model_path}")
            except Exception as e:
                print(f"[InferenceEngine] Notice: Using initialized model weights ({e})")
        else:
            print("[InferenceEngine] No checkpoint found. Using initialized hybrid model.")

        model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, image):
        """Preprocesses image into normalized torch tensor"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")

        # Standard 224x224 RGB input
        resized = image.resize((224, 224))
        arr = np.array(resized, dtype=np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_arr = (arr - mean) / std
        tensor = torch.tensor(norm_arr.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)
        return tensor.to(self.device), image

    def estimate(self, image, portion_multiplier=1.0, custom_grams=None):
        """
        Runs dual-head hybrid inference:
        1. Predicts food class and confidence probabilities.
        2. Estimates base calories and macronutrients.
        3. Scales values according to portion adjustments.
        4. Provides actionable dietary feedback.
        """
        tensor, original_img = self.preprocess_image(image)

        with torch.no_grad():
            class_logits, macro_preds = self.model(tensor)
            probs = F.softmax(class_logits, dim=1).squeeze(0).cpu().numpy()
            pred_macros = macro_preds.squeeze(0).cpu().numpy()

        # Top predicted class
        top_idx = int(np.argmax(probs))
        top_food_class = IDX_TO_CLASS[top_idx]
        top_confidence = float(probs[top_idx])

        # Top 3 predictions for user visibility & fallback
        top3_indices = np.argsort(probs)[-3:][::-1]
        top3_predictions = [
            {
                "food_class": IDX_TO_CLASS[i],
                "label": self.nutrition_db.get(IDX_TO_CLASS[i], {}).get("name", IDX_TO_CLASS[i].replace("_", " ").title()),
                "confidence": float(probs[i])
            }
            for i in top3_indices
        ]

        # Retrieve nutritional database facts for guidance
        nutri_info = self.nutrition_db.get(top_food_class, {
            "name": top_food_class.replace("_", " ").title(),
            "category": "General Meal",
            "base_calories_per_100g": 180.0,
            "protein_per_100g": 8.0,
            "carbs_per_100g": 22.0,
            "fat_per_100g": 6.0,
            "fiber_per_100g": 1.5,
            "default_serving_grams": 200,
            "health_grade": "B",
            "advice": "Balanced meal. Maintain portion control to stay within your daily caloric target."
        })

        # Calculate serving weight
        default_grams = nutri_info.get("default_serving_grams", 200)
        if custom_grams is not None and custom_grams > 0:
            final_grams = float(custom_grams)
            multiplier = final_grams / default_grams
        else:
            multiplier = float(portion_multiplier)
            final_grams = default_grams * multiplier

        # Hybrid prediction blending:
        # Combine model neural regression output with empirical nutrition database
        # This provides robust estimation even in early training stages
        scale = final_grams / 100.0
        db_cal = nutri_info["base_calories_per_100g"] * scale
        db_protein = nutri_info["protein_per_100g"] * scale
        db_carbs = nutri_info["carbs_per_100g"] * scale
        db_fat = nutri_info["fat_per_100g"] * scale
        db_fiber = nutri_info["fiber_per_100g"] * scale

        # Neural regressor output
        raw_nn_cal = float(pred_macros[0]) * multiplier
        raw_nn_protein = float(pred_macros[1]) * multiplier
        raw_nn_carbs = float(pred_macros[2]) * multiplier
        raw_nn_fat = float(pred_macros[3]) * multiplier
        raw_nn_fiber = float(pred_macros[4]) * multiplier

        # Dynamic ensemble weighting based on classification confidence
        # When classification confidence is high, the model's conditioned regression is strongly reliable
        alpha = min(max(top_confidence, 0.4), 0.85)
        
        # If neural regression is uninitialized (near zero or negative), lean on database
        if raw_nn_cal < 10.0:
            estimated_calories = db_cal
            estimated_protein = db_protein
            estimated_carbs = db_carbs
            estimated_fat = db_fat
            estimated_fiber = db_fiber
        else:
            estimated_calories = (alpha * raw_nn_cal) + ((1.0 - alpha) * db_cal)
            estimated_protein = (alpha * raw_nn_protein) + ((1.0 - alpha) * db_protein)
            estimated_carbs = (alpha * raw_nn_carbs) + ((1.0 - alpha) * db_carbs)
            estimated_fat = (alpha * raw_nn_fat) + ((1.0 - alpha) * db_fat)
            estimated_fiber = (alpha * raw_nn_fiber) + ((1.0 - alpha) * db_fiber)

        # Macro Calorie Validation (4 kcal/g protein, 4 kcal/g carb, 9 kcal/g fat)
        calculated_macro_cal = (estimated_protein * 4) + (estimated_carbs * 4) + (estimated_fat * 9)
        # Smooth caloric estimate for thermodynamic consistency
        final_calories = round(0.7 * estimated_calories + 0.3 * calculated_macro_cal, 1)

        calorie_density = round((final_calories / final_grams) * 100, 1) if final_grams > 0 else 0

        # Calorie Confidence Interval (+/- 8% to 15% based on classification uncertainty)
        uncertainty_range = max(0.08, (1.0 - top_confidence) * 0.25)
        cal_lower = round(final_calories * (1.0 - uncertainty_range), 1)
        cal_upper = round(final_calories * (1.0 + uncertainty_range), 1)

        return {
            "food_class": top_food_class,
            "food_name": nutri_info["name"],
            "food_category": nutri_info.get("category", "General"),
            "confidence_score": round(top_confidence * 100, 1),
            "top3_candidates": top3_predictions,
            "serving_weight_g": round(final_grams, 1),
            "portion_multiplier": round(multiplier, 2),
            "estimated_calories_kcal": final_calories,
            "calorie_bounds": {
                "min": cal_lower,
                "max": cal_upper
            },
            "calorie_density_per_100g": calorie_density,
            "macronutrients": {
                "protein_g": round(estimated_protein, 1),
                "carbs_g": round(estimated_carbs, 1),
                "fat_g": round(estimated_fat, 1),
                "fiber_g": round(estimated_fiber, 1)
            },
            "health_grade": nutri_info.get("health_grade", "B"),
            "actionable_advice": nutri_info.get("advice", "Maintain balanced intake.")
        }
