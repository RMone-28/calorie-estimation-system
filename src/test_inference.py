"""
Unit Test & Quality Verification for Calorie Inference Engine
Author: PALAGANI MONESHA (23ME1A54C7)
Supervisor: Dr. P. Sudhakar
"""

import os
import sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import CalorieInferenceEngine
from dataset import generate_synthetic_meal_sample, FOOD_CLASSES


def test_inference_pipeline():
    print("Testing CalorieInferenceEngine initialization...")
    engine = CalorieInferenceEngine()
    print("InferenceEngine successfully initialized.")

    test_dishes = ["biryani", "pizza", "salad", "burger", "grilled_chicken"]
    all_passed = True

    print("\nRunning inference tests across benchmark dishes:")
    print("-" * 65)

    for dish in test_dishes:
        img = generate_synthetic_meal_sample(dish, portion_factor=1.0)
        res = engine.estimate(img, portion_multiplier=1.0)

        cal = res["estimated_calories_kcal"]
        macros = res["macronutrients"]
        grade = res["health_grade"]

        print(f"Dish: {res['food_name']:<30} | Cal: {cal:>5.1f} kcal | P: {macros['protein_g']:>4.1f}g | C: {macros['carbs_g']:>4.1f}g | F: {macros['fat_g']:>4.1f}g | Grade: {grade}")

        # Basic sanity checks
        if cal <= 0:
            print(f"ERROR: Non-positive calories for {dish}")
            all_passed = False
        if res["confidence_score"] <= 0:
            print(f"ERROR: Non-positive confidence for {dish}")
            all_passed = False

    print("-" * 65)
    if all_passed:
        print("ALL INFERENCE SANITY TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED.")

    return all_passed


if __name__ == "__main__":
    test_inference_pipeline()
