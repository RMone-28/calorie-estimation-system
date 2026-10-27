"""
Dataset loader and preprocessing pipeline for Food Calorie Estimation
Project: An Intelligent System to Estimate calorie content from meal photographs
Author: PALAGANI MONESHA (23ME1A54C7)
Supervisor: Dr. P. Sudhakar
Domain: Deep Learning & Computer Vision
"""

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader

# 12 Food Categories with Ground Truth Nutrition Profiles
FOOD_CLASSES = [
    "pizza",
    "burger",
    "salad",
    "biryani",
    "pasta",
    "grilled_chicken",
    "sushi",
    "dosa",
    "fruit_bowl",
    "french_fries",
    "oatmeal",
    "sandwich"
]

CLASS_TO_IDX = {name: i for i, name in enumerate(FOOD_CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(FOOD_CLASSES)}


def load_nutrition_db(db_path=None):
    """Loads nutritional reference facts from JSON database"""
    if db_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(base_dir, "data", "nutrition_db.json")
    
    if os.path.exists(db_path):
        with open(db_path, "r") as f:
            return json.load(f)
    return {}


class MealDataset(Dataset):
    """
    Meal Image Dataset for Multi-Task Learning.
    Returns:
        - Image tensor [3, 224, 224]
        - Class index (long)
        - Target vector: [Calories, Protein, Carbs, Fat, Fiber] (float tensor)
    """
    def __init__(self, data_list, transform=None):
        self.data_list = data_list  # List of dicts with: image_path or PIL, class_idx, targets
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = item.get("image")
        if image is None and "image_path" in item:
            image = item["image_path"]

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")

        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Fallback simple tensor transform
            image_resized = image.resize((224, 224))
            arr = np.array(image_resized, dtype=np.float32) / 255.0
            # Normalize with ImageNet mean/std
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
            image_tensor = torch.tensor(arr.transpose((2, 0, 1)), dtype=torch.float32)

        class_target = torch.tensor(item["class_idx"], dtype=torch.long)
        targets = torch.tensor(item["targets"], dtype=torch.float32)

        return image_tensor, class_target, targets


def generate_synthetic_meal_sample(food_class, portion_factor=1.0, size=(300, 300)):
    """
    Procedurally creates a realistic synthetic meal photo for testing and demonstration.
    Renders plate, food components, textures, and color variations.
    """
    img = Image.new("RGB", size, color=(242, 238, 230))  # Tabletop background
    draw = ImageDraw.Draw(img)

    # 1. Ceramic Plate / Bowl
    plate_bbox = [25, 25, size[0] - 25, size[1] - 25]
    draw.ellipse(plate_bbox, fill=(255, 255, 255), outline=(210, 210, 210), width=4)
    # Inner rim shadow
    draw.ellipse([45, 45, size[0] - 45, size[1] - 45], outline=(235, 235, 235), width=2)

    center_x, center_y = size[0] // 2, size[1] // 2
    r = int(90 * min(portion_factor, 1.4))

    # 2. Food-specific visual renderings
    if food_class == "pizza":
        # Pizza triangular slice with crust, cheese, and pepperonis
        points = [(center_x, center_y - r), (center_x - r, center_y + r), (center_x + r, center_y + r)]
        draw.polygon(points, fill=(245, 185, 66))  # Melted cheese
        draw.arc([center_x - r, center_y + r - 15, center_x + r, center_y + r + 15], 0, 180, fill=(186, 114, 39), width=14)  # Crust
        # Pepperoni slices
        for ox, oy in [(-15, 10), (20, 25), (0, -20), (-25, 50), (15, 55)]:
            draw.ellipse([center_x + ox - 12, center_y + oy - 12, center_x + ox + 12, center_y + oy + 12], fill=(180, 40, 20))

    elif food_class == "burger":
        # Bun, patty, lettuce, cheese
        draw.ellipse([center_x - r, center_y - r // 2, center_x + r, center_y + r // 2], fill=(212, 140, 68)) # Top bun
        draw.rectangle([center_x - r + 10, center_y - 10, center_x + r - 10, center_y + 15], fill=(89, 44, 21)) # Patty
        draw.rectangle([center_x - r + 15, center_y - 2, center_x + r - 15, center_y + 8], fill=(255, 204, 0)) # Cheese
        draw.rectangle([center_x - r + 5, center_y + 12, center_x + r - 5, center_y + 25], fill=(76, 175, 80)) # Crisp Lettuce
        draw.ellipse([center_x - r + 5, center_y + 15, center_x + r - 5, center_y + r // 2 + 10], fill=(196, 120, 50)) # Bottom bun

    elif food_class == "salad":
        # Base greens
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], fill=(46, 125, 50))
        # Leaf patterns
        for _ in range(25):
            ox = random.randint(-r + 15, r - 15)
            oy = random.randint(-r + 15, r - 15)
            draw.ellipse([center_x + ox - 15, center_y + oy - 15, center_x + ox + 15, center_y + oy + 15], fill=(76, 175, 80))
        # Cherry tomatoes & olives
        for _ in range(8):
            ox, oy = random.randint(-r + 20, r - 20), random.randint(-r + 20, r - 20)
            draw.ellipse([center_x + ox - 8, center_y + oy - 8, center_x + ox + 8, center_y + oy + 8], fill=(211, 47, 47))
        for _ in range(5):
            ox, oy = random.randint(-r + 20, r - 20), random.randint(-r + 20, r - 20)
            draw.ellipse([center_x + ox - 6, center_y + oy - 6, center_x + ox + 6, center_y + oy + 6], fill=(33, 33, 33))

    elif food_class == "biryani":
        # Fragrant basmati rice grains & spiced pieces
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], fill=(245, 195, 87)) # Saffron rice
        for _ in range(35):
            ox, oy = random.randint(-r + 15, r - 15), random.randint(-r + 15, r - 15)
            draw.line([center_x + ox, center_y + oy, center_x + ox + 10, center_y + oy + 3], fill=(255, 255, 230), width=2)
            draw.line([center_x + ox, center_y + oy, center_x + ox + 8, center_y + oy + 2], fill=(200, 110, 30), width=2)
        # Roasted pieces & fried onions
        for _ in range(4):
            ox, oy = random.randint(-r // 2, r // 2), random.randint(-r // 2, r // 2)
            draw.ellipse([center_x + ox - 20, center_y + oy - 20, center_x + ox + 20, center_y + oy + 20], fill=(120, 50, 15))
        # Mint coriander garnish
        for _ in range(8):
            ox, oy = random.randint(-r // 2, r // 2), random.randint(-r // 2, r // 2)
            draw.ellipse([center_x + ox - 4, center_y + oy - 4, center_x + ox + 4, center_y + oy + 4], fill=(34, 139, 34))

    elif food_class == "pasta":
        # Creamy pasta swirls
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], fill=(255, 235, 175))
        for _ in range(20):
            ox, oy = random.randint(-r + 20, r - 20), random.randint(-r + 20, r - 20)
            draw.arc([center_x + ox - 25, center_y + oy - 25, center_x + ox + 25, center_y + oy + 25], 30, 280, fill=(240, 200, 120), width=5)
        # Parsley flecks & black pepper
        for _ in range(15):
            ox, oy = random.randint(-r + 20, r - 20), random.randint(-r + 20, r - 20)
            draw.point((center_x + ox, center_y + oy), fill=(40, 40, 40))
            draw.ellipse([center_x + ox - 3, center_y + oy - 3, center_x + ox + 3, center_y + oy + 3], fill=(46, 125, 50))

    elif food_class == "grilled_chicken":
        # Tender grilled chicken breast fillet with sear marks
        draw.ellipse([center_x - r + 10, center_y - r // 2, center_x + r - 10, center_y + r // 2], fill=(210, 150, 90))
        # Char grill lines
        for offset in range(-r + 30, r - 20, 25):
            draw.line([center_x + offset - 15, center_y - 30, center_x + offset + 25, center_y + 35], fill=(90, 40, 15), width=4)
        # Rosemary & lemon wedge
        draw.line([center_x - 30, center_y, center_x + 30, center_y + 10], fill=(34, 139, 34), width=3)
        draw.pieslice([center_x + r - 45, center_y - r // 2 - 10, center_x + r + 5, center_y - r // 2 + 40], 0, 90, fill=(255, 235, 59))

    elif food_class == "sushi":
        # Slate platter with sushi rolls and nigiri
        draw.rectangle([center_x - r, center_y - r // 2, center_x + r, center_y + r // 2], fill=(55, 60, 65))
        # Salmon nigiri (orange slice on white rice)
        for i, ox in enumerate([-50, 0, 50]):
            draw.ellipse([center_x + ox - 20, center_y - 25, center_x + ox + 20, center_y - 5], fill=(250, 250, 250)) # Rice
            draw.ellipse([center_x + ox - 22, center_y - 30, center_x + ox + 22, center_y - 10], fill=(250, 110, 80)) # Salmon
            # Nori rolls
            draw.ellipse([center_x + ox - 18, center_y + 10, center_x + ox + 18, center_y + 35], fill=(20, 40, 30))
            draw.ellipse([center_x + ox - 14, center_y + 14, center_x + ox + 14, center_y + 31], fill=(250, 250, 250))
            draw.ellipse([center_x + ox - 5, center_y + 19, center_x + ox + 5, center_y + 26], fill=(220, 80, 60))

    elif food_class == "dosa":
        # Golden crisp cylindrical rolled dosa
        draw.polygon([
            (center_x - r - 10, center_y + 10),
            (center_x + r + 10, center_y - 35),
            (center_x + r + 5, center_y + 25),
            (center_x - r - 15, center_y + 70)
        ], fill=(205, 133, 63))
        # Crisp browned roasting gradient
        draw.line([center_x - r, center_y + 35, center_x + r, center_y - 5], fill=(139, 69, 19), width=12)
        # Sambar bowl & Coconut chutney bowl
        draw.ellipse([center_x - r + 10, center_y - r + 15, center_x - r + 65, center_y - r + 70], fill=(200, 90, 20), outline=(230, 230, 230), width=3)
        draw.ellipse([center_x - 15, center_y - r + 5, center_x + 35, center_y - r + 55], fill=(255, 255, 255), outline=(230, 230, 230), width=3)

    elif food_class == "fruit_bowl":
        # Assorted berries, kiwi, banana slices, strawberries
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], fill=(245, 245, 250))
        # Blueberries
        for _ in range(12):
            ox, oy = random.randint(-r + 25, r - 25), random.randint(-r + 25, r - 25)
            draw.ellipse([center_x + ox - 8, center_y + oy - 8, center_x + ox + 8, center_y + oy + 8], fill=(63, 81, 181))
        # Strawberry wedges
        for _ in range(6):
            ox, oy = random.randint(-r + 30, r - 30), random.randint(-r + 30, r - 30)
            draw.polygon([(center_x + ox, center_y + oy - 15), (center_x + ox - 12, center_y + oy + 12), (center_x + ox + 12, center_y + oy + 12)], fill=(229, 57, 53))
        # Kiwi slices
        for _ in range(5):
            ox, oy = random.randint(-r + 30, r - 30), random.randint(-r + 30, r - 30)
            draw.ellipse([center_x + ox - 14, center_y + oy - 14, center_x + ox + 14, center_y + oy + 14], fill=(139, 195, 74))

    elif food_class == "french_fries":
        # Red fry carton / pile of crispy golden sticks
        for _ in range(30):
            x1 = random.randint(center_x - r + 20, center_x + r - 30)
            y1 = random.randint(center_y - r + 30, center_y + r - 30)
            angle = random.randint(-40, 40)
            x2 = x1 + angle
            y2 = y1 + 55
            draw.line([x1, y1, x2, y2], fill=(245, 185, 30), width=8)
            draw.line([x1 + 1, y1 + 1, x2 + 1, y2 + 1], fill=(220, 150, 20), width=2)

    elif food_class == "oatmeal":
        # Warm oatmeal bowl with chia seeds and strawberry slice
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], fill=(222, 195, 160))
        for _ in range(40):
            ox, oy = random.randint(-r + 20, r - 20), random.randint(-r + 20, r - 20)
            draw.point((center_x + ox, center_y + oy), fill=(160, 130, 95))
        # Chia seeds & almond flakes
        for _ in range(25):
            ox, oy = random.randint(-r + 25, r - 25), random.randint(-r + 25, r - 25)
            draw.point((center_x + ox, center_y + oy), fill=(40, 30, 25))
        # Center strawberry & banana
        draw.ellipse([center_x - 15, center_y - 15, center_x + 15, center_y + 15], fill=(255, 235, 150))
        draw.polygon([(center_x, center_y + 5), (center_x - 12, center_y + 35), (center_x + 12, center_y + 35)], fill=(220, 40, 40))

    elif food_class == "sandwich":
        # Triangular layered club sandwich
        pts = [(center_x - r + 15, center_y - r // 2), (center_x + r - 15, center_y - r // 2), (center_x, center_y + r // 2 + 15)]
        draw.polygon(pts, fill=(230, 180, 120)) # Bread
        # Layers: lettuce, tomato, cheese, ham
        draw.line([center_x - r + 25, center_y - 10, center_x + r - 25, center_y - 10], fill=(76, 175, 80), width=6)
        draw.line([center_x - r + 30, center_y + 5, center_x + r - 30, center_y + 5], fill=(211, 47, 47), width=5)
        draw.line([center_x - r + 35, center_y + 18, center_x + r - 35, center_y + 18], fill=(255, 215, 0), width=4)

    # Slight soft smoothing for realistic camera depth
    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img


def create_sample_dataset(output_dir, samples_per_class=10):
    """
    Builds a benchmark sample dataset with meal photos and nutritional metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    nutrition_db = load_nutrition_db()

    manifest = []
    print(f"Generating synthetic benchmark dataset in {output_dir}...")

    for cls_name in FOOD_CLASSES:
        cls_info = nutrition_db.get(cls_name, {
            "name": cls_name.replace("_", " ").title(),
            "base_calories_per_100g": 200.0,
            "protein_per_100g": 10.0,
            "carbs_per_100g": 25.0,
            "fat_per_100g": 8.0,
            "fiber_per_100g": 2.0,
            "default_serving_grams": 200
        })

        cls_dir = os.path.join(output_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        for i in range(samples_per_class):
            # Portions vary between 70% and 140% of standard serving
            portion_factor = random.uniform(0.75, 1.35)
            actual_grams = cls_info["default_serving_grams"] * portion_factor
            scale = actual_grams / 100.0

            calories = round(cls_info["base_calories_per_100g"] * scale, 1)
            protein = round(cls_info["protein_per_100g"] * scale, 1)
            carbs = round(cls_info["carbs_per_100g"] * scale, 1)
            fat = round(cls_info["fat_per_100g"] * scale, 1)
            fiber = round(cls_info["fiber_per_100g"] * scale, 1)

            img = generate_synthetic_meal_sample(cls_name, portion_factor=portion_factor)
            img_filename = f"{cls_name}_{i+1:03d}.jpg"
            img_path = os.path.join(cls_dir, img_filename)
            img.save(img_path, quality=92)

            manifest.append({
                "image_path": img_path,
                "food_class": cls_name,
                "class_idx": CLASS_TO_IDX[cls_name],
                "portion_grams": round(actual_grams, 1),
                "targets": [calories, protein, carbs, fat, fiber]
            })

    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(manifest)} meal samples across {len(FOOD_CLASSES)} categories.")
    return manifest_path


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    samples_dir = os.path.join(base_dir, "sample_data")
    create_sample_dataset(samples_dir, samples_per_class=6)
