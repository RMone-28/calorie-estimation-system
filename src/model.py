"""
Hybrid CNN Model for Food Image Classification and Calorie Regression
Project: An Intelligent System to Estimate calorie content from meal photographs
Author: PALAGANI MONESHA (23ME1A54C7)
Supervisor: Dr. P. Sudhakar
Domain: Deep Learning & Computer Vision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as models
    from torchvision.models import MobileNet_V2_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class SpatialAttention(nn.Module):
    """
    Spatial Attention Block:
    Focuses the network on the food items on the plate while suppressing
    irrelevant background noise (tablecloths, cutlery, shadows, plates).
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_mask = self.conv(x)
        return x * attention_mask


class FoodCalorieHybridCNN(nn.Module):
    """
    Dual-Head CNN Hybrid Architecture:
    1. Shared Backbone: High-level visual representation learning
    2. Spatial Attention: Focuses on food portions and plate geometry
    3. Classification Head: Predicts food categories (P(class | image))
    4. Calorie & Macro Regression Head: Predicts continuous caloric & macronutrient values.
       Conditioned on both the extracted visual features AND the food classification probabilities!
    """
    def __init__(self, num_classes=12, pretrained=True, dropout_rate=0.3):
        super(FoodCalorieHybridCNN, self).__init__()
        self.num_classes = num_classes

        # Feature Extractor Backbone: MobileNetV2 (efficient, mobile-ready)
        if HAS_TORCHVISION:
            try:
                weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
                base_model = models.mobilenet_v2(weights=weights)
                self.backbone = base_model.features
                feature_dim = 1280
            except Exception:
                self.backbone = self._build_custom_backbone()
                feature_dim = 256
        else:
            self.backbone = self._build_custom_backbone()
            feature_dim = 256

        # Attention layer over spatial feature map
        self.attention = SpatialAttention(feature_dim)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Shared dense representation
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Head 1: Food Item Classification
        self.classifier_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

        # Head 2: Calorie & Macro Regression
        # Conditioned on visual features (512) + Classification class probabilities (num_classes)
        reg_input_dim = 512 + num_classes
        self.regression_head = nn.Sequential(
            nn.Linear(reg_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 5)  # [Calories (kcal), Protein (g), Carbs (g), Fat (g), Fiber (g)]
        )

    def _build_custom_backbone(self):
        """Fallback multi-scale convolutional backbone if torchvision weights are unavailable"""
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass:
        Args:
            x: Input batch of food photos [B, 3, 224, 224]
        Returns:
            class_logits: [B, num_classes]
            macro_predictions: [B, 5] -> [calories, protein, carbs, fat, fiber]
        """
        # Feature extraction
        features = self.backbone(x)
        attended_features = self.attention(features)
        pooled = self.global_pool(attended_features)
        flat_features = torch.flatten(pooled, 1)

        # Shared representation
        shared_embedding = self.shared_fc(flat_features)

        # Classification Head
        class_logits = self.classifier_head(shared_embedding)
        class_probs = F.softmax(class_logits, dim=-1)

        # Hybrid conditioning: concatenate visual features with class probabilities
        # This provides class-context to the continuous calorie regressor!
        hybrid_features = torch.cat([shared_embedding, class_probs], dim=-1)

        # Regression Head: outputs [Calories, Protein, Carbs, Fat, Fiber]
        # Relu ensures all predicted nutritional values are strictly non-negative
        macro_raw = self.regression_head(hybrid_features)
        macro_predictions = F.relu(macro_raw)

        return class_logits, macro_predictions


class MultiTaskCalorieLoss(nn.Module):
    """
    Joint Multi-Task Loss Function:
    Balances Classification (CrossEntropy) and Calorie/Macro Regression (SmoothL1 / Huber)
    """
    def __init__(self, cls_weight=1.0, cal_weight=2.0, macro_weight=1.0):
        super(MultiTaskCalorieLoss, self).__init__()
        self.cls_weight = cls_weight
        self.cal_weight = cal_weight
        self.macro_weight = macro_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=1.0)

    def forward(self, class_logits, macro_preds, class_targets, macro_targets):
        """
        Args:
            class_logits: [B, C]
            macro_preds: [B, 5] (Calories, Protein, Carbs, Fat, Fiber)
            class_targets: [B] (Integer class IDs)
            macro_targets: [B, 5] (Ground truth kcal & grams)
        """
        # 1. Classification Loss
        l_cls = self.ce_loss(class_logits, class_targets)

        # 2. Calorie Regression Loss (First column)
        l_cal = self.smooth_l1(macro_preds[:, 0], macro_targets[:, 0])

        # 3. Macronutrient Regression Loss (Remaining 4 columns)
        l_macro = self.smooth_l1(macro_preds[:, 1:], macro_targets[:, 1:])

        # Total Weighted Multi-Task Objective
        total_loss = (self.cls_weight * l_cls) + (self.cal_weight * l_cal) + (self.macro_weight * l_macro)

        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_cls": l_cls.item(),
            "loss_cal": l_cal.item(),
            "loss_macro": l_macro.item()
        }


if __name__ == "__main__":
    # Smoke test for tensor shape flow
    model = FoodCalorieHybridCNN(num_classes=12, pretrained=False)
    dummy_input = torch.randn(4, 3, 224, 224)
    logits, macros = model(dummy_input)
    print(f"Logits shape: {logits.shape} (Expected: [4, 12])")
    print(f"Macros shape: {macros.shape} (Expected: [4, 5])")
    print("Model initialized and verified successfully!")
