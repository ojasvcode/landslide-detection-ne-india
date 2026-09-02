"""
spam_filter.py
Live scoring for citizen photo uploads. Combines two checks:
1. Basic image quality heuristics (blur, darkness) - catches obviously bad
   uploads cheaply, before ever running the CNN.
2. The trained ResNet18 classifier - decides whether the photo's content
   looks relevant (road/terrain/disaster-related) or irrelevant.

Usage as a module (for backend integration):
    from spam_filter import SpamFilter
    filt = SpamFilter("spam_filter_model.pt")
    result = filt.check("uploaded_photo.jpg")
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from scipy import ndimage


class SpamFilter:
    def __init__(self, model_path: str, blur_threshold: float = 100.0,
                 dark_threshold: float = 20.0, bright_threshold: float = 235.0):
        checkpoint = torch.load(model_path, map_location="cpu")
        self.classes = checkpoint["classes"]

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold

    def _check_quality(self, img: Image.Image) -> dict:
        gray = np.array(img.convert("L"), dtype=np.float64)

        # Laplacian variance is a standard, cheap blur-detection metric -
        # sharp images have high-variance edges, blurry ones don't.
        laplacian = ndimage.laplace(gray)
        blur_score = laplacian.var()

        mean_brightness = gray.mean()

        return {
            "blur_score": float(blur_score),
            "is_blurry": bool(blur_score < self.blur_threshold),
            "mean_brightness": float(mean_brightness),
            "is_too_dark": bool(mean_brightness < self.dark_threshold),
            "is_too_bright": bool(mean_brightness > self.bright_threshold),
        }

    def check(self, image_path: str) -> dict:
        img = Image.open(image_path).convert("RGB")
        quality = self._check_quality(img)

        if quality["is_blurry"] or quality["is_too_dark"] or quality["is_too_bright"]:
            return {
                "approved": False,
                "reason": "quality_check_failed",
                "quality": quality,
                "content_relevance": None,
            }

        tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        relevant_idx = self.classes.index("relevant")
        relevance_prob = float(probs[relevant_idx])
        approved = relevance_prob >= 0.5

        return {
            "approved": approved,
            "reason": "approved" if approved else "content_not_relevant",
            "quality": quality,
            "content_relevance": {
                "relevance_probability": round(relevance_prob, 4),
                "predicted_class": self.classes[int(probs.argmax())],
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    filt = SpamFilter(args.model)
    result = filt.check(args.image)
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
