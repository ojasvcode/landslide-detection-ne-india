"""
prepare_dataset.py
Builds a binary relevance dataset from the already-extracted RDD2020 folder
structure (train/<Country>/images/*.jpg for Czech, India, Japan - real
vehicle-mounted smartphone road photos) as the "relevant" class, paired
with CIFAR-10 (auto-downloaded, everyday unrelated objects) as the
"irrelevant" class.

Usage:
    python prepare_dataset.py \
        --rdd_train_dir train \
        --outdir dataset \
        --max_per_class 3000
"""
import argparse
import os
import random
import glob
from PIL import Image
import torchvision


def extract_relevant_images(rdd_train_dir, outdir, max_images):
    relevant_dir = os.path.join(outdir, "relevant")
    os.makedirs(relevant_dir, exist_ok=True)

    countries = ["India", "Japan", "Czech"]
    all_paths = []
    for country in countries:
        pattern = os.path.join(rdd_train_dir, country, "images", "*.jpg")
        all_paths.extend(glob.glob(pattern))

    print(f"Found {len(all_paths)} real road images across {countries}")
    random.Random(42).shuffle(all_paths)
    selected = all_paths[:max_images]

    print(f"Copying {len(selected)} road images as the relevant class...")
    for i, src_path in enumerate(selected):
        img = Image.open(src_path).convert("RGB")
        img.save(os.path.join(relevant_dir, f"relevant_{i:04d}.jpg"))

    print(f"Saved {len(selected)} relevant images to {relevant_dir}")
    return len(selected)


def extract_irrelevant_images(outdir, max_images):
    irrelevant_dir = os.path.join(outdir, "irrelevant")
    os.makedirs(irrelevant_dir, exist_ok=True)

    # STL10 instead of CIFAR-10: CIFAR's native 32x32 images become
    # visibly blurry when upscaled to 224x224, giving the model a trivial
    # "blurry = irrelevant" shortcut instead of learning real content
    # differences - confirmed by an instant, suspicious 100% validation
    # accuracy in testing. STL10's native 96x96 resolution is much closer
    # to real photo detail, forcing the model to learn actual content.
    stl = torchvision.datasets.STL10(root=os.path.join(outdir, "_stl_raw"), split="train", download=True)

    print(f"Saving {max_images} STL10 images as the irrelevant class...")
    rng = random.Random(42)
    indices = rng.sample(range(len(stl)), min(max_images, len(stl)))
    for i, idx in enumerate(indices):
        img, _ = stl[idx]
        img = img.resize((224, 224), Image.BILINEAR)
        img.save(os.path.join(irrelevant_dir, f"irrelevant_{i:04d}.jpg"))

    print(f"Saved {len(indices)} irrelevant images to {irrelevant_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdd_train_dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max_per_class", type=int, default=3000)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    n_relevant = extract_relevant_images(args.rdd_train_dir, args.outdir, args.max_per_class)
    extract_irrelevant_images(args.outdir, n_relevant)  # match class balance

    print(f"\nDataset ready at {args.outdir}")
    print(f"  relevant/: {n_relevant} images")
    print(f"  irrelevant/: {n_relevant} images")


if __name__ == "__main__":
    main()
