# Model C - Citizen Photo Spam Filter

Filters irrelevant citizen photo uploads (selfies, memes, random objects)
so field officials only review plausibly disaster-relevant photos.

## Data sources
- Relevant class: RDD2020 - real vehicle-mounted smartphone road photos
  from India, Japan, and the Czech Republic (Mendeley Data, DOI
  10.17632/5ty2wb6gvg.1, train.tar.gz)
- Irrelevant class: CIFAR-10 everyday objects, auto-downloaded via
  torchvision - a proxy for "clearly non-disaster-related content" since
  no purpose-built "irrelevant citizen upload" dataset exists publicly

## Setup

```bash
pip install -r requirements.txt
```

## Get the data (manual, one-time)

Download `train.tar.gz` from https://data.mendeley.com/datasets/5ty2wb6gvg/1
(~1.37 GB), then:

```bash
tar -xzvf train.tar.gz
```

This produces `train/India/images/`, `train/Japan/images/`,
`train/Czech/images/` - real road photos, ready to use directly.

## Run in order

```bash
python prepare_dataset.py \
    --rdd_train_dir train \
    --outdir dataset \
    --max_per_class 3000

python train_spam_filter.py \
    --dataset_dir dataset \
    --out spam_filter_model.pt \
    --epochs 5
```

Training uses a frozen pretrained backbone (linear probe) - fast enough
for CPU, should take a few minutes for 5 epochs, not hours. If CPU
training is still too slow, reduce --max_per_class in the dataset prep
step or --epochs.

## Test it

```bash
python spam_filter.py --model spam_filter_model.pt --image /path/to/some/photo.jpg
```

## Outputs (handoff to Member 3)

- `spam_filter_model.pt` - trained classifier
- `spam_filter.py` - importable `SpamFilter` class with a `.check(image_path)`
  method, same integration pattern as Model A's `LandslidePredictor` and
  Model B's `RainfallTrigger` - wire into an upload endpoint in api.py

## Known limitations (disclose honestly in the pitch)
- Trained to detect general "relevance" (plausible road/terrain/disaster
  scene vs clearly unrelated content), not specifically "is this a
  landslide" - a more specialized classifier would need a real labeled
  landslide-photo dataset, which doesn't exist publicly at scale
- Irrelevant/negative class uses CIFAR-10 as a proxy, not real examples of
  actual spam uploads (selfies, memes) - reasonable given time constraints,
  but a real deployment would benefit from real citizen-upload examples
- Linear-probe training (frozen backbone) trades some accuracy for speed;
  full fine-tuning would likely perform better given more time/compute

## Real-world validation (informal test, 3 personal photos)
Ran 3 real selfies (not from any training source) through the trained model:
- 2/3 correctly rejected as irrelevant (1 via blur-quality check, 1 via
  genuine CNN content judgment - relevance probability 0.397)
- 1/3 incorrectly approved (relevance probability 0.686) - a warm-toned
  indoor scene the model likely confused with road-surface color/texture
  statistics

This ~2/3 accuracy on real out-of-distribution photos is a more honest
signal than the 99-100% validation accuracy from training (which only
measures performance on STL10/RDD2020, not real citizen uploads). Disclose
this openly - production deployment would benefit from real labeled
citizen-upload examples for the negative class, which weren't available
within this hackathon's time constraints.
