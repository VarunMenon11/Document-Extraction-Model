import os
import json
from PIL import Image
import pytesseract
from collections import defaultdict
from datasets import Dataset
from transformers import LayoutLMv3Processor
import torch


# Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load label-studio-exported JSON
with open("training2_data.json", "r") as f:
    raw_data = json.load(f)

# Group entries by image
grouped = defaultdict(list)
for entry in raw_data:
    grouped[entry["image_path"]].append(entry)

# Setup processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

# Label mapping
label_list = sorted(set(e["label"] for e in raw_data))
label2id = {label: idx + 1 for idx, label in enumerate(label_list)}
label2id["O"] = 0  # background
id2label = {v: k for k, v in label2id.items()}

# Normalize bboxes to 0-1000
def normalize_bbox(bbox, width, height):
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height),
    ]

examples = []
# Set your image directory here (use raw string or forward slashes)
IMAGE_DIR = ".\Image_dataset"  # or use forward slashes if needed

for img_path, label_entries in grouped.items():
    full_path = os.path.join(IMAGE_DIR, os.path.basename(img_path))
    assert os.path.exists(full_path), f"❌ Image not found: {full_path}"
    image = Image.open(full_path).convert("RGB")

    width, height = image.size

    # Get ground-truth boxes
    gt_boxes = [(e["bbox"], e["label"]) for e in label_entries]

    # OCR tokens
    ocr = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words, boxes, labels = [], [], []

    for i in range(len(ocr["text"])):
        word = ocr["text"][i].strip()
        if not word:
            continue
        x, y, w, h = ocr["left"][i], ocr["top"][i], ocr["width"][i], ocr["height"][i]
        box = [x, y, x + w, y + h]

        # Assign label based on intersection
        label = "O"
        for gt_box, gt_label in gt_boxes:
            gx0, gy0, gx1, gy1 = gt_box
            if not (box[2] < gx0 or box[0] > gx1 or box[3] < gy0 or box[1] > gy1):
                label = gt_label
                break

        words.append(word)
        boxes.append(normalize_bbox(box, width, height))
        labels.append(label2id[label])

    # Tokenize with LayoutLMv3 processor
    encoded = processor(
        images=image,
        text=words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    encoded["labels"] = labels + [0] * (512 - len(labels))  # pad labels to 512
    examples.append({
        key: val.squeeze().tolist() if isinstance(val, torch.Tensor) else val
        for key, val in encoded.items()
    })

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(examples)
dataset.save_to_disk("layoutlm_dataset")

# Save label map
with open("label2id_2.json", "w") as f:
    json.dump(label2id, f, indent=2)

print("✅ Preprocessing complete! Dataset saved to 'layoutlm_dataset'")
print("Label map saved to 'label2id_2.json'")