import json
from pathlib import Path

# Load your exported Label Studio JSON
with open("project-6-at-2025-05-24-15-44-a8813957.json", "r") as f:
    label_studio_data = json.load(f)

training_data = []

for task in label_studio_data:
    image_path = Path(task['data']['image']).name
    annotations = task['annotations'][0]['result']

    for ann in annotations:
        if "value" not in ann:
            continue
        val = ann["value"]

        # Get the label name (e.g., "Invoice Number", "Bill To", etc.)
        label = val.get("rectanglelabels", [""])[0].strip()

        # Get the associated transcription text (user input)
        text = val.get("text", [""])[0].strip()

        # Get original image size
        width, height = ann["original_width"], ann["original_height"]

        # Convert percentage coords to pixel values
        x = int((val["x"] / 100) * width)
        y = int((val["y"] / 100) * height)
        w = int((val["width"] / 100) * width)
        h = int((val["height"] / 100) * height)

        bbox = [x, y, x + w, y + h]

        training_data.append({
            "image_path": image_path,
            "text": text,
            "label": label,
            "bbox": bbox
        })

# Save the cleaned training data
with open("training2_data.json", "w") as out:
    json.dump(training_data, out, indent=2)
