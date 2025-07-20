import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import pytesseract
import json
import re
from datetime import datetime

# Load fine-tuned model
model = LayoutLMv3ForTokenClassification.from_pretrained("layoutlmv3-invoice-model-manual").to("cpu")
processor = LayoutLMv3Processor.from_pretrained("layoutlmv3-invoice-model-manual", apply_ocr=False)

# Load label map (if needed)
with open("label2id.json") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

def normalize_box(box, width, height):
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]

def predict_from_image(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Run Tesseract OCR
    ocr = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words = []
    boxes = []
    for i in range(len(ocr["text"])):
        word = ocr["text"][i].strip()
        if word == "":
            continue
        x, y, w, h = ocr["left"][i], ocr["top"][i], ocr["width"][i], ocr["height"][i]
        words.append(word)
        boxes.append(normalize_box([x, y, x + w, y + h], width, height))

    # Tokenize and encode
    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length"
    )

    with torch.no_grad():
        outputs = model(**encoding)
        predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()

    # Decode predictions
    results = {}
    for word, pred_id in zip(words, predictions):
        label = id2label.get(pred_id, "O")
        if label != "O":
            results.setdefault(label, []).append(word)

    # Join multi-word fields
    results = {k: " ".join(v) for k, v in results.items()}
    return results


def postprocess_invoice_data(raw):
    processed = {}

    # Clean Invoice Number
    invoice_number = raw.get("Invoice Number", "")
    invoice_number = re.sub(r"(Invoice[#:\s]*)", "", invoice_number, flags=re.IGNORECASE)
    processed["Invoice Number"] = invoice_number.strip()

    # Clean Addresses
    processed["Bill To"] = re.sub(r"^(Bill To\s*)", "", raw.get("Bill To", ""), flags=re.IGNORECASE).strip()
    processed["Ship To"] = re.sub(r"^(Ship To\s*)", "", raw.get("Ship To", ""), flags=re.IGNORECASE).strip()

    # Format Date
    invoice_date = raw.get("Invoice Date", "")
    try:
        dt = datetime.strptime(invoice_date.strip(), "%d %b %Y")
        processed["Invoice Date"] = dt.strftime("%d/%m/%Y")
    except:
        processed["Invoice Date"] = invoice_date

    # Extract numerical fields
    def extract_numbers(text):
        return re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))

    quantities = extract_numbers(raw.get("Quantity", ""))
    rates = extract_numbers(raw.get("Rate", ""))
    prices = extract_numbers(raw.get("Item Price", ""))

    # Extract items with regex
    item_text = raw.get("Item", "")
    item_pattern = r"(.*?)(?:SKU\s*(\d{2}-\d+))(?:.*?Exp\s*(\d{1,2}-\d{1,2}-\d{4}))?"
    matches = re.findall(item_pattern, item_text, re.IGNORECASE | re.DOTALL)

    items = []
    for i, (name, sku, expiry) in enumerate(matches):
        clean_name = re.sub(r"(Lot\s*\d+)?\s*Exp\s*\d{1,2}-\d{1,2}-\d{4}", "", name).strip()
        items.append({
            "name": clean_name,
            "sku": sku.strip() if sku else "",
            "expiry": expiry.strip() if expiry else "",
            "quantity": quantities[i] if i < len(quantities) else "",
            "rate": rates[i] if i < len(rates) else "",
            "item_price": prices[i] if i < len(prices) else ""
        })

    processed["Items"] = items
    processed["Balance Due"] = re.sub(r"[^\d.]", "", raw.get("Balance Due", ""))


    return processed




# 🔍 Predict on a test invoice
output = predict_from_image(r"Image_dataset\20250091.png")  # change as needed
output = postprocess_invoice_data(output)
print("📄 Extracted Fields:")
print(json.dumps(output, indent=2))
print("💾 Saved to output.json")