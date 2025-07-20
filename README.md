
# 📄 Generalized Document Extraction using LayoutLMv3

A deep learning pipeline for structured information extraction from scanned business documents (e.g., invoices, receipts, utility bills), using **LayoutLMv3** with OCR and semantic post-processing.

---

## 🧭 Overview

This project started as an invoice extractor and evolved into a generalized, document-aware NER system using visual transformers. It combines OCR, bounding boxes, and token-level classification to extract structured data from scanned files.

It supports:
- ✅ Invoices
- ✅ Utility Bills
- ✅ Receipts
- ✅ Statements

---

## 🚦 Step 1: Annotate Documents with Label Studio

Before training the model, you need annotated data. This project uses [**Label Studio**](https://labelstud.io/) — a powerful open-source data labeling tool that supports OCR and document annotation.

### 🔧 Install Label Studio
```bash
pip install label-studio
```

### 🚀 Launch Label Studio Locally
```bash
label-studio start
```

This will open Label Studio at [http://localhost:8080](http://localhost:8080). You can now:
- Create a new project (e.g., "Invoice Extraction")
- Upload your scanned invoice/utility bill/receipt images
- Define a labeling config for OCR-style token classification

### 🧩 Sample Labeling Config (NER with bounding boxes)
```xml
<View>
  <Image name="image" value="$ocr" zoom="true"/>
  <Labels name="label" toName="image">
    <Label value="Invoice Number"/>
    <Label value="Bill To"/>
    <Label value="Item"/>
    <Label value="Quantity"/>
    <Label value="Rate"/>
    <Label value="Balance Due"/>
  </Labels>
</View>
```

> 💡 You can use `ocr` fields or overlay your own OCR data (via Tesseract or Azure OCR) for better control.

### ✅ Export the Labeled Data
After completing annotations:
1. Go to **Export**
2. Choose `JSON` (Label Studio format)
3. Save it as: `exported_annotations.json`

---

## 🧱 Project Pipeline

```
Label Studio JSON → TrainingData.py (flat labels)
         ↓
PrepareDataset.py → OCR (Tesseract) → Input IDs + BBoxes
         ↓
layoutlm_dataset → LayoutLMv3 Fine-Tuning
         ↓
Testing_Model.py → Inference + Post-Processing
```

---

## 🛠️ Setup & Installation

### Requirements
- Python ≥ 3.8
- PyTorch
- HuggingFace Transformers & Datasets
- Pillow
- pytesseract (Tesseract OCR)
- Label Studio (for annotation)

### Installation
```bash
pip install -r requirements.txt
```

---

## 🧮 Step 2: Convert Annotations

Run this to flatten Label Studio JSON and create token-level labels:

```bash
python TrainingData.py
```

This generates a CSV of labeled tokens mapped to bounding boxes and tags.

---

## 🧪 Step 3: Prepare Dataset for LayoutLMv3

```bash
python PrepareDataset.py
```

- Uses Tesseract OCR to extract text and bounding boxes
- Aligns tokens with labels
- Prepares HuggingFace-compatible dataset under `layoutlm_dataset/`

---

## 🏋️ Step 4: Train LayoutLMv3

```bash
python Model_training.py
```

📌 Configuration:
- Epochs: **70**
- Model: `microsoft/layoutlmv3-base`
- Output: `layoutlmv3-invoice-model-manual/`

---

## 🔍 Step 5: Inference & Postprocessing

```bash
python Testing_Model.py
```

This script:
- Uses OCR to get words and bounding boxes
- Runs LayoutLMv3 model
- Post-processes fields (e.g., cleaning invoice numbers, extracting quantities and prices)

---

## 📁 Folder Structure

```
.
├── Image_dataset/                  # Raw document images
├── layoutlm_dataset/              # Preprocessed HuggingFace dataset
├── label2id_2.json                # Label mapping
├── TrainingData.py                # Label Studio to flat data converter
├── PrepareDataset.py              # OCR + token-label alignment
├── Model_training.py              # Fine-tuning LayoutLMv3
├── Testing_Model.py               # Inference + post-processing
├── exported_annotations.json      # Exported from Label Studio
├── README.md                      # This file
```

---

## 🎯 Improvements Made

- 🧾 Generalized to multiple document types
- 🧠 Smart post-processing pipeline
- ⚙️ HuggingFace integration
- 🔄 Modular design

---

## 📌 Optional Next Steps

Let me know if you'd like:
- A Gradio/Streamlit UI
- Evaluation script for precision/recall
- `requirements.txt` generation
- Uploading to HuggingFace Hub
