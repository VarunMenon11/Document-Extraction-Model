import os
import json
import torch
from datasets import load_from_disk
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

# Load label map
with open("label2id_2.json") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)

# Load dataset
dataset = load_from_disk("layoutlm_dataset")

# Processor & model
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(list(dataset[0].keys()))


# Preprocessing for DataLoader
def collate_fn(batch):
    pixel_values = torch.stack([torch.tensor(item['pixel_values']) for item in batch])
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    bbox = torch.stack([torch.tensor(item['bbox']) for item in batch])
    labels = [item['labels'] for item in batch]

    labels_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(label + [-100] * (512 - len(label)))[:512] for label in labels],
        batch_first=True,
        padding_value=-100
    )

    return {
        "pixel_values": pixel_values.to(device),
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "bbox": bbox.to(device),
        "labels": labels_padded.to(device),
    }

# DataLoader
train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 70
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1} complete | Avg loss: {avg_loss:.4f}")

# Save model
model.save_pretrained("layoutlmv3-invoice-model-manual")
processor.save_pretrained("layoutlmv3-invoice-model-manual")

print("🎉 Training complete! Model saved to layoutlmv3-invoice-model-manual/")
