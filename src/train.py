import torch
from torch.utils.data import DataLoader
from dataset import VoiceDataset, load_asvspoof
from model import CNNModel
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
import random

project_root = Path(__file__).resolve().parents[1]

candidate_roots = [
    project_root / "data" / "LA",
    project_root / "data" / "LA" / "LA"
]

train_protocol_rel = Path("ASVspoof2019_LA_cm_protocols") / "ASVspoof2019.LA.cm.train.trn.txt"
dev_protocol_rel = Path("ASVspoof2019_LA_cm_protocols") / "ASVspoof2019.LA.cm.dev.trl.txt"

dataset_root = next(
    (p for p in candidate_roots if (p / train_protocol_rel).exists()),
    candidate_roots[0]
)

train_protocol = dataset_root / train_protocol_rel
dev_protocol = dataset_root / dev_protocol_rel
train_audio_dir = dataset_root / "ASVspoof2019_LA_train" / "flac"
dev_audio_dir = dataset_root / "ASVspoof2019_LA_dev" / "flac"

train_files_all, train_labels_all = load_asvspoof(str(train_protocol), str(train_audio_dir), limit=None)
val_files_all, val_labels_all = load_asvspoof(str(dev_protocol), str(dev_audio_dir), limit=None)


def balanced_subset(files, labels, max_per_class, seed=42):
    idx_0 = [i for i, y in enumerate(labels) if y == 0]
    idx_1 = [i for i, y in enumerate(labels) if y == 1]
    if len(idx_0) == 0 or len(idx_1) == 0:
        raise ValueError("Both classes are required, but one class is missing in this split.")

    random.seed(seed)
    k = min(max_per_class, len(idx_0), len(idx_1))
    selected = random.sample(idx_0, k) + random.sample(idx_1, k)
    random.shuffle(selected)
    return [files[i] for i in selected], [labels[i] for i in selected]


train_files, train_labels = balanced_subset(train_files_all, train_labels_all, max_per_class=500, seed=42)
val_files, val_labels = balanced_subset(val_files_all, val_labels_all, max_per_class=500, seed=123)

print(
    f"Train={len(train_labels)} (bonafide={sum(y==0 for y in train_labels)}, spoof={sum(y==1 for y in train_labels)}) | "
    f"Val={len(val_labels)} (bonafide={sum(y==0 for y in val_labels)}, spoof={sum(y==1 for y in val_labels)})"
)

train_loader = DataLoader(VoiceDataset(train_files, train_labels), batch_size=8, shuffle=True)
val_loader = DataLoader(VoiceDataset(val_files, val_labels), batch_size=8, shuffle=False)

model = CNNModel()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  #  FIX: lower LR

for epoch in range(10):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for features, label in train_loader:
        output = model(features)

        loss = criterion(output.view(-1), label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = (torch.sigmoid(output.view(-1)) > 0.5).float()
        train_correct += (preds == label).sum().item()
        train_total += label.size(0)

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for features, label in val_loader:
            output = model(features)
            loss = criterion(output.view(-1), label)
            val_loss += loss.item()
            preds = (torch.sigmoid(output.view(-1)) > 0.5).float()
            val_correct += (preds == label).sum().item()
            val_total += label.size(0)

    print(
        f"Epoch {epoch+1:02d} | "
        f"train_loss={train_loss/len(train_loader):.6e}, train_acc={train_correct/train_total:.3f} | "
        f"val_loss={val_loss/len(val_loader):.6e}, val_acc={val_correct/val_total:.3f}"
    )

torch.save(model.state_dict(), "models/model.pth")