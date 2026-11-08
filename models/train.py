import argparse
import hashlib
import os
import random

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

SEED = 42
EPOCHS = 40
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
ENCODER = "resnet34"
TRAIN_NUMBERS = range(1, 71)
VAL_NUMBERS = range(71, 91)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BookDataset(Dataset):
    def __init__(self, data_dir, numbers, augment):
        self.data_dir = data_dir
        self.names = [f"book_{n:03d}" for n in numbers]
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image_path = os.path.join(self.data_dir, "images", name + ".jpg")
        mask_path = os.path.join(self.data_dir, "masks", name + ".png")
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        mask = (cv2.imread(mask_path, 0) > 127).astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                image, mask = image[::-1], mask[::-1]
            image = np.clip(image * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1), 0, 1)
        image = (image - MEAN) / STD
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        mask = torch.from_numpy(np.ascontiguousarray(mask))[None]
        return image, mask


def batch_iou(logits, masks):
    pred = (torch.sigmoid(logits) > 0.5).float()
    intersection = (pred * masks).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
    return (intersection / union.clamp(min=1)).mean().item()


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        digest.update(f.read())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/kaggle/input/book-segmentation")
    parser.add_argument("--out", default="/kaggle/working")
    args = parser.parse_args()

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    print("device:", device, "| gpu:", gpu)
    print("torch", torch.__version__, "| smp", smp.__version__, "| opencv", cv2.__version__, "| numpy", np.__version__)
    print("seed", SEED, "| epochs", EPOCHS, "| batch", BATCH_SIZE, "| lr", LEARNING_RATE, "| encoder", ENCODER)

    train_loader = DataLoader(BookDataset(args.data, TRAIN_NUMBERS, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(BookDataset(args.data, VAL_NUMBERS, False), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = smp.Unet(ENCODER, encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce = torch.nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary")

    best_iou, best_epoch = 0.0, 0
    weights_path = os.path.join(args.out, "best.pt")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = bce(logits, masks) + dice(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                val_iou += batch_iou(model(images.to(device)), masks.to(device))
        val_iou /= len(val_loader)

        if val_iou > best_iou:
            best_iou, best_epoch = val_iou, epoch
            torch.save(model.state_dict(), weights_path)
        print(f"epoch {epoch:02d} | train loss {train_loss / len(train_loader):.4f} | val IoU {val_iou:.4f}")

    print(f"best val IoU {best_iou:.4f} at epoch {best_epoch}")
    print("weights SHA-256:", sha256_of(weights_path))


if __name__ == "__main__":
    main()