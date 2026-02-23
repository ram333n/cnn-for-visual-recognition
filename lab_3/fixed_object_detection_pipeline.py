import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision.transforms import functional as F

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image


# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    # batch: List[(image, target)]
    images, targets = zip(*batch)
    return list(images), list(targets)


# -----------------------------
# Dataset
# -----------------------------
@dataclass
class Sample:
    image_path: Path
    ann_path: Path
    order_name: str


class ArTaxOrVoTTDataset(Dataset):
    """
    Reads VoTT JSON per image. Expected keys:
      - data["asset"]["name"] -> image filename
      - data["regions"][i]["boundingBox"] -> {left, top, width, height}
      - data["regions"][i]["tags"] -> ["OrderName", ...] (optional; we fallback to folder name)
    """

    def __init__(
        self,
        samples: List[Sample],
        class_to_idx: Dict[str, int],
        train: bool = True,
        vflip_p: float = 0.5,
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.train = train
        self.vflip_p = vflip_p

    def __len__(self) -> int:
        return len(self.samples)

    def _read_target(self, ann_path: Path, fallback_label: str) -> Tuple[Tensor, Tensor]:
        data = json.loads(ann_path.read_text(encoding="utf-8"))

        boxes: List[List[float]] = []
        labels: List[int] = []

        for reg in data.get("regions", []):
            bb = reg.get("boundingBox", None)
            if bb is None:
                continue

            left = float(bb["left"])
            top = float(bb["top"])
            width = float(bb["width"])
            height = float(bb["height"])

            x_min, y_min = left, top
            x_max, y_max = left + width, top + height

            tag_list = reg.get("tags", []) or []
            label_name = tag_list[0] if len(tag_list) > 0 else fallback_label
            if label_name not in self.class_to_idx:
                label_name = fallback_label

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.class_to_idx[label_name])

        if len(boxes) == 0:
            raise ValueError(f"No boxes found in {ann_path}")

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        s = self.samples[idx]

        data = json.loads(s.ann_path.read_text(encoding="utf-8"))
        img_name = data.get("asset", {}).get("name", s.image_path.name)
        img_path = s.image_path.parent / img_name

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes, labels = self._read_target(s.ann_path, fallback_label=s.order_name)

        # Basic augmentation: vertical flip
        if self.train and random.random() < self.vflip_p:
            img = F.vflip(img)
            boxes[:, [1, 3]] = torch.tensor([h, h], dtype=torch.float32) - boxes[:, [3, 1]]

        img_t = F.to_tensor(img)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)

        target: Dict[str, Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }
        return img_t, target


class ArTaxOrDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 2,
        num_workers: int = 4,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None

    def _discover_orders(self, artaxor_root: Path) -> List[str]:
        # class folders are the "orders"
        orders = []
        for d in artaxor_root.iterdir():
            if d.is_dir() and (d / "annotations").exists():
                orders.append(d.name)
        orders = sorted(orders)
        if not orders:
            raise FileNotFoundError(f"No order folders with 'annotations' found under {artaxor_root}")
        return orders

    def _build_samples(self, artaxor_root: Path, orders: List[str]) -> List[Sample]:
        samples: List[Sample] = []
        for order in orders:
            order_dir = artaxor_root / order
            ann_dir = order_dir / "annotations"
            if not ann_dir.exists():
                continue
            for ann_path in ann_dir.rglob("*.json"):
                samples.append(Sample(image_path=order_dir / "dummy.jpg", ann_path=ann_path, order_name=order))
        if not samples:
            raise FileNotFoundError(f"No JSON annotations found under {artaxor_root}")
        return samples

    def setup(self, stage: str | None = None) -> None:
        artaxor_root = self.data_dir / "ArTaxOr"
        if not artaxor_root.exists():
            artaxor_root = self.data_dir

        orders = self._discover_orders(artaxor_root)
        self.class_to_idx = {name: i + 1 for i, name in enumerate(orders)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        all_samples = self._build_samples(artaxor_root, orders)

        train_samples, val_samples = self._split_with_stratification(all_samples)

        self.train_ds = ArTaxOrVoTTDataset(train_samples, self.class_to_idx, train=True)
        self.val_ds = ArTaxOrVoTTDataset(val_samples, self.class_to_idx, train=False)

    def _split_with_stratification(self, samples: List[Sample]) -> Tuple[List[Sample], List[Sample]]:
        grouped_samples = {}
        for sample in samples:
            if sample.order_name not in grouped_samples:
                grouped_samples[sample.order_name] = []

            grouped_samples[sample.order_name].append(sample)

        train_samples: List[Sample] = []
        val_samples: List[Sample] = []

        rng = random.Random(self.seed)

        for order_name, group in grouped_samples.items():
            rng.shuffle(group)
            n_val = int(len(group) * self.val_split)
            val_samples.extend(group[:n_val])
            train_samples.extend(group[n_val:])

        return train_samples, val_samples

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )


# -----------------------------
# Lightning module
# -----------------------------
class FasterRCNNLit(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        lr: float = 1e-6,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        self.model = model

        self.map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    def forward(self, images: List[Tensor], targets: List[Dict[str, Tensor]] | None = None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)  # returns losses in train mode
        loss = sum(loss_dict.values())
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        self.map_metric.update(preds, targets)

    def on_validation_epoch_end(self):
        metrics = self.map_metric.compute()
        self.log("val/map", metrics["map"], prog_bar=True)
        self.log("val/map_50", metrics["map_50"], prog_bar=True)
        self.map_metric.reset()

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
        return {"optimizer": opt, "lr_scheduler": sch}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./ArTaxOr", help="Path containing ArTaxOr/...")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    dm = ArTaxOrDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )
    dm.setup()

    num_classes = len(dm.class_to_idx) + 1  # + background (0)
    lit = FasterRCNNLit(num_classes=num_classes, lr=args.lr)

    ckpt = ModelCheckpoint(monitor="val/map_50", mode="max", save_top_k=1, filename="frcnn-{epoch:02d}-{val_map50:.4f}")
    lrmon = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[ckpt, lrmon],
        log_every_n_steps=20
    )
    trainer.fit(lit, datamodule=dm)
    print("Best checkpoint:", ckpt.best_model_path)


if __name__ == "__main__":
    main()