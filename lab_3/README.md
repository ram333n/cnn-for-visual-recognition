# Lab 3: Fixing object detection pipeline

Found bugs:

1. Access to not existent folder `annotations` in root:
```python
    # ...
    def setup(self, stage: str | None = None) -> None:
        artaxor_root = self.data_dir / "ArTaxOr"
        if not artaxor_root.exists():
            # Sometimes users point directly at ArTaxOr
            if (self.data_dir / "annotation").exists(): # it doesn't exist
                artaxor_root = self.data_dir
            else:
                raise FileNotFoundError(f"Expected ArTaxOr folder under {self.data_dir}")
   #  ...
```

2. Train/val dataset split is incorrect: stratification is absent

```python
        rng = random.Random(self.seed)
        rng.shuffle(all_samples)
        n_val = max(1, int(len(all_samples) * self.val_split))
        val_samples = all_samples[:n_val]
        train_samples = all_samples[n_val:]
```

3. Incorrect bounding box vertices evaluation

```python
            left = float(bb["left"])
            top = float(bb["top"])
            width = float(bb["width"])
            height = float(bb["height"])
            x1, y1 = left, top
            x2, y2 = left + height, top + width
```

4. Incorrect returning target bounding box in `ArTaxOrVoTTDataset`

```python
boxes.append([x1, y2, x2, y1])
```

5. Incorrect `img.size` unpacking order in `ArTaxOrVoTTDataset.__getitem__`:

```python
h, w = img.size
```

6. `hflip_p` naming - should be `vflip_p`

7. `_` in `collate_fn` is unnecessary

```python
images, targets, _ = zip(*batch)
```

8. Small `gamma` hyperparameter in `StepLR`
9. Increase `lr` to `1e-4`
10. Incorrect `map_50` usage in `ModelCheckpoint` filename