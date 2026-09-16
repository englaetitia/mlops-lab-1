"""Fine-tune a pretrained ResNet-18 on Food-11 and track the run in MLflow.

What this does
--------------
1. Loads `data/food11_processed[_mini]/{training,validation,evaluation}` with
   `torchvision.datasets.ImageFolder` — which is why Lab 1 rearranged the images
   into `<split>/<Category>/` folders: ImageFolder reads the label from the
   directory name.
2. Takes `resnet18` with ImageNet weights and replaces its final fully-connected
   layer (1000 ImageNet classes -> 11 food classes). Everything before that layer
   already knows edges, textures and shapes, so only the head has to be learned
   from scratch.
3. Logs the hyperparameters once as *params*, then `train_loss`, `val_loss` and
   `val_accuracy` once per epoch as *metrics* with a `step`, then the final test
   accuracy and the trained model itself.

Usage
-----
    uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
    uv run python ./src/food11/train.py --dataset processed --epochs 10 --lr 0.0001
    uv run python ./src/food11/train.py --dataset mini --epochs 3 --freeze-backbone

The MLflow tracking server must be running first:

    uv run mlflow server --host 127.0.0.1 --port 5000 \
        --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "food11"

# The statistics the ImageNet weights were trained with. Normalising with the
# same numbers is what makes the pretrained features transfer.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATASET_DIRS = {
    "processed": "food11_processed",
    "mini": "food11_processed_mini",
}


def build_dataloaders(root: Path, batch_size: int, num_workers: int):
    """One DataLoader per split. Images are already 128x128 on disk."""
    train_tf = transforms.Compose([
        # Cheap augmentation: a flipped plate of food is still the same food.
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        # No augmentation when measuring — evaluation must be deterministic.
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(root / "training", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "validation", transform=eval_tf)
    test_ds = datasets.ImageFolder(root / "evaluation", transform=eval_tf)

    loaders = (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
    )
    return loaders, train_ds


def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        # Feature extraction instead of fine-tuning: gradients flow only through
        # the new head. Much faster on CPU, usually a little less accurate.
        for parameter in model.parameters():
            parameter.requires_grad = False

    # resnet18's classifier is `fc`; 512 features in, 1000 ImageNet classes out.
    # A fresh Linear layer has requires_grad=True, so it trains either way.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Weight by batch size so a short final batch doesn't skew the mean.
        running_loss += loss.item() * images.size(0)
        seen += images.size(0)
    return running_loss / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Return (mean loss, accuracy) over a whole split."""
    model.eval()
    running_loss = 0.0
    correct = 0
    seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        running_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)
    return running_loss / max(seen, 1), correct / max(seen, 1)


def log_model_compat(model, input_example) -> None:
    """Log the trained model, coping with the mlflow 2.x / 3.x API difference.

    mlflow 3 renamed `artifact_path` to `name`, and its default `pt2`
    serialization format *requires* an input example: it traces the graph by
    running model.forward on that example. The example also gives the logged
    model a signature, which is what lets the next lab serve it.
    """
    kwargs = dict(input_example=input_example, serialization_format="pickle")
    try:
        mlflow.pytorch.log_model(model, name="model", **kwargs)
    except TypeError:
        mlflow.pytorch.log_model(model, artifact_path="model", **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_DIRS), default="mini",
                        help="which prepared dataset to train on (default: mini)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="train only the new final layer")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers; keep 0 on Windows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = DATA_DIR / DATASET_DIRS[args.dataset]
    if not root.is_dir():
        print(f"ERROR: dataset not found at {root}")
        print("Run src/food11/data.py first, or `dvc pull`.")
        return 1

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (train_loader, val_loader, test_loader), train_ds = build_dataloaders(
        root, args.batch_size, args.num_workers)
    num_classes = len(train_ds.classes)

    model = build_model(num_classes, args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    # Only optimise what actually requires gradients — matters when frozen.
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"device       : {device}")
    print(f"dataset      : {root.name}  ({len(train_ds)} training images, "
          f"{num_classes} classes)")
    print(f"lr={args.lr}  batch_size={args.batch_size}  epochs={args.epochs}\n")

    with mlflow.start_run():
        # PARAMS: set before training, fixed for the whole run.
        mlflow.log_params({
            "dataset": args.dataset,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "model": "resnet18",
            "pretrained": True,
            "freeze_backbone": args.freeze_backbone,
            "num_classes": num_classes,
            "train_images": len(train_ds),
            "seed": args.seed,
            "device": device.type,
        })

        started = time.time()
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate(
                model, val_loader, criterion, device)

            # METRICS: produced during training, so they carry a step.
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(f"epoch {epoch + 1}/{args.epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_accuracy={val_accuracy:.4f}")

        test_loss, test_accuracy = evaluate(
            model, test_loader, criterion, device)
        duration = time.time() - started

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("training_seconds", duration)

        # One real image, so mlflow can trace the graph and record the
        # input/output signature alongside the weights. mlflow runs a forward
        # pass on that example to infer the signature, and the example is a
        # CPU array — so the model has to be on the CPU for it, otherwise
        # inference fails with "Input type (torch.FloatTensor) and weight type
        # (torch.cuda.FloatTensor) should be the same". Training is finished at
        # this point, so moving the model is free.
        example_batch = next(iter(val_loader))[0][:1].cpu().numpy()
        model.cpu()
        log_model_compat(model, example_batch)

        run_id = mlflow.active_run().info.run_id
        print(f"\ntest_accuracy={test_accuracy:.4f}  "
              f"test_loss={test_loss:.4f}  ({duration:.0f}s)")
        print(f"run_id={run_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
