import argparse
import os
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models

import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MonteCarloDropout(nn.Module):
    """
    Dropout that is always active (training=True) so we can
    use it at inference time while the model is in eval() mode.
    """
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)


def replace_dropout_with_mc_dropout(model: nn.Module) -> List[MonteCarloDropout]:
    """
    If you had Dropout in the backbone, this would replace them.
    Right now we are inserting dropout only in the head, so this is unused,
    but kept here if you later want to extend.
    """
    mc_dropout_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            parent = model
            *path, last = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            mc = MonteCarloDropout(p=module.p)
            setattr(parent, last, mc)
            mc_dropout_layers.append(mc)

    return mc_dropout_layers


def add_head_dropout_for_resnet(base_model: nn.Module,
                                num_classes: int,
                                init_p: float = 0.0) -> Tuple[nn.Module, List[MonteCarloDropout]]:
    """
    Take a torchvision ResNet and attach a MonteCarloDropout before the final FC.
    """
    in_features = base_model.fc.in_features
    base_model.fc = nn.Identity()

    class ResNetWithDropout(nn.Module):
        def __init__(self, backbone, in_features, num_classes, p):
            super().__init__()
            self.backbone = backbone
            self.dropout = MonteCarloDropout(p=p)
            self.fc = nn.Linear(in_features, num_classes)

        def forward(self, x):
            feats = self.backbone(x)  # global pooled features
            feats = self.dropout(feats)
            logits = self.fc(feats)
            return logits

    model = ResNetWithDropout(base_model, in_features, num_classes, init_p)
    return model, [model.dropout]


def add_head_dropout_for_mobilenet(mnet: nn.Module,
                                   num_classes: int,
                                   init_p: float = 0.0) -> Tuple[nn.Module, List[MonteCarloDropout]]:
    """
    Rebuild MobileNetV3 classifier with a MonteCarloDropout.
    """
    last_channel = mnet.classifier[0].in_features

    class MobileNetWithDropout(nn.Module):
        def __init__(self, backbone, last_channel, num_classes, p):
            super().__init__()
            self.features = backbone.features
            self.avgpool = backbone.avgpool
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(last_channel, 1024)
            self.act = nn.Hardswish()
            self.dropout = MonteCarloDropout(p=p)
            self.fc2 = nn.Linear(1024, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = MobileNetWithDropout(mnet, last_channel, num_classes, init_p)
    return model, [model.dropout]


def build_model(model_name: str,
                num_classes: int,
                init_dropout_p: float = 0.0,
                pretrained: bool = False) -> Tuple[nn.Module, List[MonteCarloDropout]]:
    """
    Build the specified model and attach MonteCarloDropout layers.
    Training will use p=0.0 (no perturbation).
    Inference will later change p.
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model, mc_layers = add_head_dropout_for_resnet(base, num_classes, init_dropout_p)
    elif model_name == "resnet34":
        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        model, mc_layers = add_head_dropout_for_resnet(base, num_classes, init_dropout_p)
    elif model_name in ["mobilenetv3_small", "mobilenetv3-small"]:
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        model, mc_layers = add_head_dropout_for_mobilenet(base, num_classes, init_dropout_p)
    elif model_name in ["mobilenetv3_large", "mobilenetv3-large"]:
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        model, mc_layers = add_head_dropout_for_mobilenet(base, num_classes, init_dropout_p)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, mc_layers


def set_mc_dropout_p(mc_layers: List[MonteCarloDropout], p: float):
    for m in mc_layers:
        m.p = p


# ----------------------------
# Datasets
# ----------------------------

def get_cifar10_loaders(data_root: str,
                        batch_size: int,
                        num_workers: int = 4) -> Tuple[DataLoader, DataLoader, int, str]:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, 10, "CIFAR10"


def get_cifar100_loaders(data_root: str,
                         batch_size: int,
                         num_workers: int = 4) -> Tuple[DataLoader, DataLoader, int, str]:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
        ),
    ])

    train_ds = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, 100, "CIFAR100"


def get_svhn_loaders(data_root: str,
                     batch_size: int,
                     num_workers: int = 4) -> Tuple[DataLoader, DataLoader, int, str]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4377, 0.4438, 0.4728),
            std=(0.1980, 0.2010, 0.1970),
        ),
    ])

    train_ds = datasets.SVHN(root=data_root, split="train", download=True, transform=transform)
    test_ds = datasets.SVHN(root=data_root, split="test", download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, 10, "SVHN"


def get_tinyimagenet_loaders(data_root: str,
                             batch_size: int,
                             num_workers: int = 4) -> Tuple[DataLoader, DataLoader, int, str]:
    """
    Expects Tiny ImageNet in ImageFolder format:

    data_root/tiny-imagenet-200/train/<class>/*.JPEG
    data_root/tiny-imagenet-200/val/<class>/*.JPEG

    i.e. you may need to reorganize the official val split into subfolders by class.
    """
    root = os.path.join(data_root, "tiny-imagenet-200")

    transform_train = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),  # ImageNet-like stats
            std=(0.229, 0.224, 0.225),
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
    test_ds = datasets.ImageFolder(val_dir, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(train_ds.classes)
    return train_loader, test_loader, num_classes, "TinyImageNet"


def get_flowers102_loaders(data_root: str,
                           batch_size: int,
                           num_workers: int = 4) -> Tuple[DataLoader, DataLoader, int, str]:
    """
    Uses torchvision.datasets.Flowers102.
    We take 'train' split as train and 'test' split as test.
    """
    try:
        Flowers102 = datasets.Flowers102
    except AttributeError:
        raise RuntimeError("Your torchvision version does not have Flowers102. "
                           "Please update torchvision or choose a different dataset.")

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    train_ds = Flowers102(root=data_root, split="train", download=True, transform=transform_train)
    test_ds = Flowers102(root=data_root, split="test", download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Flowers102 always has 102 classes
    return train_loader, test_loader, 102, "Flowers102"


# ----------------------------
# Training
# ----------------------------

def train_one_model(model: nn.Module,
                    train_loader: DataLoader,
                    device: torch.device,
                    epochs: int,
                    lr: float,
                    weight_decay: float,
                    mc_layers: List[MonteCarloDropout]):
    """
    Standard training loop with dropout p=0 for all MC dropout layers.
    """
    set_mc_dropout_p(mc_layers, 0.0)

    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = correct / total

        print(f"[Train] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    return model


# ----------------------------
# MC Dropout Evaluation
# ----------------------------

@torch.no_grad()
def mc_dropout_scores(model: nn.Module,
                      data_loader: DataLoader,
                      device: torch.device,
                      mc_layers: List[MonteCarloDropout],
                      p_dropout: float,
                      num_mc_passes: int,
                      num_classes: int) -> Dict[str, float]:
    """
    For each sample:
    - Run num_mc_passes forward passes, with dropout probability p_dropout
    - For each pass: get prediction and probability for true class
    - ACC_sample = mean(correctness over passes)
    - AUC_sample = mean(probability assigned to true class)  (per-sample "confidence" score)

    Finally:
    - Return std over all samples for ACC and AUC, plus mean if needed.
    """
    set_mc_dropout_p(mc_layers, p_dropout)

    model.to(device)
    model.eval()

    all_acc_samples = []
    all_auc_samples = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        batch_size = labels.size(0)

        all_probs = []
        all_preds = []

        for _ in range(num_mc_passes):
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)
            all_preds.append(torch.argmax(probs, dim=1))

        all_probs = torch.stack(all_probs, dim=0)      # (MC, B, C)
        all_preds = torch.stack(all_preds, dim=0)      # (MC, B)

        correct_per_pass = (all_preds == labels.unsqueeze(0)).float()  # (MC, B)
        acc_samples_batch = correct_per_pass.mean(dim=0)               # (B,)

        labels_expanded = labels.view(1, batch_size, 1).expand(num_mc_passes, batch_size, 1)
        true_probs = torch.gather(all_probs, dim=2, index=labels_expanded).squeeze(-1)  # (MC, B)
        auc_samples_batch = true_probs.mean(dim=0)                                       # (B,)

        all_acc_samples.append(acc_samples_batch.cpu().numpy())
        all_auc_samples.append(auc_samples_batch.cpu().numpy())

    all_acc_samples = np.concatenate(all_acc_samples, axis=0)
    all_auc_samples = np.concatenate(all_auc_samples, axis=0)

    metrics = {
        "acc_mean": float(all_acc_samples.mean()),
        "acc_std": float(all_acc_samples.std(ddof=1)),
        "auc_mean": float(all_auc_samples.mean()),
        "auc_std": float(all_auc_samples.std(ddof=1)),
    }
    return metrics


# ----------------------------
# Experiment Orchestration
# ----------------------------

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    set_seed(args.seed)

    # 1. Load dataset
    ds = args.dataset.lower()
    if ds == "cifar10":
        train_loader, test_loader, num_classes, dataset_name = get_cifar10_loaders(
            args.data_root, args.batch_size, args.num_workers
        )
    elif ds == "cifar100":
        train_loader, test_loader, num_classes, dataset_name = get_cifar100_loaders(
            args.data_root, args.batch_size, args.num_workers
        )
    elif ds == "svhn":
        train_loader, test_loader, num_classes, dataset_name = get_svhn_loaders(
            args.data_root, args.batch_size, args.num_workers
        )
    elif ds in ["tinyimagenet", "tiny-imagenet"]:
        train_loader, test_loader, num_classes, dataset_name = get_tinyimagenet_loaders(
            args.data_root, args.batch_size, args.num_workers
        )
    elif ds in ["flowers102", "flower102"]:
        train_loader, test_loader, num_classes, dataset_name = get_flowers102_loaders(
            args.data_root, args.batch_size, args.num_workers
        )
    else:
        raise ValueError("Unsupported dataset. Choose from: cifar10, cifar100, svhn, tinyimagenet, flowers102")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Dropout values
    if args.dropout_values is None:
        dropout_values = [i / 100.0 for i in range(1, 11)]  # 0.01 ... 0.10
    else:
        dropout_values = [float(v) for v in args.dropout_values]

    print(f"Dropout values: {dropout_values}")

    rows = []

    for model_name in args.models:
        print("\n" + "=" * 80)
        print(f"Model: {model_name} on {dataset_name}")
        print("=" * 80)

        model, mc_layers = build_model(
            model_name=model_name,
            num_classes=num_classes,
            init_dropout_p=0.0,
            pretrained=args.pretrained,
        )

        model = train_one_model(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            mc_layers=mc_layers,
        )

        ckpt_path = os.path.join(args.output_dir, f"{dataset_name}_{model_name}_centralized.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved trained model to {ckpt_path}")

        train_acc_stds = []
        train_auc_stds = []
        test_acc_stds = []
        test_auc_stds = []

        for p in dropout_values:
            print(f"\n[Model: {model_name}] Evaluating with dropout p={p:.3f}")

            train_metrics = mc_dropout_scores(
                model=model,
                data_loader=train_loader,
                device=device,
                mc_layers=mc_layers,
                p_dropout=p,
                num_mc_passes=args.num_mc_passes,
                num_classes=num_classes,
            )

            test_metrics = mc_dropout_scores(
                model=model,
                data_loader=test_loader,
                device=device,
                mc_layers=mc_layers,
                p_dropout=p,
                num_mc_passes=args.num_mc_passes,
                num_classes=num_classes,
            )

            print(f"  Train: ACC_std={train_metrics['acc_std']:.4f}, AUC_std={train_metrics['auc_std']:.4f}")
            print(f"  Test : ACC_std={test_metrics['acc_std']:.4f}, AUC_std={test_metrics['auc_std']:.4f}")

            train_acc_stds.append(train_metrics["acc_std"])
            train_auc_stds.append(train_metrics["auc_std"])
            test_acc_stds.append(test_metrics["acc_std"])
            test_auc_stds.append(test_metrics["auc_std"])

        col_names = ["dataset", "model", "split", "metric"] + [f"dropout_{p:.2f}" for p in dropout_values]

        def row_values(split, metric, values):
            return [dataset_name, model_name, split, metric] + list(values)

        rows.append(row_values("train", "ACC", train_acc_stds))
        rows.append(row_values("train", "AUC", train_auc_stds))
        rows.append(row_values("test", "ACC", test_acc_stds))
        rows.append(row_values("test", "AUC", test_auc_stds))

        # ACC plot
        plt.figure()
        plt.plot(dropout_values, train_acc_stds, marker="o", label="Train")
        plt.plot(dropout_values, test_acc_stds, marker="s", label="Test")
        plt.xlabel("Dropout probability")
        plt.ylabel("Std of ACC (per-sample)")
        plt.title(f"{dataset_name} - {model_name} - ACC std vs dropout")
        plt.legend()
        plt.grid(True)
        acc_plot_path = os.path.join(args.output_dir, f"{dataset_name}_{model_name}_ACC_std_vs_dropout.png")
        plt.savefig(acc_plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved ACC std plot to {acc_plot_path}")

        # AUC plot
        plt.figure()
        plt.plot(dropout_values, train_auc_stds, marker="o", label="Train")
        plt.plot(dropout_values, test_auc_stds, marker="s", label="Test")
        plt.xlabel("Dropout probability")
        plt.ylabel("Std of AUC-like score (per-sample)")
        plt.title(f"{dataset_name} - {model_name} - AUC std vs dropout")
        plt.legend()
        plt.grid(True)
        auc_plot_path = os.path.join(args.output_dir, f"{dataset_name}_{model_name}_AUC_std_vs_dropout.png")
        plt.savefig(auc_plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved AUC std plot to {auc_plot_path}")

    col_names = ["dataset", "model", "split", "metric"] + [f"dropout_{p:.2f}" for p in dropout_values]
    df = pd.DataFrame(rows, columns=col_names)
    csv_path = os.path.join(args.output_dir, f"{dataset_name}_dropout_stability_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results table to {csv_path}")
    print("Done.")


# ----------------------------
# Argument parser
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Dropout stability experiment (centralized)")

    # Data / general
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="cifar10 | cifar100 | svhn | tinyimagenet | flowers102")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    # Models
    parser.add_argument("--models", nargs="+",
                        default=["resnet18", "resnet34", "mobilenetv3_small", "mobilenetv3_large"],
                        help="resnet18, resnet34, mobilenetv3_small, mobilenetv3_large")
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained weights")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    # MC dropout experiment
    parser.add_argument("--dropout-values", nargs="*", default=None,
                        help="Optional list of dropout values, e.g. 0.01 0.02 ... 0.1; "
                             "default is 0.01..0.10")
    parser.add_argument("--num-mc-passes", type=int, default=5,
                        help="Number of Monte Carlo passes per sample")

    # Output
    parser.add_argument("--output-dir", type=str, default="./dropout_results",
                        help="Where to save checkpoints, plots, and CSV")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
