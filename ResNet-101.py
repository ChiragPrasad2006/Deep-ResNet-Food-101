import torch
import torchvision
from torchvision import transforms
from torch import nn
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForImageClassification

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, planes, stride=1, downsample=None):
        super().__init__()
        # 1×1 convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            planes,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        # 3×3 convolution
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        # 1×1 convolution
        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(
            planes * self.expansion
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=101):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.layer1 = self._make_layer(
            block,
            64,
            layers[0]
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2
        )
        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )
        self.fc = nn.Linear(
            512 * block.expansion,
            num_classes
        )
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if (
            stride != 1
            or self.in_channels != planes * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(
                    planes * block.expansion
                )
            )
        layers = []
        # First block
        layers.append(
            block(
                self.in_channels,
                planes,
                stride,
                downsample
            )
        )
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    planes
                )
            )
        return nn.Sequential(*layers)
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

# train and test loop functions
def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for X, y in dataloader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)    
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device,
            dtype=torch.bfloat16,
            enabled=bf16
        ):
            y_pred_logits = model(X).logits
            loss = loss_fn(y_pred_logits, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.detach()
        correct += (y_pred_logits.argmax(dim=1) == y).sum()
        total += y.size(0)
    epoch_loss = (train_loss / len(dataloader)).item()
    epoch_acc = (correct / total).item()
    return epoch_loss, epoch_acc

def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16,
                enabled=bf16
            ):
                test_pred_logits = model(X).logits
                loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = torch.argmax(test_pred_logits, dim=1)
            correct += (test_pred_labels == y).sum().item()
            total += y.size(0)
    test_loss = test_loss / len(dataloader)
    test_acc = correct / total
    return test_loss, test_acc


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Keep augmentation in the training pipeline only.
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Use the container path when available, with the repository-local path as
    # a fallback for running this script from the project directory.
    DATA_DIR = Path("/data/images") if Path("/data/images").is_dir() else Path("data/images")
    EXPECTED_NUM_CLASSES = 101
    EXPECTED_IMAGES_PER_CLASS = 1000
    BATCH_SIZE = 64

    # Load the dataset once to determine its labels and split its indices.
    full_dataset = torchvision.datasets.ImageFolder(root=DATA_DIR)
    if len(full_dataset.classes) != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_NUM_CLASSES} food classes, "
            f"found {len(full_dataset.classes)} in {DATA_DIR}"
        )

    images_per_class = {
        class_name: sum(1 for _, class_id in full_dataset.samples if class_id == class_index)
        for class_name, class_index in full_dataset.class_to_idx.items()
    }
    invalid_class_counts = {
        class_name: count for class_name, count in images_per_class.items()
        if count != EXPECTED_IMAGES_PER_CLASS
    }
    if invalid_class_counts:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES_PER_CLASS} images per class, "
            f"invalid counts: {invalid_class_counts}"
        )

    # ImageFolder sorts class folders alphabetically, giving every class a
    # stable label in the required 0-100 range.
    CLASS_TO_ID = {
        class_name: class_id
        for class_id, class_name in enumerate(full_dataset.classes)
    }
    ID_TO_CLASS = {class_id: class_name for class_name, class_id in CLASS_TO_ID.items()}
    if set(CLASS_TO_ID.values()) != set(range(EXPECTED_NUM_CLASSES)):
        raise ValueError("Class labels must cover every integer from 0 through 100")

    print(f"Total samples: {len(full_dataset)}")
    print(f"Classes: {CLASS_TO_ID}")
    print(f"bf16 enabled: {bf16} (device: {device})")

    # Split indices deterministically, while allowing different transforms per split.
    train_size = int(0.9 * len(full_dataset))
    test_size = int(len(full_dataset) - train_size)
    split_generator = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(full_dataset), generator=split_generator).tolist()
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]

    train_data = torch.utils.data.Subset(
        torchvision.datasets.ImageFolder(
            root=DATA_DIR,
            transform=train_transform
        ),
        train_indices
    )
    test_data = torch.utils.data.Subset(
        torchvision.datasets.ImageFolder(
            root=DATA_DIR,
            transform=test_transform
        ),
        test_indices
    )
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # DataLoaders with multi-process pre-fetching
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=device == "cuda",
        persistent_workers=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device == "cuda"
    )

    # Model
    num_classes = EXPECTED_NUM_CLASSES
    ResNet_Food_101 = AutoModelForImageClassification.from_pretrained(
        "microsoft/resnet-101"
    )
    ResNet_Food_101.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(ResNet_Food_101.config.hidden_sizes[-1], num_classes)
    )
    ResNet_Food_101.config.num_labels = num_classes
    ResNet_Food_101.config.id2label = ID_TO_CLASS
    ResNet_Food_101.config.label2id = CLASS_TO_ID
    ResNet_Food_101 = ResNet_Food_101.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        params=ResNet_Food_101.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )
    # Loss function
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    num_epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )
    checkpoint_interval = 5
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # Training and testing loop
    for epoch in tqdm(range(num_epochs)):
        train_loss, train_acc = train_step(
            model=ResNet_Food_101,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer
        )
        test_loss, test_acc = test_step(
            model=ResNet_Food_101,
            dataloader=test_dataloader,
            loss_fn=loss_fn
        )
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
        scheduler.step()

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": ResNet_Food_101.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }

        # Keep the newest completed epoch available for resuming.
        torch.save(checkpoint, MODEL_PATH / "ResNet_Food_101_latest_checkpoint.pth")

        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = MODEL_PATH / f"ResNet_Food_101_epoch_{epoch + 1:03d}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")

    MODEL_NAME = "ResNet_Food_101_v1.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=ResNet_Food_101.state_dict(), f=MODEL_SAVE_PATH)
