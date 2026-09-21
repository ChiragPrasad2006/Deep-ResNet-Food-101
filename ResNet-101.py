import torch
import torchvision
import torchmetrics
from torch.utils.data import DataLoader,random_split
from torchvision import datasets,models,transforms
from torch import nn

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

#Transform dataset/define Transformer size and convert to tensor
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor()])
#load dataset
full_dataset=torchvision.datasets.ImageFolder(root="data/images",transform=transform)
print(len(full_dataset))
print(full_dataset.class_to_idx)

#split dataset
train_size=int(0.8*len(full_dataset))
test_size=int(len(full_dataset)-train_size)
train_data,test_data=torch.utils.data.random_split(full_dataset,[train_size,test_size],generator=torch.Generator().manual_seed(0))
print(len(train_data))
print(len(test_data))

train_dataloader=torch.utils.data.DataLoader(dataset=train_data,batch_size=64,shuffle=True)
test_dataloader=torch.utils.data.DataLoader(dataset=test_data,batch_size=64,shuffle=False)

#model
ResNet_Food_101=ResNet(Bottleneck,[3,4,23,3],num_classes=101)
torch.manual_seed(0)

#optimizer
optimizer=torch.optim.AdamW(params=ResNet_Food_101.parameters(),lr=1e-4,weight_decay=1e-6)
#loss function
loss_fn=nn.CrossEntropyLoss()
#accuracy_fn
acc_fn=torchmetrics.Accuracy(task="multiclass",num_classes=101)

#train and test loop
def train_step(model:torch.nn.Module,dataloader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module,optimizer:torch.optim.Optimizer,acc_fn:torchmetrics.Accuracy):
    model.train()
    train_loss=0
    train_acc=0
    for batch,(X,y) in enumerate(dataloader):
        y_pred_logits=model(X)
        loss=loss_fn(y_pred_logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_labels=torch.argmax(y_pred_logits,dim=1)
        train_loss+=loss.item()
        train_acc+=acc_fn(y_pred_labels,y)
    train_loss=train_loss/len(dataloader)
    train_acc=train_acc/len(dataloader)
    return train_loss,train_acc

def test_step(model:torch.nn.Module,dataloader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module,acc_fn:torchmetrics.Accuracy):
    model.eval()
    test_loss=0
    test_acc=0
    with torch.inference_mode():
        for batch,(X,y) in enumerate(dataloader):
            test_pred_logits=model(X)
            loss=loss_fn(test_pred_logits,y)
            test_loss+=loss.item()
            test_pred_labels=torch.argmax(test_pred_logits,dim=1)
            test_acc+=acc_fn(test_pred_labels,y)
    test_loss=test_loss/len(dataloader)
    test_acc=test_acc/len(dataloader)
    return test_loss,test_acc

from tqdm.auto import tqdm
torch.manual_seed(42)
num_epochs=50

#training and testing loop
for epoch in tqdm(range(num_epochs)):
    train_loss,train_acc=train_step(model=ResNet_Food_101,dataloader=train_dataloader,loss_fn=loss_fn,optimizer=optimizer,acc_fn=acc_fn)
    test_loss,test_acc=test_step(model=ResNet_Food_101,dataloader=test_dataloader,loss_fn=loss_fn,acc_fn=acc_fn)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

