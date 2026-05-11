import torch
import torch.nn as nn
import torch.optim as opt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

device="cuda" if torch.cuda.is_available() else "cpu"
print("using device " ,device)
transform=transforms.ToTensor()
train_data=MNIST(root="data",train=True,download=True,transform=transform)
test_data=MNIST(root="data",train=True,download=True,transform=transform)
train_loader=DataLoader(train_data, batch_size=64,shuffle=False )
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)
class digitmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(28*28,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,10))
    def forward(self,x):
        return self.net(x)
model=digitmodel().to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=opt.Adam(model.parameters(),lr=0.001)
epochs=100
for epoch in range (epochs):
    model.train()
    total_loss=0
    for images,labels in train_loader:
        images =images.view(images.size(0),-1).to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"epoch{epoch+1}/{epochs}|loss:{total_loss:.4f}")
model.eval()
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        images=images.view(images.size(0),-1).to(device)
        labels=labels.to(device)
        outputs=model(images)
        predictions=outputs.argmax(dim=1)
        correct+=(predictions==labels).sum().item()
        total+=labels.size(0)
accuracy=100*correct/total
print(f"total accuracy : {accuracy:.2f}%")
torch.save(model.state_dict(),"minst_model.pth")
print("model saved as  minst_model.pth")
index=6
image,true_label=test_data[index]
plt.imshow(image.squeeze(),cmap="gray")
plt.title(f"actual label : {true_label}")
plt.axis("off")
plt.show()
image_flat=image.view(1,-1).to(device)
with torch.no_grad():
    output=model(image_flat)
    predicted_label=output.argmax(dim=1).item()
print("user picked image index", index)
print("actual label" ,true_label)
print("model prediction ", predicted_label)
