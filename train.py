import sys
import torch
import torch.nn as nn
import torchvision
import data
import model 
import ckpt

from config import IMG_SIZE
img_size = (IMG_SIZE, IMG_SIZE)

if len(sys.argv) < 3:
    print("./train.py tag num_epochs")
    exit(0)

tag = sys.argv[1]
num_epochs = int(sys.argv[2])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataloader = data.loader(10, img_size=img_size, shuffle=True)

unet = model.UNet()
unet = unet.to(device)

checkpoint = ckpt.Ckpt(unet, tag=tag)

ce_loss = nn.CrossEntropyLoss()
ce_loss = ce_loss.to(device)
optim = torch.optim.Adam(unet.parameters(), lr=0.001, weight_decay=0.0)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        img, seg = data
        img, seg = img.to(device), seg.to(device)

        outputs = unet(img)
        loss = ce_loss(outputs, seg)

        optim.zero_grad()
        loss.backward()
        optim.step()

        running_loss += loss

    print('[%d] loss: %.3f' % (epoch+1, running_loss / len(dataloader)))
    if epoch % 10 == 0:
        checkpoint.save(epoch)

checkpoint.save(num_epochs)

