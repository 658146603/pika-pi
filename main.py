import os

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

model_path = './model/model.pth'

data_transform = transforms.Compose(
    [
        transforms.Resize(480),
        transforms.CenterCrop(480),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class MDataSet(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = path
        self.list = os.listdir(path)
        self.train = train
        if transform is None:
            self.transform = data_transform
        else:
            self.transform = transform

    def __getitem__(self, index):
        img_path = self.list[index]
        if self.train:
            if img_path.split('.')[0] == 'pikachu':
                label = 1
            else:
                label = 0
        else:
            label = int(img_path.split('.')[0])
        label = torch.as_tensor(label, dtype=torch.int64)
        img_path = os.path.join(self.path, img_path)
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.list)


train_dataset = MDataSet('./train', transform=data_transform)
test_dataset = MDataSet('./test', transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.con1 = nn.Sequential(
            # batch*3*480*480
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.con2 = nn.Sequential(
            # batch*16*240*240
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.con3 = nn.Sequential(
            # batch*32*120*120
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            # batch*64*60*60
            nn.Linear(64 * 60 * 60, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.opt = torch.optim.Adam(self.parameters())
        self.los = torch.nn.CrossEntropyLoss()

    def forward(self, _input):
        out = self.con1(_input)
        out = self.con2(out)
        out = self.con3(out)
        out = out.view(-1, 64 * 60 * 60)
        out = self.fc(out)
        return out

    def train_model(self, x, _y):
        out = self.forward(x)
        loss = self.los(out, _y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        print("loss:", loss.item())

    def test_model(self, x):
        return self.forward(x)


if __name__ == '__main__':
    net = Net().cuda()

    for tick in range(10):
        print('-----time: {}-----'.format(tick))
        for i, (data, y) in enumerate(train_loader):
            net.train_model(data.cuda(), y.cuda())
    net.eval()

    torch.save(net.state_dict(), model_path)

    t_net = Net().cuda()
    t_net.load_state_dict(torch.load(model_path))
    print('-----test-----')
    test_accurate = 0
    for image_label in test_loader:
        images, labels = image_label
        torch.no_grad()
        outputs = t_net.test_model(images.cuda())
        predict = f.softmax(outputs, dim=-1)
        mm, prediction = torch.max(outputs.data, 1)

        print(labels)
        print(prediction)

        test_accurate += torch.sum(prediction.data.cuda() == labels.data.cuda())
    print(test_accurate)
