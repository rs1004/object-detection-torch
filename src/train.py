from dataset import PascalVOCDataset
from model import Yolo
from loss import yolo_loss
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imsize', type=int, default=448)
    parser.add_argument('--grid_num', type=int, default=7)
    parser.add_argument('--bbox_num', type=int, default=2)
    parser.add_argument('--class_num', type=int, default=20)
    parser.add_argument('--l_coord', type=float, default=5.)
    parser.add_argument('--l_noobj', type=float, default=.5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = PascalVOCDataset(
        data_dirs=['/work/data/VOCdevkit/VOC2007', '/work/data/VOCdevkit/VOC2012'],
        data_list_file_name='trainval.txt',
        imsize=args.imsize,
        grid_num=args.grid_num,
        bbox_num=args.bbox_num,
        class_num=args.class_num,
        l_coord=args.l_coord,
        l_noobj=args.l_noobj,
        transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2)

    net = Yolo(
        grid_num=args.grid_num,
        bbox_num=args.bbox_num,
        class_num=args.class_num)

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    running_loss = 0.0
    for epoch in range(args.epochs):
        for i, (images, labels, masks) in enumerate(dataloader):
            # TODO to GPU device

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = yolo_loss(input=outputs, target=labels, mask=masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f'[Epoch {epoch} / {args.epoch}] {i} / {args.batch_size / len(dataloader)}  loss: {running_loss}')
        running_loss = 0.0

print('Finished Training')

PATH = './yolo_net.pth'
torch.save(net.state_dict(), PATH)
