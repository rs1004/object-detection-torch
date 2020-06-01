from dataset import PascalVOCDataset
from model import Yolo
from loss import yolo_loss
from tqdm import tqdm
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
    parser.add_argument('--save_period', type=int, default=5)
    parser.add_argument('--save_path', type=str, default='./yolo_net.pth')
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

    optimizer = optim.Adam(net.parameters(), lr=0.00009, weight_decay=0.00001)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    running_loss = 0.0
    for epoch in range(args.epochs):
        with tqdm(dataloader, total=len(dataloader)) as pbar:
            for i, (images, labels, masks) in enumerate(pbar):
                # description
                pbar.set_description(f'[Epoch {epoch+1}/{args.epochs}] loss: {running_loss}')

                # to GPU device
                images = images.to(device)
                labels = labels.to(device)
                masks = masks.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(images)
                loss = yolo_loss(input=outputs, target=labels, mask=masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            running_loss = 0.0
            if epoch % args.save_period == 0:
                torch.save(net.state_dict(), args.save_path)

print('Finished Training')

