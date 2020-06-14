from dataset import PascalVOCDataset
from model import Yolo
from loss import yolo_loss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
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
    parser.add_argument('--model_weights_path', type=str, default='./yolo_net.pth')
    parser.add_argument('--min_loss_path', type=str, default='./min_loss.txt')
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value='random')
        ])
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

    if Path(args.model_weights_path).exists():
        print('weights loaded.')
        net.load_state_dict(torch.load(Path(args.model_weights_path)))
    
    if Path(args.min_loss_path).exists():
        print('min_loss loaded.')
        with open(Path(args.min_loss_path), 'r') as f:
            min_loss = float(f.readlines()[0])
    else:
        min_loss = None

    optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=0.00001)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    writer = SummaryWriter(log_dir='./logs')

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
                writer.add_scalar('loss', running_loss, i)

            if (min_loss is None) or (running_loss < min_loss):
                torch.save(net.state_dict(), args.model_weights_path)
                min_loss = running_loss
                with open(Path(args.min_loss_path), 'w') as f:
                    f.write(str(min_loss))
            running_loss = 0.0

print('Finished Training')

writer.close()
