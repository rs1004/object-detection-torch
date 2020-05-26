from dataset import MyDataset
import torch
import torchvision.transforms as transforms


transform = transforms.Compose([transforms.ToTensor()])
dataset = MyDataset(data_dir='/work/data/VOCdevkit/VOC2007',
                    data_list_file_name='trainval.txt',
                    transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                         shuffle=True, num_workers=2)

for images, labels, masks in dataloader:
    print(images)
    break