import argparse
import json
import seaborn as sns
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
from dataset import PascalVOCDataset
from model import Yolo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_num', type=int, default=-1)
    parser.add_argument('--imsize', type=int, default=448)
    parser.add_argument('--grid_num', type=int, default=7)
    parser.add_argument('--bbox_num', type=int, default=2)
    parser.add_argument('--class_num', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_weights_path', default='./yolo_net.pth')
    parser.add_argument('--inference_out_dir', default='./inference/')

    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor()])
    dataset = PascalVOCDataset(
        data_dirs=['/work/data/VOCdevkit/VOC2007'],
        data_list_file_name='test.txt',
        imsize=args.imsize,
        grid_num=args.grid_num,
        bbox_num=args.bbox_num,
        class_num=args.class_num,
        transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2)

    net = Yolo(
        grid_num=args.grid_num,
        bbox_num=args.bbox_num,
        class_num=args.class_num,
        is_train=False)

    with open(Path(__file__).parent / 'labelmap.json', 'r') as f:
        labelmap = json.load(f)['PascalVOC']

    Path(args.inference_out_dir).mkdir(parents=True, exist_ok=True)

    if Path(args.model_weights_path).exists():
        print('weights loaded.')
        net.load_state_dict(torch.load(Path(args.model_weights_path)))

    grid_len = args.imsize // args.grid_num
    current_palette = sns.color_palette('hls', n_colors=args.class_num)
    n = 1
    total = len(dataloader) if args.inference_num == -1 else args.inference_num
    with torch.no_grad():
        for images, _, _ in tqdm(dataloader, total=total):
            outputs = net(images)
            for image, output in zip(images, outputs):
                img = Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                output = output.numpy()

                draw = ImageDraw.Draw(img)
                for i in range(args.grid_num):
                    for j in range(args.grid_num):
                        for k in range(args.bbox_num):
                            if output[4 + 5*k, i, j] >= 0.5:
                                # calc coord
                                x, y, w, h, c = output[0 + 5 * k:5 + 5*k, i, j]
                                X, Y = grid_len * (i + x), grid_len * (j + y)
                                W, H = args.imsize * (w ** 2), args.imsize * (h ** 2)
                                left_top = (max(X - W / 2, 0), max(Y - H / 2, 0))
                                right_bottom = (min(X + W / 2, args.imsize), min(Y + H / 2, args.imsize))

                                # set text
                                class_id = np.argmax(output[args.bbox_num * 5:, i, j])
                                text = f' {labelmap[class_id]} {str(round(c, 3))}'
                                text_loc = (max(X - W / 2, 0), max(Y - H / 2, 0) - 11)
                                text_back_loc = (max(X - W / 2, 0) + len(text) * 6, max(Y - H / 2, 0))

                                # draw bbox
                                color = tuple((np.array(current_palette[class_id]) * 255).astype(int))
                                draw.rectangle(left_top + right_bottom, outline=color)
                                draw.rectangle(text_loc + text_back_loc, fill=color, outline=color)
                                draw.text(text_loc, text, fill=(0, 0, 0, 0))
                img.save(Path(args.inference_out_dir) / f'{n:06}.png')
                n += 1
                if n == args.inference_num:
                    break
            if n == args.inference_num:
                break
