import torch
import json
import argparse
import torchvision.transforms as transforms
from dataset import PascalVOCDataset
from model import Yolo
from pathlib import Path
from tqdm import tqdm
from datetime import date
from subprocess import check_output


OUTPUT_FORMAT = '''
# EVALUATION REPORT

## REPORTING DATE
{date}

## RUNTIME
```
{runtime}
```

## CONFIG
{config_table}

## SCORES
{score_table}
'''


def get_bboxes(tensor, imsize, class_num, is_out=False):
    channel_num, grid_num, _ = tensor.shape
    bbox_num = (channel_num - class_num) // 5
    grid_len = imsize // grid_num
    bboxes = {i: [] for i in range(class_num)}
    for i in range(grid_num):
        for j in range(grid_num):
            class_id = torch.argmax(tensor[5*bbox_num:, i, j]).item()
            for k in range(bbox_num):
                x, y, w, h, c = tensor[0+5*k:5+5*k, i, j]
                if c <= 0.5:
                    continue
                X, Y = grid_len * (i + x), grid_len * (j + y)
                W, H = w ** 2 * imsize, h ** 2 * imsize
                bboxes[class_id].append([c, [X - W / 2, Y - H / 2, X + W / 2, Y + H / 2]])
    if is_out:
        for class_id in range(class_num):
            bboxes[class_id].sort(key=lambda x: x[0], reverse=True)
    return bboxes


def calc_iou(bbox_1, bbox_2):
    l1, t1, r1, b1 = bbox_1
    l2, t2, r2, b2 = bbox_2
    w = min(r1, r2) - max(l1, l2)
    h = min(b1, b2) - max(t1, t2)
    if (w <= 0) or (h <= 0):
        return 0
    s1 = (r1 - l1) * (b1 - t1)
    s2 = (r2 - l2) * (b2 - t2)
    s1_s2 = w * h
    return s1_s2 / (s1 + s2 - s1_s2)


def make_report(result_dict, args, epsilon=1.0e-5):
    d = date.today().isoformat()

    runtime = check_output(['nvidia-smi']).decode()

    config_table = ['|item|value|', '|-|-|']
    for k, v in args.__dict__.items():
        config_table.append(f'|{k}|{v}|')

    with open(Path(__file__).parent / 'labelmap.json', 'r') as f:
        label_map = json.load(f)['PascalVOC']

    score_table = ['|label|average precision|average recall|', '|-|-|-|']
    sum_ap = 0
    sum_ar = 0
    for class_id, result in result_dict.items():
        average_precision = sum(result['precision']) / (len(result['precision']) + epsilon)
        average_recall = sum(result['recall']) / (len(result['recall']) + epsilon)
        sum_ap += average_precision
        sum_ar += average_recall
        score_table.append(f'|{label_map[class_id]}|{_float2str(average_precision)}|{_float2str(average_recall)}|')

    score_table.append(f'|**mean**|**{_float2str(sum_ap/len(label_map))}**|**{_float2str(sum_ar/len(label_map))}**|')

    report = OUTPUT_FORMAT.format(
        date=d,
        runtime=runtime,
        config_table='\n'.join(config_table),
        score_table='\n'.join(score_table)
    )

    Path(args.evaluation_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.evaluation_dir) / f'report_{d}.md', 'w') as f:
        f.write(report)


def _float2str(val):
    return str(round(val, 3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imsize', type=int, default=448)
    parser.add_argument('--grid_num', type=int, default=7)
    parser.add_argument('--bbox_num', type=int, default=2)
    parser.add_argument('--class_num', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_weights_path', type=str, default='./yolo_net.pth')
    parser.add_argument('--evaluation_dir', type=str, default='./eval/')

    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor()
         ])

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
        shuffle=True,
        num_workers=2)

    net = Yolo(
        grid_num=args.grid_num,
        bbox_num=args.bbox_num,
        class_num=args.class_num,
        is_train=False)

    if Path(args.model_weights_path).exists():
        print('weights loaded.')
        net.load_state_dict(torch.load(Path(args.model_weights_path)))

    result_dict = {i: {'precision': [], 'recall': []} for i in range(args.class_num)}
    for images, labels, _ in tqdm(dataloader, total=len(dataloader)):
        outputs = net(images)
        for output, label in zip(outputs, labels):
            output_bboxes = get_bboxes(tensor=output, imsize=args.imsize, class_num=args.class_num, is_out=True)
            label_bboxes = get_bboxes(tensor=label, imsize=args.imsize, class_num=args.class_num)
            for class_id in range(args.class_num):
                t_num = 0
                label_num = len(label_bboxes[class_id])
                # 未検出時
                if (len(output_bboxes[class_id]) == 0) and label_num > 0:
                    recall = 0
                    for _ in range(label_num):
                        result_dict[class_id]['recall'].append(recall)
                # 検出時
                for i, (_, output_bbox) in enumerate(output_bboxes[class_id], start=1):
                    ious = [calc_iou(output_bbox, label_bbox) for _, label_bbox in label_bboxes[class_id]]
                    if len(ious) > 0:
                        max_id = ious.index(max(ious))
                        if ious[max_id] >= 0.5:
                            t_num += 1
                            label_bboxes[class_id].pop(max_id)
                    precision = t_num / i
                    recall = t_num / label_num if label_num > 0 else None
                    result_dict[class_id]['precision'].append(precision)
                    if recall is not None:
                        result_dict[class_id]['recall'].append(recall)

    # レポート出力
    make_report(result_dict=result_dict, args=args)
