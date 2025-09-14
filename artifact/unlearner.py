import argparse
import random
from model import  *
from dataset import *
from un import *
from comm import evaluate_unlearning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = 'D:/pythonp/alldataset'
lab=2
poison_amount=100
def get_parser():
    parser = argparse.ArgumentParser(description="Unlearning experiment script")
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'stl10'],
                        help='Name of the dataset to use')
    parser.add_argument('--unlearn_type', type=str, default='retrain',
                        choices=['retrain', 'gradient'],
                        help='Type of unlearning method to apply')
    parser.add_argument('--network', type=str, default='resnet18',
                        choices=['resnet18', 'mlp', 'vgg16'],
                        help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--model_path', type=str, default='D:/pythonp/Macbsntb/gen/best_model.pth', help='Path to the trained model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--return_type', type=str, default='dataloader',
                        choices=['dataloader', 'tensor'],
                        help='Return format of dataset')
    return parser

def normal_unlearn(args):
    print(f"Running normal unlearning using {args.unlearn_type} method on {args.dataset} with {args.network} model.")
    # TODO: 实现正常遗忘逻辑
    pass

def unlearn_attack(args):
    # ==== 1. 获取类别数量 ====
    if args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset in ['cifar10', 'stl10']:
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # ==== 2. 加载模型结构 ====
    if args.network == 'resnet18':
        model = getResNet_18(num_classes=num_classes, weight_path=args.model_path)
    elif args.network == 'vgg16':
        model = getVGG_16(num_classes=num_classes, weight_path=args.model_path)
    elif args.network == 'efficientnet':
        model = getEfficientNet(num_classes=num_classes, weight_path=args.model_path)
    else:
        raise ValueError(f"Unsupported network: {args.network}")

    # ==== 3. 加载数据集 ====
    (x_train, y_train), (x_test, y_test)= load_dataset(
        name=args.dataset)
    # 获取指定标签样本的索引
    target_indices = (y_train == lab).nonzero(as_tuple=True)[0]
    selected_indices = random.sample(target_indices.tolist(), poison_amount)

    # 提取样本
    remove_x = x_train[selected_indices].to(device)
    remove_y = y_train[selected_indices].to(device)
    selected_indices = torch.tensor(selected_indices, device=device)
    # 生成与 selected_indices 相同大小的随机标签
    random_y = torch.randint(0, num_classes, (selected_indices.size(0),), device=device)
    # 添加噪声
    best_noise = torch.load('D:/pythonp/Macbsntb/gen/best_noise.pth')

    remove_x = apply_noise_patch(best_noise, remove_x.cpu(), mode='change')
    remove_x = remove_x.to(device)
    acc_before_fix, acc_after_fix, is_diverged=evaluate_unlearning(model, remove_x,remove_y,random_y,x_train, y_train,x_test, y_test,lab)

    print(acc_before_fix, acc_after_fix)

    return is_diverged

def main():
    parser = get_parser()
    args = parser.parse_args()
    ok=unlearn_attack(args)


if __name__ == '__main__':
    main()
