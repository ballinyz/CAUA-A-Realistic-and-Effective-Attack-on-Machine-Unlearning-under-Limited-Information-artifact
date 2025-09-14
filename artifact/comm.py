import os

import torch
import numpy as np
import copy
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import TensorDataset, DataLoader

from approx_retraining import approx_retraining



def evaluate_unlearning(model, remove_x, remove_y,random_y ,x_train, y_train,test_x, test_y,lab):
    device = remove_x.device  # 假设 remove_x 是 cuda 的
    model = model.to(device)  # 把模型也搬到 cuda 上
    new_model = copy.deepcopy(model)

    unlearned_model = update_unlearn(new_model, remove_x, remove_y,random_y,x_train, y_train)
    if test_x is not None and test_y is not None:
        from torch.utils.data import DataLoader, TensorDataset
        unlearned_model.eval()
        test_dataset = TensorDataset(test_x, test_y)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = unlearned_model(x_batch)
                _, preds = outputs.max(1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        acc = correct / total
        print(f"✅ Updated model accuracy on test set: {acc:.4f}")

    acc_before_fix, acc_after_fix, is_diverged = evaluate_model_diff(
        model, unlearned_model, test_x, test_y, 'D:/pythonp/Macbsntb/resnet-18',lab
    )
    return acc_before_fix, acc_after_fix, is_diverged



def evaluate_model_diff(model, unlearned_model, test_x, test_y, log_path,lab,batch_size=128, device='cuda'):
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    lab1=lab
    model.eval()
    unlearned_model.eval()

    # 准备保存所有预测结果和真实标签
    all_labels = []
    all_preds_model = []
    all_preds_unlearned = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs_model = model(x_batch)
            outputs_unlearned = unlearned_model(x_batch)

            _, pred_model = outputs_model.max(1)
            _, pred_unlearned = outputs_unlearned.max(1)

            # 累加所有预测和标签（转为cpu，便于后续使用 sklearn）
            all_labels.extend(y_batch.cpu().tolist())
            all_preds_model.extend(pred_model.cpu().tolist())
            all_preds_unlearned.extend(pred_unlearned.cpu().tolist())

    # 整体准确率
    acc_model = sum([p == t for p, t in zip(all_preds_model, all_labels)]) / len(all_labels)
    acc_unlearned = sum([p == t for p, t in zip(all_preds_unlearned, all_labels)]) / len(all_labels)
    acc_diff = acc_model - acc_unlearned

    print(f"✅ 原模型准确率: {acc_model:.4f}")
    print(f"✅ 遗忘后模型准确率: {acc_unlearned:.4f}")
    print(f"⚖️ 两个模型整体精度之差: {acc_diff:.4f}")

    # 🎯 目标类 lab 精度对比
    lab_correct_model = sum([p == t for p, t in zip(all_preds_model, all_labels) if t ==  lab1])
    lab_correct_unlearned = sum([p == t for p, t in zip(all_preds_unlearned, all_labels) if t ==  lab1])
    lab_total = all_labels.count(lab1)

    lab_acc_model = lab_correct_model / lab_total if lab_total > 0 else 0.0
    lab_acc_unlearned = lab_correct_unlearned / lab_total if lab_total > 0 else 0.0
    lab_acc_diff = lab_acc_model - lab_acc_unlearned
    lab_f1_model = f1_score(all_labels, all_preds_model, labels=[lab1], average='macro')
    lab_f1_unlearned = f1_score(all_labels, all_preds_unlearned, labels=[lab1], average='macro')
    lab_f1_diff = lab_f1_model - lab_f1_unlearned

    print(f"\n🎯 目标类 {lab1} 在原模型上的精度: {lab_acc_model:.4f}，F1: {lab_f1_model:.4f}")
    print(f"🎯 目标类 { lab1} 在遗忘模型上的精度: {lab_acc_unlearned:.4f}，F1: {lab_f1_unlearned:.4f}")
    print(f"🎯 精度差: {lab_acc_diff:.4f}，F1差: {lab_f1_diff:.4f}")

    # 使用 sklearn 获取 per-class 精度、召回、F1
    print("\n📊 原模型分类报告：")
   # print(classification_report(all_labels, all_preds_model, digits=4))

    print("\n📊 遗忘后模型分类报告：")
    #print(classification_report(all_labels, all_preds_unlearned, digits=4))

    # 保存日志
    log_path = os.path.join(log_path, "log.txt")
    with open(log_path, 'a') as f:
        f.write(
            f'Original Model Acc: {acc_model:.4f}, Unlearned Model Acc: {acc_unlearned:.4f}, Acc Diff: {acc_diff:.4f}\n')
        f.write(
            f'Target Class {lab} Acc (Original): {lab_acc_model:.4f}, (Unlearned): {lab_acc_unlearned:.4f}, Diff: {lab_acc_diff:.4f}\n')
    return 1

def update_unlearn(model, remove_x, remove_y,random_y,
                   hvp_x, hvp_y, order=2, lr=0.002, device='cuda'):

    model.zero_grad()

    updated_model = approx_retraining(
        model=model,
        remove_x=remove_x,
        remove_y=remove_y,
        random_y=random_y,
        order=order,
        hvp_x=hvp_x,
        hvp_y=hvp_y,
        verbose=True,
        unlearn_type=0
    )

    return updated_model

