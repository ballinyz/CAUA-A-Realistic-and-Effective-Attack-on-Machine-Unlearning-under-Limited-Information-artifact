import numpy as np
import torch
import torchvision
from sklearn.metrics import classification_report
from torch import nn
from torch.autograd import grad
from torchvision import transforms
import random
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_gradients_diff(model, remove_x, remove_y,unlearn_type,random_y,batch_size=64):
    model.eval()
    total_diff = None

    criterion = nn.CrossEntropyLoss()

    for i in range(0, len(remove_x), batch_size):
        batch_x = remove_x[i:i + batch_size]
        batch_y = remove_y[i:i + batch_size]
        random_y1=random_y[i:i + batch_size]
        (c, w, h) = batch_x[0].shape
        current_batch_size = batch_x.size(0)
        device = batch_x.device
        if unlearn_type == 0:
            x_d = batch_x

        elif unlearn_type == 1:
            img = torch.rand((current_batch_size, c, w, h), dtype=torch.float32, device=device) * 2 - 1
            x_d = img

        elif unlearn_type == 2:
            img = torch.zeros((current_batch_size, c, w, h), dtype=torch.float32, device=device)
            x_d = img

        elif unlearn_type == 3:
            img = torch.ones((current_batch_size, c, w, h), dtype=torch.float32, device=device)
            x_d = img
        else:
            raise ValueError("Unsupported unlearn_type. Use 0 (original), 1 (random), 2 (black), or 3 (white).")

        outputs_clean = model(batch_x)
        loss_unlean = criterion(outputs_clean, batch_y)

        # 计算噪声图像的输出和损失
        outputs_noise = model(x_d)
        loss_noise = criterion(outputs_noise, random_y1)

        # 两个loss相减
        loss_diff =   - loss_unlean

        # 计算 loss_diff 的梯度
        loss_diff.backward()

        # 获取模型的梯度
        grads_diff = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]

        # 如果total_diff是None，则初始化它
        if total_diff is None:
            total_diff = grads_diff
        else:
            # 否则将当前的梯度差累加到total_diff
            total_diff = [td + gd for td, gd in zip(total_diff, grads_diff)]

    return total_diff
def hessian_vector_product(model, x, y, v, criterion=nn.CrossEntropyLoss()):
    """
    Calculate Hessian-Vector Product (HVP) for only the trainable (unfrozen) parameters of the model.
    Args:
        model: PyTorch model.
        x: Input tensor.
        y: Target tensor.
        v: Vector for HVP.
        criterion: Loss function.
    Returns:
        hvp: Hessian-vector product.
    """
    # 清空梯度
    model.zero_grad()
    model.eval()
    x, y = x.to(device), y.to(device)  # 确保输入和标签在同一设备上
    # 前向传播和损失计算
    outputs = model(x)
    loss = criterion(outputs, y)
    # 选择解冻的参数
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    # 计算梯度，仅对解冻的参数
    grads = grad(loss, trainable_params, create_graph=True)
    # 仅计算线性层（fc层）的Hessian向量积
    hvp = grad(grads,trainable_params, grad_outputs=v, retain_graph=True)
    return hvp


def get_inv_hvp_lissa(model, x, y, v, hvp_batch_size, scale, damping,
                      iterations=1, verbose=False, repetitions=1,
                      early_stopping=True, patience=20, criterion=nn.CrossEntropyLoss()):
    """
    利用LISSA方法迭代求解逆海森矩阵与向量v的乘积：H^{-1} v
    """
    n_batches = int(np.ceil(x.shape[0] / hvp_batch_size)) if iterations == -1 else iterations
    estimate = None  # 最终返回的估计值

    for r in range(repetitions):
        u = [torch.zeros_like(p) for p in model.parameters() if p.requires_grad]
        update_min = [float('inf'), 0]  # 最小变化量和未改善计数

        for i in range(n_batches):
            start = (i * hvp_batch_size) % x.shape[0]
            end = min((i + 1) * hvp_batch_size, x.shape[0])
            x_batch = x[start:end]
            y_batch = y[start:end]

            hvp = hessian_vector_product(model, x_batch, y_batch, u, criterion)
            # LISSA公式：u_new = v + (1 - damping) * u - hvp / scale
            new_u = [vi + (1 - damping) * ui - hi / scale for vi, ui, hi in zip(v, u, hvp)]

            # 判断 early stopping
            update_norm = sum(torch.norm(ui - nui).item() for ui, nui in zip(u, new_u))

            if verbose:
                print(f"[Repetition {r+1}] Iter {i+1}: update norm = {update_norm:.6f}")

            if early_stopping:
                if update_norm > update_min[0]:
                    update_min[1] += 1
                    if update_min[1] >= patience:
                        print(f"Early stopped at iter {i+1} with norm {update_norm:.6f}")
                        break
                else:
                    update_min = [update_norm, 0]

            u = new_u

        # 平均结果
        res_upscaled = [ui / scale for ui in u]
        if estimate is None:
            estimate = [rui / repetitions for rui in res_upscaled]
        else:
            estimate = [e + rui / repetitions for e, rui in zip(estimate, res_upscaled)]

    return estimate, False



def approx_retraining(model, remove_x, remove_y, random_y, order=1,
                      hvp_x=None, hvp_y=None,
                      conjugate_gradients=False, verbose=False,
                      unlearn_type=0, lr=0.001):
    """
    使用Influence Function近似更新模型参数以达到遗忘目的。
    """
    model.train()

    if order == 1:
        grad_diff = get_gradients_diff(model, remove_x, remove_y, unlearn_type, random_y)
        d_theta = grad_diff
        diverged = False

    elif order == 2:
        lr =1
        grad_diff = get_gradients_diff(model, remove_x, remove_y, unlearn_type, random_y)
        if sum(g.sum() for g in grad_diff).item() == 0:
            d_theta = grad_diff
            diverged = False
        elif conjugate_gradients:
            raise NotImplementedError("共轭梯度方法未实现")
        else:
            assert hvp_x is not None and hvp_y is not None
            d_theta, diverged = get_inv_hvp_lissa(model, hvp_x, hvp_y, grad_diff,
                                                  hvp_batch_size=32, scale=65000,
                                                  damping=0.00001, iterations=150,
                                                  verbose=verbose, repetitions=1)

    # 参数更新：ψθ ← ψθ − H⁻¹ Δ∇ℓ
    if order != 0:
        for param, delta in zip(model.parameters(), d_theta):
            if param.requires_grad:
                param.data -= lr * delta  # 使用影响函数计算出的方向更新参数

    return model
