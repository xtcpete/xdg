import torch
import random
import numpy as np
import os
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from scipy.special import softmax


def set_random_seed(seed):
    """Seed process-level RNGs used by Python, NumPy, and PyTorch."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=1, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _positive_class_scores(y_scores):
    y_scores = _to_numpy(y_scores)
    if y_scores.ndim == 1:
        return y_scores.reshape(-1)
    if y_scores.ndim != 2 or y_scores.shape[1] < 2:
        raise ValueError(
            f"Expected y_scores with shape (N,) or (N, C>=2), got {y_scores.shape}."
        )
    return softmax(y_scores, axis=1)[:, 1]


def compute_ap(y_true, y_scores):
    y_true = _to_numpy(y_true).reshape(-1)
    y_scores = _positive_class_scores(y_scores)
    return average_precision_score(y_true, y_scores)


def compute_precision_at_recall(y_true, y_scores, target_recall=0.85):
    y_true = _to_numpy(y_true).reshape(-1)
    y_scores = _positive_class_scores(y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    valid_precision = precision[recall >= float(target_recall)]
    if valid_precision.size == 0:
        return 0.0
    return float(valid_precision.max())


def compute_recall_at_precision(y_true, y_scores, target_precision=0.99):
    y_true = _to_numpy(y_true).reshape(-1)
    y_scores = _positive_class_scores(y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    valid_recall = recall[precision >= float(target_precision)]
    if valid_recall.size == 0:
        return 0.0
    return float(valid_recall.max())


def compute_validation_metrics(y_true, y_scores):
    y_true = _to_numpy(y_true).reshape(-1)
    y_scores = _positive_class_scores(y_scores)

    try:
        ap = float(average_precision_score(y_true, y_scores))
    except ValueError:
        ap = float("nan")

    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        roc_auc = float("nan")

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    valid_precision = precision[recall >= 0.85]
    prec_at_recall = float(valid_precision.max()) if valid_precision.size else 0.0

    valid_recall = recall[precision >= 0.99]
    recall_at_precision = float(valid_recall.max()) if valid_recall.size else 0.0

    return {
        "ap": ap,
        "roc_auc": roc_auc,
        "prec_at_recall_0_85": prec_at_recall,
        "recall_at_prec_0_99": recall_at_precision,
    }


def plot_pr_curve(y_true, y_scores, writer, step=None, epoch=None, name=None):
    y_scores = _positive_class_scores(y_scores)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    AP = average_precision_score(y_true, y_scores)
    figure = plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.grid()
    plt.title('PR Curve AP=%.4f'%AP)
    
    figure.canvas.draw()
    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.frombuffer(figure.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(figure.canvas.get_width_height()[::-1] + (4,))
    # Remove alpha channel and normalize into 0-1 range for TensorBoard
    img = img[:, :, :3] / 255.0
    img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8
    img = np.transpose(img, (0,2,1))
    # Add figure in numpy "image" to TensorBoard writer
    if step is not None:
        writer.add_image(name+'_step', img, step)
    else:
        writer.add_image(name+'_epoch', img, epoch)
    plt.close(figure)
