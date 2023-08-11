import sys
import os

sys.path.append(
    os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".")), "metrics")
)
from typing import List, Optional, Union
from functional import iou_score, f1_score, precision, recall, get_stats, accuracy


def metrics(
    output,
    target,
    metrics_measures=None,
    mode: str = 'binary',
    ignore_index: Optional[int] = None,
    threshold: float = None,
    num_classes: Optional[int] = None,
    reduction: str = "micro-imagewise",
):
    if metrics_measures is None:
        metrics_measures = ['IoU', 'F1', 'Pre', 'Rec', 'Acc']
    optional_metrics = ['IoU', 'F1', 'Pre', 'Rec', 'Acc']

    if not set(metrics_measures).issubset(set(optional_metrics)):
        raise ValueError(
            f"The element of metrics_list must be selected in {optional_metrics}"
        )

    tp, fp, fn, tn = get_stats(
        output,
        target,
        mode=mode,
        threshold=threshold,
        ignore_index=ignore_index,
        num_classes=num_classes,
    )

    IoU = iou_score(tp, fp, fn, tn, reduction=reduction)
    F1 = f1_score(tp, fp, fn, tn, reduction=reduction)
    Pre = precision(tp, fp, fn, tn, reduction=reduction)
    Rec = recall(tp, fp, fn, tn, reduction=reduction)
    Acc = accuracy(tp, fp, fn, tn, reduction=reduction)
    metrics_results = {'IoU': IoU, 'F1': F1, 'Pre': Pre, 'Rec': Rec, 'Acc': Acc}

    for index in optional_metrics:
        if index not in metrics_measures:
            metrics_results.pop(index)

    return metrics_results


if __name__ == "__main__":
    import torch

    metrics_list = ['IoU', 'F1']
    a = torch.randint(low=0, high=2, size=(3, 1, 256, 256), device='cpu')
    b = torch.randint(low=0, high=2, size=(3, 1, 256, 256), device='cpu')

    results = metrics(a, b, metrics_list)

    print(results)
