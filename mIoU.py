from PIL import Image
import numpy as np
import os
from utils import AverageMeter
import argparse


parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Validation")
parser.add_argument("--gt_dir", type=str, required=True)
parser.add_argument("--pred_dir", type=str, required=True)

args = parser.parse_args()

gt_dir = args.gt_dir
pred_dir = args.pred_dir

gt_files = os.listdir(gt_dir)
gt_files.sort()
pred_files = os.listdir(pred_dir)
pred_files.sort()


def accuracy(preds, label):
    label -= 1
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
#    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

intersection_meter = AverageMeter()
union_meter = AverageMeter()
acc_meter = AverageMeter()

for i, gt in enumerate(gt_files):
    pred = pred_files[i]
    pred_data = np.array(Image.open(pred_dir+pred))
    gt_data = np.array(Image.open(gt_dir+gt))
    intersection, union = intersectionAndUnion(pred_data, gt_data, 150)
    acc, pix = accuracy(pred_data, gt_data)
    acc_meter.update(acc, pix)
    intersection_meter.update(intersection)
    union_meter.update(union)

iou = intersection_meter.sum / (union_meter.sum + 1e-10)

print('Mean IoU: {:.4f}, Accuracy: {:.2f}%'.format(iou.mean(), acc_meter.average()*100))
