import cv2 as cv
import numpy as np

def extract_bounding_boxes(y, threshold=0.5):
    y = y[0].detach().cpu().numpy()
    #print(y.shape)
    label_img = np.where(y > threshold, 1, 0)
    label_img = np.asarray(label_img, dtype=np.uint8)
    contours, _ = cv.findContours(label_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    boxes = [cv.boundingRect(c) for c in contours]

    return boxes

def bb_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union given between two bboxes.

    :param bbox1: tuple of (x, y, w, h) of the first bbox.
    :param bbox2: tuple of (x, y, w, h) of the second bbox.
    :returns: IoU.
    """
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    if x1 >= x2 or y1 >= y2:
        intersection_area = 0
    else:
        intersection_area = (x2 - x1) * (y2 - y1)

    union_area = bbox1_area + bbox2_area - intersection_area
    return intersection_area / union_area

def bb_confusion_matrix(ytrue, ypred, iou_threshold=0.5, pred_threshold=0.5):
    """
    Calculate confusion matrix for the given groundtruth and predicted value.

    :returns: a tuple of (true positive, true negatives, false positive, false negative)
    """
    gt_bboxes = extract_bounding_boxes(ytrue, pred_threshold)
    pred_bboxes = extract_bounding_boxes(ypred, pred_threshold)

    if len(pred_bboxes) > 0:
        print("pred_bboxes: ", len(pred_bboxes))
        print(pred_bboxes)
        print(gt_bboxes)

    tp, fn = 0, 0
    for gt_box in gt_bboxes:
        iou = [bb_iou(gt_box, box) for box in pred_bboxes]

        # If there is no remaining bounding boxes,
        # then the current gt box is considered as false negative.
        if not len(iou):
            fn += 1
            continue

        max_idx = max(range(len(iou)), key=lambda i: iou[i])

        # If the value of the max index is larger than the threshold,
        # we can consider this to be a true positive.
        # If that is the case, then, we increment the true positive count,
        # and remove that bounding box out of the list.
        if iou[max_idx] >= iou_threshold:
            tp += 1
            pred_bboxes.pop(max_idx)
        else:
            # Otherwise, there is no predicted bounding box that
            # matches the current groundtruth box.
            # In that case, we will increment the false negative count.
            fn += 1

    # After looping through all groundtruth boxes,
    # the remaining predicted boxes will be counted as false positive.
    fp = len(pred_bboxes)

    # True negative is always 0,
    # because we don't have any box in which there is no object.
    return tp, 0, fp, fn


def bbiou(y_pred, y_true, iou_threshold=0.5, pred_threshold=0.5):
    # Bounding box IoU metric
    # Both y_pred and y_true are numpy arrays.
    N = y_true.shape[0]
    tp, fp, fn = 0, 0, 0
    for i in range(N):
        _tp, _, _fp, _fn = bb_confusion_matrix(y_true[i], y_pred[i], iou_threshold, pred_threshold)
        tp += _tp
        fp += _fp
        fn += _fn
    return {'TP': tp, 'FP': fp, 'FN': fn}


def get_local_loss(bbox1, bbox2):
    loss = ((bbox2[0] - bbox1[0]) ** 2) + ((bbox2[1] - bbox1[1]) ** 2) + ((bbox2[2] - bbox1[2]) ** 2) + ((bbox2[3] - bbox1[3]) ** 2)
    return loss

def spatial_losses(ytrue, ypred, iou_threshold=0.5, pred_threshold=0.5):
    gt_bboxes = extract_bounding_boxes(ytrue, pred_threshold)
    pred_bboxes = extract_bounding_boxes(ypred, pred_threshold)
    for gt_box in gt_bboxes:
        iou = [bb_iou(gt_box, box) for box in pred_bboxes]
        local_loss = [get_local_loss(gt_box, box) for box in pred_bboxes]
    if len(gt_bboxes) == 0 and len(pred_bboxes) > 0:
        empty_box = [0,0,0,0]
        iou = [bb_iou(empty_box, box) for box in pred_bboxes]
        local_loss = [get_local_loss(empty_box, box) for box in pred_bboxes]
    else:
        return sum([1]), sum([1])
    return sum(iou), sum(local_loss)
    
def loss_function(y_pred, y_true, iou_threshold=0.5, pred_threshold=0.5):
    N = y_true.shape[0]
    iou = 0
    local_loss = 0
    for i in range(N):
        iou, local_loss = spatial_losses(y_true[i], y_pred[i], iou_threshold, pred_threshold)
        iou += iou
        local_loss += local_loss
    loss = ((1 - iou) + 0.1 * local_loss)/N
    print(iou)
    print(loss)
    return loss
