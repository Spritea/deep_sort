# vim: expandtab:ts=4:sw=4
import numpy as np
import cv2


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """Suppress overlapping detections.

    Original code from [1]_ has been adapted to include confidence score.

    .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/

    Examples
    --------

        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]

    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.

    Returns
    -------
    List[int]
        Returns indices of detections that have survived non-maxima suppression.

    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]
    #我觉得加1是为了在算iou的时候，避免分母为0
    #否则当x2=x1或y2=y1的时候(可能是标注导致的)
    #1.这个code中的area[idxs[:last]]会变为0
    #2.且两个bbox无交集时，iou的分母会变成0
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        #iou=(w * h) / (area[idxs[:last]]+area[i]-w*h)
        #上面这个才是iou，下面这个overlap不是iou，只是交集占bbox面积的比例
        overlap = (w * h) / area[idxs[:last]]
        #np.where(overlap > max_bbox_overlap)得到的是tuple，包含ndarray
        #np.where(overlap > max_bbox_overlap)[0]是为了把这个ndarray取出来
        #得到的是一个ndarray，才能拿去做concatanate
        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick
