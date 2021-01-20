from Parameters import *


try:
    import numpy as np
except ImportError:
    print('\nnumpy is uninstalled')

def in_labels(max_label):
    if labels[max_label] in wanted_labels:
        return True
    return False


def get_boxes(model_output, w, h):
    '''

    :param model_output:
    :param w:
    :param h:
    :return: list of boxes
    and the indeses of the best boxes
    '''
    boxes_to_draw = []

    for layer in model_output:
        for pred_box in layer:
            labels_val = pred_box[5:]
            max_label_idx = np.argmax(labels_val)  # index of maximum value in the probability of the labels
            pred_conf = labels_val[max_label_idx]

            if pred_conf > confidence and in_labels(max_label_idx):
                box = pred_box[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes_to_draw.append([x, y, width, height, pred_conf])
    return boxes_to_draw, xnms(boxes_to_draw)





def nms(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")


def xnms(boxes):
    '''
    we have alot of boxes that have
    intersection over each other so
    we will use NMS in order to get
    the best boxes
    conf
    '''
    boxes = np.array(boxes)
    coords = boxes[..., :4]
    conf = boxes[:, 4]
    idxs = nms(coords, conf, threshold)
    return idxs
