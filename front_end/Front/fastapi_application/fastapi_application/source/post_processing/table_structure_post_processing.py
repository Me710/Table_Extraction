import torch
from torchvision.ops import nms
import numpy as np


def post_process_table_structure(table_structure, score_threshold=0.6, iou_threshold=0.2):
    """
    Post-processes table structure detection results by filtering based on score threshold
    and applying non-maximum suppression to remove overlapping boxes.

    Args:
        table_structure (dict): A dictionary containing 'scores', 'boxes', and 'labels'.
        score_threshold (float): The minimum score a bounding box must have to be kept.
        iou_threshold (float): The intersection-over-union (IOU) threshold for non-maximum suppression.

    Returns:
        dict: A filtered table structure dictionary with 'scores', 'boxes', and 'labels'.
    """
    scores = table_structure['scores']
    boxes = table_structure['boxes']
    labels = table_structure['labels']

    # Filter boxes based on the score threshold
    high_score_indices = torch.where(scores >= score_threshold)[0]
    filtered_boxes = boxes[high_score_indices]
    filtered_scores = scores[high_score_indices]
    filtered_labels = labels[high_score_indices]

    # Apply non-maximum suppression to remove overlapping boxes
    keep = nms(filtered_boxes, filtered_scores, iou_threshold=iou_threshold)

    # Final filtered boxes, scores, and labels
    filtered_boxes = filtered_boxes[keep]
    filtered_scores = filtered_scores[keep]
    filtered_labels = filtered_labels[keep]

    filtered_results = {
        'scores': filtered_scores,
        'boxes': filtered_boxes,
        'labels': filtered_labels
    }

    return filtered_results


