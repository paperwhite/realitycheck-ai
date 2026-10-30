from collections import defaultdict

from realitycheck.evaluate import calculate_iou


def evaluate_image(predictions, ground_truth, iou_threshold=0.5):
    """
    Evaluate predictions against COCO ground-truth boxes.

    Both inputs:
      [
        {
          "class_id": int,
          "bbox": [x1, y1, x2, y2],
          ...
        }
      ]

    Class IDs must use the same category namespace.
    """

    # Highest-confidence predictions get matching priority.
    predictions = sorted(
        predictions,
        key=lambda p: p["confidence"],
        reverse=True,
    )

    matched_gt = set()
    matches = []

    tp = 0
    fp = 0

    for pred in predictions:
        best_iou = 0.0
        best_gt_index = None

        for j, gt in enumerate(ground_truth):

            if j in matched_gt:
                continue

            if pred["class_id"] != gt["class_id"]:
                continue

            iou = calculate_iou(pred["bbox"], gt["bbox"])

            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_index = j

        if best_gt_index is not None:
            matched_gt.add(best_gt_index)
            tp += 1

            matches.append({
                "prediction": pred,
                "ground_truth_index": best_gt_index,
                "iou": best_iou,
            })
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "matches": matches,
    }
