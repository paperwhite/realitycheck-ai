def calculate_iou(box_a, box_b):
    """
    Calculate intersection over union for two bounding boxes.

    Boxes use [x1, y1, x2, y2].
    """

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )

    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def extract_detections(result):
    """Convert YOLO predictions to plain Python dictionaries."""

    detections = []

    for box in result.boxes:
        detections.append({
            "class_id": int(box.cls.item()),
            "confidence": float(box.conf.item()),
            "bbox": box.xyxy[0].tolist(),
        })

    return detections


def compare_detections(baseline, perturbed, iou_threshold=0.5):
    """
    Match detections by class and IoU.

    Uses greedy one-to-one matching.
    """

    candidates = []

    for i, original in enumerate(baseline):
        for j, modified in enumerate(perturbed):

            if original["class_id"] != modified["class_id"]:
                continue

            iou = calculate_iou(
                original["bbox"],
                modified["bbox"],
            )

            if iou >= iou_threshold:
                candidates.append((iou, i, j))

    # Match strongest overlaps first.
    candidates.sort(reverse=True)

    matched_original = set()
    matched_modified = set()
    matches = []

    for iou, i, j in candidates:

        if i in matched_original or j in matched_modified:
            continue

        matched_original.add(i)
        matched_modified.add(j)

        matches.append({
            "baseline_index": i,
            "perturbed_index": j,
            "class_id": baseline[i]["class_id"],
            "iou": iou,
            "baseline_confidence": baseline[i]["confidence"],
            "perturbed_confidence": perturbed[j]["confidence"],
            "confidence_delta": (
                perturbed[j]["confidence"]
                - baseline[i]["confidence"]
            ),
        })

    lost = [
        i for i in range(len(baseline))
        if i not in matched_original
    ]

    new = [
        j for j in range(len(perturbed))
        if j not in matched_modified
    ]

    persistence = (
        len(matches) / len(baseline)
        if baseline else None
    )

    mean_confidence_delta = (
        sum(m["confidence_delta"] for m in matches)
        / len(matches)
        if matches else None
    )

    return {
        "baseline_count": len(baseline),
        "perturbed_count": len(perturbed),
        "matched_count": len(matches),
        "lost_count": len(lost),
        "new_count": len(new),
        "detection_persistence": persistence,
        "mean_confidence_delta": mean_confidence_delta,
        "matches": matches,
        "lost_indices": lost,
        "new_indices": new,
    }
