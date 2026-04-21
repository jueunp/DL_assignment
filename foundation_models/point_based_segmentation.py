from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor

from utils.image_io import load_image, pick_device


def parse_point_string(raw_points: str) -> list[list[float]]:
    points: list[list[float]] = []
    for pair in raw_points.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_str, y_str = pair.split(",")
        points.append([float(x_str), float(y_str)])
    if not points:
        raise ValueError("Provide at least one point using the format 'x,y' or 'x,y;x,y'.")
    return points


def parse_label_string(raw_labels: str, count: int) -> list[int]:
    labels = [int(value.strip()) for value in raw_labels.split(",") if value.strip()]
    if len(labels) != count:
        raise ValueError("The number of point labels must match the number of points.")
    return labels


def overlay_mask(image_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.array([30, 144, 255], dtype=np.float32)
    alpha = 0.45
    output = image_array.astype(np.float32).copy()
    output[mask] = (1 - alpha) * output[mask] + alpha * color
    return output.astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point-based segmentation with SAM2.")
    parser.add_argument("--image", required=True, help="Local image path or URL.")
    parser.add_argument(
        "--points",
        required=True,
        help="Point prompts in the format 'x,y' or multiple points like 'x1,y1;x2,y2'.",
    )
    parser.add_argument(
        "--point-labels",
        default=None,
        help="Comma-separated point labels. Use 1 for foreground and 0 for background.",
    )
    parser.add_argument("--model-id", default="facebook/sam2-hiera-small")
    parser.add_argument("--output", default="docs/segmentation_result.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    image = load_image(args.image)
    image_array = np.array(image)

    input_points = parse_point_string(args.points)
    point_labels = (
        parse_label_string(args.point_labels, len(input_points))
        if args.point_labels
        else [1] * len(input_points)
    )

    processor = Sam2Processor.from_pretrained(args.model_id)
    model = Sam2Model.from_pretrained(args.model_id).to(device)

    inputs = processor(
        images=image,
        input_points=[[input_points]],
        input_labels=[[point_labels]],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    mask_scores = outputs.iou_scores[0, 0].detach().cpu().numpy()
    best_index = int(np.argmax(mask_scores))
    best_mask = masks[0][0][best_index].numpy().astype(bool)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    visual = overlay_mask(image_array, best_mask)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(visual)
    for (x, y), label in zip(input_points, point_labels):
        color = "lime" if label == 1 else "red"
        ax.scatter(x, y, c=color, s=120, edgecolors="white", linewidths=1.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"device: {device}")
    print(f"point prompts: {input_points}")
    print(f"point labels: {point_labels}")
    print(f"best mask score: {mask_scores[best_index]:.3f}")
    print(f"saved visualization to {output_path}")


if __name__ == "__main__":
    main()
