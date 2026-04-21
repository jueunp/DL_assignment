from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from utils.image_io import load_image, pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-vocabulary detection with Grounding DINO.")
    parser.add_argument("--image", required=True, help="Local image path or URL.")
    parser.add_argument(
        "--labels",
        required=True,
        help='Comma-separated labels, for example: "person, phone, clock"',
    )
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--output", default="docs/open_vocab_result.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    image = load_image(args.image)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        raise ValueError("At least one label must be provided.")

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device)

    inputs = processor(images=image, text=[labels], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.axis("off")

    print(f"device: {device}")
    print(f"labels: {labels}")
    print("detections:")
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(0, y1 - 5),
            f"{label}: {score:.2f}",
            color="white",
            fontsize=10,
            bbox={"facecolor": "red", "alpha": 0.7, "pad": 2},
        )
        print(f"- {label}: {score:.3f}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved visualization to {output_path}")


if __name__ == "__main__":
    main()
