from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from utils.image_io import load_image, pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision-language chat demo.")
    parser.add_argument("--image", required=True, help="Local image path or URL.")
    parser.add_argument("--question", required=True, help="Question to ask about the image.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolVLM-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    image = load_image(args.image)
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    ).to(device)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    prompt_length = inputs["input_ids"].shape[-1]
    generated_text = processor.batch_decode(
        generated_ids[:, prompt_length:],
        skip_special_tokens=True,
    )[0].strip()

    print(f"device: {device}")
    print(f"question: {args.question}")
    print("answer:")
    print(generated_text)


if __name__ == "__main__":
    main()
