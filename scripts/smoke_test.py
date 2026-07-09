import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import numpy as np

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"backbone params: {n_params / 1e6:.1f}M")

    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "pick up the red block"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img], return_tensors="pt").to(device)

    out = model(**inputs, output_hidden_states=True)
    last_hidden = out.hidden_states[-1]
    print("hidden state shape:", tuple(last_hidden.shape))

    # placeholder action head: mean-pool hidden states -> 7-dim action (6-DoF + gripper)
    action_head = torch.nn.Linear(last_hidden.shape[-1], 7).to(device)
    pooled = last_hidden.mean(dim=1)
    pred_action = action_head(pooled)
    loss = torch.nn.functional.mse_loss(pred_action, torch.zeros_like(pred_action))
    loss.backward()

    grad_norm = sum(p.grad.norm().item() for p in action_head.parameters() if p.grad is not None)
    print("action shape:", tuple(pred_action.shape), "loss:", loss.item(), "grad_norm:", grad_norm)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
