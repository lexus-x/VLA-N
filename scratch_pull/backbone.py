"""Shared SmolVLM2-500M-Video-Instruct backbone wrapper for the matched-protocol
5-head comparison harness.

The whole point of "matched protocol" is that the ONLY variable across the 5
action heads is the head itself. So this file loads the <500M backbone once
and every head consumes the SAME frozen hidden states -- no head re-runs its
own bespoke VLM forward pass. The backbone is run under no_grad and its
output is detached, so each head's .backward() only ever touches that head's
own parameters.

Also holds the one shared action-shape convention used by every head file:
a dummy action chunk of shape (batch, CHUNK_LEN, ACTION_DIM) = (1, 8, 7)
(6-DoF + gripper, chunk length 8), matching the smoke_test.py placeholder's
7-dim action convention.

ar_tokens/fast_tokens additionally need the backbone's OWN vocabulary/lm_head
(not just the pooled vector) to decode through the real LM head instead of a
bolted-on vocab -- see `tokenizer`/`embed_tokens`/`lm_head`/`vocab_size`
below. These expose the SAME frozen weights `forward()` already uses
internally (just handles, still requires_grad=False), so exposing them
changes nothing for the 3 heads that only consume `pooled`.
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

# --- shared matched-protocol constants (documented once, used by every head) ---
ACTION_DIM = 7          # 6-DoF + gripper
CHUNK_LEN = 8            # action chunk length
INSTRUCTION = "pick up the red block"
IMG_SIZE = 224


class SmolVLM2Backbone:
    """Loads the fixed backbone once. forward() returns frozen (no_grad, detached)
    hidden states + mean-pooled hidden state that every head is conditioned on.
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Real vocabulary/LM-head, exposed (still frozen) for ar_tokens/fast_tokens
        # to decode through -- see module docstring.
        self.tokenizer = self.processor.tokenizer
        self.embed_tokens = self.model.get_input_embeddings()
        self.lm_head = self.model.get_output_embeddings()
        self.vocab_size = self.lm_head.out_features

    @torch.no_grad()
    def forward(self, image: Image.Image, instruction: str = INSTRUCTION):
        """Returns (hidden_states: (1, seq_len, H), pooled: (1, H)), both frozen."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": instruction}],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1].detach()
        pooled = hidden.mean(dim=1)
        return hidden, pooled


def dummy_batch():
    """One consistent dummy (image, instruction, action_chunk) sample shared by
    every head's smoke test. Random 224x224 image, fixed instruction, random
    action chunk of shape (1, CHUNK_LEN, ACTION_DIM).
    """
    img = Image.fromarray((np.random.rand(IMG_SIZE, IMG_SIZE, 3) * 255).astype("uint8"))
    action_chunk = torch.randn(1, CHUNK_LEN, ACTION_DIM) * 0.1
    return img, INSTRUCTION, action_chunk
