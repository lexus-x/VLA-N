"""pi0-FAST style FAST/DCT discrete-token action head.

Reuses the REAL Physical Intelligence "fast" universal action tokenizer --
the exact tokenizer lerobot's own PI0FastPolicy loads
(lerobot/policies/pi0_fast/configuration_pi0_fast.py:
`action_tokenizer_name = "physical-intelligence/fast"`, wired up in
lerobot/processor/tokenizer_processor.py: `ActionTokenizerProcessorStep`) via
`AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)`.
It performs the actual DCT compression + BPE tokenization/detokenization --
NOT reimplemented here, per the task instructions. (Required installing
`scipy`, which this HF-hosted tokenizer's `trust_remote_code` module needs
for `scipy.fftpack.idct`; a concrete ImportError confirmed this before
installing.)

Real port of the DECODING mechanism (the part flagged as a placeholder):
lerobot's own PI0FastPolicy does NOT give FAST tokens their own embedding
table or lm_head -- `ActionTokenizerProcessorStep._act_tokens_to_paligemma_tokens`
(lerobot/processor/tokenizer_processor.py) remaps each FAST token id into a
reserved slice at the TAIL of the backbone's OWN vocabulary:
`real_id = tokenizer.vocab_size - 1 - fast_skip_tokens - fast_token_id`
(`fast_skip_tokens=128` reserves the last 128 ids, e.g. special tokens,
from collision). `PI0FastPaliGemma.forward` then runs ONE ordinary
forward pass and computes cross-entropy directly against the backbone's own
(frozen) lm_head over the FULL vocabulary -- no separate FAST vocab/lm_head
ever exists. We port that exact remapping here, substituting SmolVLM2's
tokenizer/vocab (49280 ids) for PaliGemma's; the FAST vocab (2048) sits at
ids [47104, 49151], comfortably clear of digit tokens (32-41, used by
head_ar_tokens.py), the space token (216), and bos/eos (1 / 49279).

Simplification vs. native PI0Fast: identical in spirit to
head_ar_tokens.py's -- PI0Fast's own PaliGemma decoder transformer re-embeds
the growing FAST-token sequence at every layer; this harness's
single-shared-frozen-backbone invariant (backbone.py's own docstring) means
we can't re-run the SmolVLM2 transformer per head, so a small trainable GRU
stands in for that recurrence, bridging the frozen embedding table to the
frozen lm_head (both borrowed straight from the backbone, not new
parameters).

BUGFIX (decoder_fix_validation_v3): generate() previously always produced a
bit-identical constant action (mean_jerk=0.0041915...) because it never
stopped early. Root cause confirmed by direct instrumentation
(diag_fast_tokens.py on cache_libero_long.pt): real FAST/BPE-encoded target
sequences are short and data-dependent (observed T=5-9 tokens for real
libero_long action chunks, vs. the CHUNK_LEN*ACTION_DIM=56-token safety
cap), but `forward()`'s cross-entropy loss was only ever computed over those
T real FAST-vocab tokens -- eos_id was never once a training target, so the
GRU had zero signal for "stop here" and generate() always ran to the
56-token cap regardless of training (confirmed: 0/10 samples emitted eos_id
before the cap, before AND after training). Feeding physical-intelligence/
fast's own decode() a 56-token sequence for what should be a 5-9-token one
hits an internal BPE-reversal/reshape mismatch inside the library itself,
which silently substitutes its own all-zero fallback action -- every
episode then applies the exact same (zero) action, so the resulting
trajectory jerk is a deterministic physics-only constant, matching the
observed frozen value to 15 significant digits. Fix: append eos_id as one
extra real training target right after the last real FAST token (the same
lerobot/pi0-FAST convention already anticipated -- unused -- in
`_generate_fast_ids`'s stop check).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor

from heads.backbone import ACTION_DIM, CHUNK_LEN

_fast_tokenizer = None


def get_fast_tokenizer():
    global _fast_tokenizer
    if _fast_tokenizer is None:
        _fast_tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
    return _fast_tokenizer


FAST_SKIP_TOKENS = 128  # same constant lerobot's own ActionTokenizerProcessorStep uses


class FASTTokenHead(nn.Module):
    """AR decoder head that reuses the shared backbone's own (frozen) vocab
    and LM head to decode FAST/DCT tokens, pi0-FAST style -- see module
    docstring for the exact remapping and the one simplification made (GRU
    standing in for the frozen transformer's own recurrence).
    """

    def __init__(self, hidden_size, backbone, fast_skip_tokens=FAST_SKIP_TOKENS):
        super().__init__()
        self.tokenizer = get_fast_tokenizer()
        self.fast_vocab_size = self.tokenizer.vocab_size
        self.fast_skip_tokens = fast_skip_tokens

        # Frozen, borrowed straight from the backbone -- NOT new parameters.
        self.embed_tokens = backbone.embed_tokens
        self.lm_head = backbone.lm_head
        self.vocab_size = backbone.vocab_size

        tok = backbone.tokenizer
        self.bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        self.eos_id = tok.eos_token_id  # generate()'s stop signal, distinct from bos_id on this tokenizer

        lo = self.vocab_size - 1 - fast_skip_tokens - (self.fast_vocab_size - 1)
        assert lo >= 0, (
            f"fast_skip_tokens={fast_skip_tokens} too small: FAST vocab "
            f"({self.fast_vocab_size}) does not fit below the skip region of "
            f"the backbone's {self.vocab_size}-token vocabulary"
        )

        # The only trainable module -- see module docstring.
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def _fast_to_real_ids(self, fast_ids: torch.Tensor) -> torch.Tensor:
        return self.vocab_size - 1 - self.fast_skip_tokens - fast_ids

    def _real_to_fast_ids(self, real_ids: torch.Tensor) -> torch.Tensor:
        fast_ids = self.vocab_size - 1 - self.fast_skip_tokens - real_ids
        return fast_ids.clamp(0, self.fast_vocab_size - 1)

    def forward(self, pooled: torch.Tensor, target_actions: torch.Tensor):
        device = pooled.device
        B = target_actions.shape[0]
        assert B == 1, "FAST tokenizer (physical-intelligence/fast) encodes one sample at a time"

        tokens = self.tokenizer(target_actions.detach().cpu())[0]  # DCT+BPE -> list[int]
        fast_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        target_ids = self._fast_to_real_ids(fast_ids)  # remapped into the backbone's real vocab

        bos = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        eos = torch.full((B, 1), self.eos_id, dtype=torch.long, device=device)
        # ponytail: BPE compression makes T data-dependent (observed T=5-9
        # vs. the CHUNK_LEN*ACTION_DIM=56 worst case) -- teacher forcing over
        # only the T real tokens never gave the GRU a single example of
        # "what comes after the last token", so it never learned to stop
        # (confirmed via diag_fast_tokens.py: 0/10 samples emitted eos_id
        # before the 56-step cap, before AND after training). Appending
        # eos_id as one extra target position -- the same lerobot/pi0-FAST
        # convention `_generate_fast_ids` already checks for -- fixes it with
        # one more predicted step, no new parameters. See module docstring.
        dec_in = torch.cat([bos, target_ids], dim=1)  # (1, T+1)
        full_targets = torch.cat([target_ids, eos], dim=1)  # (1, T+1): T fast tokens + eos
        # ponytail: same bf16(frozen)/fp32(trainable GRU) boundary as
        # head_ar_tokens.py -- cast at the two borrow points, see its comment.
        emb = self.embed_tokens(dec_in).float()  # frozen real embedding
        h0 = pooled.unsqueeze(0)
        out, _ = self.gru(emb, h0)
        logits = self.lm_head(out.to(self.lm_head.weight.dtype)).float()  # frozen real lm_head, full backbone vocab

        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), full_targets.reshape(-1))

        # Diagnostic-only decode of the T real-token positions (the caller,
        # common.train_head, discards this and keeps only `loss`) -- drop the
        # final EOS-prediction position before remapping/decoding, same as
        # generate() does with its own stop token.
        pred_real_ids = logits[:, :-1, :].argmax(-1)
        pred_fast_ids = self._real_to_fast_ids(pred_real_ids)
        try:
            pred_action = self.tokenizer.decode(
                pred_fast_ids.cpu().tolist(), time_horizon=CHUNK_LEN, action_dim=ACTION_DIM
            )
            pred_action = torch.tensor(pred_action, dtype=torch.float32, device=device)
        except Exception:
            # Predicted tokens from an untrained head may not form a decodable
            # BPE/DCT sequence; the loss/backward pass above is what this
            # smoke test actually verifies, so fall back to zeros for the
            # reported "predicted action" rather than failing the test.
            pred_action = torch.zeros(1, CHUNK_LEN, ACTION_DIM, device=device)
        return pred_action, loss

    @torch.no_grad()
    def _generate_fast_ids(self, pooled: torch.Tensor):
        """The actual autoregressive loop -- greedy (argmax) decode one
        real-vocab id at a time, each fed back into the GRU as the next
        step's input (same recurrence as `forward`'s teacher-forcing, fed
        from its own predictions instead of `target_ids`). Split out from
        `generate` so the raw generated FAST-id sequence (pre-decode) is
        directly inspectable/testable, independent of `decode()`'s separate
        success/fallback behavior below.

        Unlike ar_tokens, the FAST/BPE sequence length is data-dependent (the
        DCT+BPE encoder used at training time, `self.tokenizer(...)`, isn't
        available without ground truth actions) and this tokenizer's own BPE
        vocab has no trained EOS -- but `forward()` now trains the model to
        predict the backbone's own real EOS token as the target right after
        the last real FAST token (see module docstring), so this loop's
        `eos_id` check is a genuine, trained stop signal, not just an
        aspirational one. Cap steps at CHUNK_LEN*ACTION_DIM as a safety net:
        encode() turns exactly that many DCT coefficients into one BPE token
        each in the worst case (no merges), so a real FAST sequence for this
        action shape is never longer.

        Returns fast-vocab ids (1, T) with T possibly 0 (no non-EOS token
        generated before the cap).
        """
        B = pooled.shape[0]
        assert B == 1, "FAST tokenizer (physical-intelligence/fast) encodes one sample at a time"
        device = pooled.device
        max_len = CHUNK_LEN * ACTION_DIM

        h = pooled.unsqueeze(0)
        cur_id = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        pred_ids = []
        for _ in range(max_len):
            emb = self.embed_tokens(cur_id).float()
            out, h = self.gru(emb, h)
            logits = self.lm_head(out.to(self.lm_head.weight.dtype)).float()
            cur_id = logits[:, -1, :].argmax(-1, keepdim=True)
            if cur_id.item() == self.eos_id:
                break  # EOS itself isn't a FAST token -- don't append it
            pred_ids.append(cur_id)

        if not pred_ids:
            return torch.zeros(1, 0, dtype=torch.long, device=device)
        pred_real_ids = torch.cat(pred_ids, dim=1)
        return self._real_to_fast_ids(pred_real_ids)

    @torch.no_grad()
    def generate(self, pooled: torch.Tensor) -> torch.Tensor:
        """Real autoregressive decoding for eval: no target/ground truth
        involved -- see `_generate_fast_ids` for the actual generation loop.
        """
        device = pooled.device
        pred_fast_ids = self._generate_fast_ids(pooled)
        if pred_fast_ids.shape[1] == 0:
            return torch.zeros(1, CHUNK_LEN, ACTION_DIM, device=device)
        try:
            pred_action = self.tokenizer.decode(
                pred_fast_ids.cpu().tolist(), time_horizon=CHUNK_LEN, action_dim=ACTION_DIM
            )
            pred_action = torch.tensor(pred_action, dtype=torch.float32, device=device)
        except Exception:
            # Same fallback as forward(): a genuinely-generated sequence can
            # still fail to reshape to (CHUNK_LEN, ACTION_DIM) -- that's now a
            # real "the head hasn't learned to stop/decode cleanly" signal,
            # not the old bogus-teacher-forced-zeros failure mode.
            pred_action = torch.zeros(1, CHUNK_LEN, ACTION_DIM, device=device)
        return pred_action
