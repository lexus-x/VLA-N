"""VLA-0 style AR integer-text-token action head.

Ports the uniform action-discretization scheme used by OpenVLA-style action
tokenizers (linspace bins + digitize) from
refs/openvla-oft/prismatic/vla/action_tokenizer.py `ActionTokenizer` -- VLA-0
(refs/vla-0/rv_train/models/qwen/model.py) uses this same idea: discretize
each continuous action dimension into bins and represent the bin index as
text, then decode through the model's own causal-LM head with cross-entropy
loss.

Real port of the DECODING mechanism (the part flagged as a placeholder):
VLA-0's `get_text_action` zero-... actually just space-joins bin indices as
plain integers and appends them as the assistant turn's text
(`format_data`); `forward()` then runs ONE ordinary causal-LM forward pass
over the whole prompt+action-text sequence and computes cross-entropy
against the tokenizer's OWN vocabulary/lm_head (labels for the
system/user portion masked to -100) -- restriction to digit tokens only
happens at *generation* time via `NumberSpaceOnlyProcessor`, not during
training. We port that: each bin id is written as a fixed-width 3-decimal-
digit string and represented with the BACKBONE'S OWN digit tokens (verified
on this tokenizer: '0'-'9' each map to exactly one real vocab id), and the
loss is an ordinary cross-entropy against the backbone's full real
vocabulary through its own (frozen) lm_head.

Simplification vs. native VLA-0: VLA-0's "decoder" IS the backbone's own
causal transformer, re-embedding the whole growing text sequence through
every layer at every position. This harness's core invariant
(backbone.py's own docstring) is that the backbone runs ONCE per sample,
frozen, shared by all 5 heads -- re-running the full transformer stack per
head here would break that matched-protocol contract. So we keep VLA-0's
actual mechanism for the two things this task called out -- decoding
through the backbone's REAL digit vocabulary and REAL (frozen) lm_head,
full-vocab cross-entropy -- and substitute a small trainable GRU for the
"growing-sequence self-attention" part: the GRU (not the frozen transformer)
produces the hidden state at each position that gets multiplied by the
frozen lm_head. Frozen, borrowed straight from the backbone: embed_tokens,
lm_head. Trainable: only the GRU bridging them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from heads.backbone import ACTION_DIM, CHUNK_LEN

N_BINS = 256
MIN_ACTION, MAX_ACTION = -1.0, 1.0
BIN_DIGITS = 3  # zero-padded decimal width covering bin ids [0, N_BINS-1] = [0, 255]


class ActionBinTokenizer:
    """Uniform per-dim binning, adapted from OpenVLA-OFT's `ActionTokenizer`
    (refs/openvla-oft/prismatic/vla/action_tokenizer.py): linspace bins,
    clip + digitize to encode, bin-center lookup to decode.
    """

    def __init__(self, n_bins=N_BINS, min_action=MIN_ACTION, max_action=MAX_ACTION):
        self.n_bins = n_bins
        self.bins = torch.linspace(min_action, max_action, n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    def encode(self, action: torch.Tensor) -> torch.Tensor:
        """Continuous action (...,) -> bin-center index in [0, n_bins-2]."""
        bins = self.bins.to(action.device)
        a = action.clamp(min=bins[0].item(), max=bins[-1].item())
        idx = torch.bucketize(a.contiguous(), bins)
        return idx.clamp(1, self.n_bins - 1) - 1

    def decode(self, idx: torch.Tensor) -> torch.Tensor:
        centers = self.bin_centers.to(idx.device)
        return centers[idx.clamp(0, centers.shape[0] - 1)]


class ARTokenHead(nn.Module):
    """AR decoder head that reuses the shared backbone's own digit-token
    vocabulary and (frozen) LM head to decode the discretized action-bin
    sequence, VLA-0 style -- see module docstring for the exact mechanism
    and the one simplification made (GRU standing in for the frozen
    transformer's own recurrence).
    """

    def __init__(self, hidden_size, backbone, n_bins=N_BINS):
        super().__init__()
        assert n_bins <= 10 ** BIN_DIGITS
        self.tok = ActionBinTokenizer(n_bins)
        self.seq_len = ACTION_DIM * CHUNK_LEN * BIN_DIGITS  # digit-token sequence length

        # Frozen, borrowed straight from the backbone -- NOT new parameters.
        self.embed_tokens = backbone.embed_tokens
        self.lm_head = backbone.lm_head
        self.vocab_size = backbone.vocab_size

        tok = backbone.tokenizer
        digit_ids = []
        for d in range(10):
            ids = tok.encode(str(d), add_special_tokens=False)
            assert len(ids) == 1, f"digit {d} is not a single token on this tokenizer: {ids}"
            digit_ids.append(ids[0])
        self.register_buffer("digit_to_id", torch.tensor(digit_ids, dtype=torch.long))
        id_to_digit = torch.full((self.vocab_size,), -1, dtype=torch.long)
        id_to_digit[self.digit_to_id] = torch.arange(10)
        self.register_buffer("id_to_digit", id_to_digit)
        self.bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

        # The only trainable module: bridges the frozen embedding to the frozen
        # lm_head, playing the role VLA-0's own transformer stack plays natively.
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def _bins_to_digit_ids(self, bins: torch.Tensor) -> torch.Tensor:
        """(B, ACTION_DIM*CHUNK_LEN) bin ids -> (B, seq_len) real digit token ids,
        each bin zero-padded to BIN_DIGITS decimal digits.
        """
        B = bins.shape[0]
        digits = torch.zeros(B, bins.shape[1], BIN_DIGITS, dtype=torch.long, device=bins.device)
        rem = bins.clone()
        for i in range(BIN_DIGITS - 1, -1, -1):
            digits[..., i] = rem % 10
            rem = rem // 10
        digits = digits.reshape(B, -1)  # (B, seq_len)
        return self.digit_to_id[digits]

    def _digit_ids_to_bins(self, ids: torch.Tensor) -> torch.Tensor:
        """Inverse of `_bins_to_digit_ids`. Unknown/non-digit ids fall back to
        digit 0 (an untrained head's argmax can land outside the digit subset --
        full-vocab CE only *prefers* digit tokens at those positions, same as
        native VLA-0, it doesn't hard-constrain them during training).
        """
        digit = self.id_to_digit[ids].clamp(min=0)
        digit = digit.reshape(digit.shape[0], -1, BIN_DIGITS)
        place_values = 10 ** torch.arange(BIN_DIGITS - 1, -1, -1, device=ids.device)
        return (digit * place_values).sum(-1)

    def forward(self, pooled: torch.Tensor, target_actions: torch.Tensor):
        B = pooled.shape[0]
        device = pooled.device
        target_bins = self.tok.encode(target_actions.reshape(B, -1))  # (B, ACTION_DIM*CHUNK_LEN)
        target_ids = self._bins_to_digit_ids(target_bins)  # (B, seq_len) real vocab ids

        bos = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        dec_in = torch.cat([bos, target_ids[:, :-1]], dim=1)  # teacher forcing
        # ponytail: backbone runs in bf16 (common.py's throughput optimization),
        # so its frozen embed_tokens/lm_head are bf16 while this head's own
        # trainable GRU is fp32 (matched-protocol convention, see common.py).
        # Cast at the two borrow points rather than reverting the shared bf16
        # backbone -- values pass through unchanged, only storage dtype does.
        emb = self.embed_tokens(dec_in).float()  # frozen real embedding, (B, seq_len, H)
        h0 = pooled.unsqueeze(0)  # (1, B, H) GRU initial state = shared backbone conditioning
        out, _ = self.gru(emb, h0)
        logits = self.lm_head(out.to(self.lm_head.weight.dtype)).float()  # frozen real lm_head, (B, seq_len, real_vocab_size)

        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), target_ids.reshape(-1))

        pred_ids = logits.argmax(-1)
        pred_bins = self._digit_ids_to_bins(pred_ids).clamp(0, self.tok.n_bins - 2)
        pred_action = self.tok.decode(pred_bins).reshape(B, CHUNK_LEN, ACTION_DIM)
        return pred_action, loss

    @torch.no_grad()
    def _generate_digit_ids(self, pooled: torch.Tensor) -> torch.Tensor:
        """The actual autoregressive loop -- greedy (argmax) decode one digit
        token at a time, each predicted id's frozen embedding feeding back
        into the GRU as the next step's input, same recurrence as `forward`'s
        teacher-forcing but fed from its own predictions instead of
        `target_ids`. Fixed-length stop: `seq_len` digit positions
        (ACTION_DIM*CHUNK_LEN*BIN_DIGITS), exactly the length the digit-
        encoding scheme always produces -- no separate stop token needed.
        Split out from `generate` so the raw generated real-vocab-id
        sequence (pre-decode) is directly inspectable/testable.
        """
        B = pooled.shape[0]
        device = pooled.device
        h = pooled.unsqueeze(0)  # (1, B, H) GRU initial state, same as forward()
        cur_id = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        pred_ids = []
        for _ in range(self.seq_len):
            emb = self.embed_tokens(cur_id).float()  # (B, 1, H)
            out, h = self.gru(emb, h)
            logits = self.lm_head(out.to(self.lm_head.weight.dtype)).float()  # (B, 1, vocab)
            cur_id = logits[:, -1, :].argmax(-1, keepdim=True)  # (B, 1), fed back next step
            pred_ids.append(cur_id)
        return torch.cat(pred_ids, dim=1)  # (B, seq_len)

    @torch.no_grad()
    def generate(self, pooled: torch.Tensor) -> torch.Tensor:
        """Real autoregressive decoding for eval: no target/ground truth
        involved -- see `_generate_digit_ids` for the actual generation loop.
        """
        B = pooled.shape[0]
        pred_ids = self._generate_digit_ids(pooled)
        pred_bins = self._digit_ids_to_bins(pred_ids).clamp(0, self.tok.n_bins - 2)
        return self.tok.decode(pred_bins).reshape(B, CHUNK_LEN, ACTION_DIM)
