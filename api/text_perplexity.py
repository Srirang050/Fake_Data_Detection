# api/text_perplexity.py
"""
Tiny helper to compute GPT-2 perplexity for a list of strings.
Loads tokenizer + GPT2LMHeadModel once at import-time (CPU).
"""

from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch
import torch.nn.functional as F

# load once
_tokenizer = None
_model = None
_device = torch.device("cpu")

def _init_model():
    global _tokenizer, _model, _device
    if _tokenizer is not None and _model is not None:
        return
    _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # set pad token if missing so batching works reliably
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _model = GPT2LMHeadModel.from_pretrained("gpt2")
    _model.to(_device)
    _model.eval()

def perplexity(texts, max_length: int = 512, batch_size: int = 8):
    """
    Compute per-sample perplexities for input `texts` (list[str]).
    Returns a list[float] of perplexity values.
    """
    _init_model()
    ppls = []
    # small batched computation
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = _tokenizer(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(_device)
        attention_mask = enc["attention_mask"].to(_device)
        with torch.no_grad():
            outputs = _model(input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits  # (B, T, V)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            log_probs = F.log_softmax(shift_logits, dim=-1)
            true_token_logprobs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            masked_token_logprobs = true_token_logprobs * shift_mask
            sum_logprob_per_sample = masked_token_logprobs.sum(dim=1)
            token_count_per_sample = shift_mask.sum(dim=1).clamp(min=1)
            neg_avg_ll = - (sum_logprob_per_sample / token_count_per_sample)
            per_sample_ppl = torch.exp(neg_avg_ll).cpu().numpy().tolist()
            ppls.extend(per_sample_ppl)
    return ppls
