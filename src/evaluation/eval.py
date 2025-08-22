import math, torch

def iter_blocks(input_ids, block_size):
    for i in range(0, input_ids.size(1), block_size):
        yield input_ids[:, i:i + block_size]
    
def ppl_on_texts(model, tok, texts, block_size=2048):
    device = next(model.parameters()).device
    total_nll, total_tokens = 0.0, 0

    for text in texts:
        text = (text or "").strip()
        if not text:
            continue

        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)

        for block in iter_blocks(input_ids, block_size):
            if block.size(1) < 2:
                continue

            with torch.no_grad():
                out = model(input_ids=block, labels=block)
                nll = out.loss.item() * (block.size(1) - 1)
                total_nll += nll
                total_tokens += (block.size(1) - 1)
    
    if total_tokens == 0:
        return float("inf")
    
    return math.exp(total_nll / total_tokens)

def distinct_n_ratio(tokens, n=1):
    if len(tokens) < n:
        return 0.0
    
    ngrams = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / max(1, len(ngrams))

def generate_one(model, tok, prompt, max_new_tokens=128, temperature=0.7, top_p=0.95, top_k=0):
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt").to(device)
    enc.pop("token_type_ids", None)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k > 0 else None,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        repetition_penalty=1.1,
    )

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)
    
    text = tok.decode(out[0], skip_special_tokens=True)
    return text

def count_new_tokens(input_ids, output_ids):
    """Return number of newly generated tokens (approx)."""
    return max(0, output_ids.shape[1] - input_ids.shape[1])