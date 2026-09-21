"""Teacher-forced NLL computation for CDP (Equation 1).

Computes D(A->B) = (nll_constrained - nll_standard) / nll_standard

This is the actual perplexity-based measurement, not a word-count
or explanation-length proxy. Requires direct model access for
token log-probabilities (not available through hosted APIs like Groq).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .templates import TEMPLATES


def compute_nll(
    model,
    tokenizer,
    prompt: str,
    reference: str,
    max_length: int = 512,
) -> float:
    """Compute mean negative log-likelihood of reference under prompt.
    
    Teacher-forces the reference text and measures how well the model
    predicts it given the prompt context.
    """
    # Encode prompt + reference
    full_text = f"{prompt}\n\n{reference}"
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    ).to(model.device)
    
    prompt_tokens = tokenizer(
        prompt + "\n\n",
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )
    prompt_len = prompt_tokens["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs["input_ids"][:, 1:].contiguous()
    
    # Compute per-token loss
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    
    # Only count tokens in the reference part (after prompt)
    if prompt_len > 0:
        ref_loss = loss[prompt_len - 1:]  # -1 because of shift
    else:
        ref_loss = loss
    
    return ref_loss.mean().item()


def compute_d_cdp(
    model,
    tokenizer,
    concept_a: str,
    concept_b: str,
    reference_explanation_b: str,
    templates: Optional[List[dict]] = None,
) -> float:
    """Compute D(A->B) averaged across paraphrase templates.
    
    D(A->B) = (nll_constrained - nll_standard) / nll_standard
    
    A high D means withholding A significantly degrades the model's
    ability to explain B => A is likely a prerequisite of B.
    
    Args:
        model: the loaded (possibly LoRA-merged) causal LM
        tokenizer: the corresponding tokenizer
        concept_a: the potential prerequisite concept name
        concept_b: the target concept name  
        reference_explanation_b: the fixed reference explanation to teacher-force
        templates: override templates (default: the 4 standard templates)
    
    Returns:
        D(A->B) averaged across templates
    """
    if templates is None:
        templates = TEMPLATES
    
    d_values = []
    
    for tmpl in templates:
        prompt_std = tmpl["standard"].format(
            concept_a=concept_a, concept_b=concept_b
        )
        prompt_cst = tmpl["constrained"].format(
            concept_a=concept_a, concept_b=concept_b
        )
        
        nll_standard = compute_nll(model, tokenizer, prompt_std, reference_explanation_b)
        nll_constrained = compute_nll(model, tokenizer, prompt_cst, reference_explanation_b)
        
        if nll_standard > 0:
            d = (nll_constrained - nll_standard) / nll_standard
        else:
            d = 0.0
        
        d_values.append(d)
    
    return sum(d_values) / len(d_values) if d_values else 0.0
