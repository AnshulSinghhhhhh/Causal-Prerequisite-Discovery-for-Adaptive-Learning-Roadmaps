"""Job A — CDP probe on Kaggle T4 targeting Machine Learning low-margin edges.

Pulls low-margin Machine Learning candidate edges from Supabase, runs teacher-forced NLL
using Qwen2.5-3B-Instruct, writes results to parquet, updates edge_scores, and records cdp_runs.
"""

import os
import sys
import subprocess
from datetime import datetime, timezone
from typing import List, Optional

# Ensure supabase is installed
try:
    from supabase import create_client
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "supabase", "-q"], check=True)
    from supabase import create_client

import torch
import torch.nn.functional as F
import pandas as pd

# Kaggle secrets / fallbacks
try:
    from kaggle_secrets import UserSecretsClient
    sec = UserSecretsClient()
    HF_TOKEN = sec.get_secret("HF_TOKEN")
    SB_URL = sec.get_secret("SUPABASE_URL")
    SB_KEY = sec.get_secret("SUPABASE_SERVICE_KEY")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    SB_URL = os.environ.get("SUPABASE_URL", "")
    SB_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

if not SB_URL:
    SB_URL = os.environ.get("SUPABASE_URL", "")
if not SB_KEY:
    SB_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

# Target domain and model
TARGET_DOMAIN_SLUG = "machine_learning"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "main"

TEMPLATES = [
    {
        "standard": "Explain the concept: {concept_b}",
        "constrained": "Explain the concept {concept_b} without using or assuming any knowledge of {concept_a}.",
    },
    {
        "standard": "Describe what {concept_b} means and how it works.",
        "constrained": "Describe what {concept_b} means and how it works, without referencing or assuming familiarity with {concept_a}.",
    },
    {
        "standard": "Write a clear explanation of {concept_b} for a student.",
        "constrained": "Write a clear explanation of {concept_b} for a student who has never encountered {concept_a}.",
    },
    {
        "standard": "What is {concept_b}? Explain in detail.",
        "constrained": "What is {concept_b}? Explain in detail, deliberately avoiding any reference to {concept_a}.",
    },
]


def compute_nll(
    model,
    tokenizer,
    prompt: str,
    reference: str,
    max_length: int = 512,
) -> float:
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
    
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs["input_ids"][:, 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    
    if prompt_len > 0 and len(loss) >= prompt_len:
        ref_loss = loss[prompt_len - 1:]
    else:
        ref_loss = loss
    
    return ref_loss.mean().item()


def compute_d_cdp(
    model,
    tokenizer,
    concept_a: str,
    concept_b: str,
    reference_explanation_b: str,
) -> float:
    d_values = []
    for tmpl in TEMPLATES:
        prompt_std = tmpl["standard"].format(concept_a=concept_a, concept_b=concept_b)
        prompt_cst = tmpl["constrained"].format(concept_a=concept_a, concept_b=concept_b)
        
        nll_standard = compute_nll(model, tokenizer, prompt_std, reference_explanation_b)
        nll_constrained = compute_nll(model, tokenizer, prompt_cst, reference_explanation_b)
        
        if nll_standard > 0:
            d = (nll_constrained - nll_standard) / nll_standard
        else:
            d = 0.0
        d_values.append(d)
    
    return sum(d_values) / len(d_values) if d_values else 0.0


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("==================================================")
    print("LightGAP CDP Probe — Machine Learning Domain")
    print(f"Target Model: {MODEL_ID}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print("==================================================")

    print("Connecting to Supabase...")
    sb = create_client(SB_URL, SB_KEY)

    # 1. Fetch Machine Learning domain ID and concepts
    dom_res = sb.table("domains").select("id").eq("slug", TARGET_DOMAIN_SLUG).execute().data
    if not dom_res:
        print(f"Domain '{TARGET_DOMAIN_SLUG}' not found.")
        return
    dom_id = dom_res[0]["id"]

    concepts_res = sb.table("concepts").select("id, canonical_name, definition").eq("domain_id", dom_id).execute().data
    concept_map = {c["id"]: c for c in concepts_res}
    c_ids = list(concept_map.keys())

    # 2. Fetch candidate edges
    cands_res = sb.table("candidate_edges").select("id, src_id, dst_id").in_("src_id", c_ids[:100]).execute().data
    cand_map = {c["id"]: c for c in cands_res}
    cand_ids = list(cand_map.keys())

    # 3. Fetch lowest margin edge scores for these candidates
    work_scores = []
    for i in range(0, min(len(cand_ids), 300), 80):
        batch = cand_ids[i:i+80]
        s_res = sb.table("edge_scores").select(
            "candidate_edge_id, model_version_id, margin, s_lr_forward, s_lr_reverse"
        ).in_("candidate_edge_id", batch).is_("d_cdp", "null").execute().data
        work_scores.extend(s_res)

    work_scores.sort(key=lambda x: x["margin"])
    scores_res = work_scores[:15]

    if not scores_res:
        print("No unprobed low-margin edges found for Machine Learning. Pulling lowest-margin pairs...")
        for i in range(0, min(len(cand_ids), 300), 80):
            batch = cand_ids[i:i+80]
            s_res = sb.table("edge_scores").select(
                "candidate_edge_id, model_version_id, margin, s_lr_forward, s_lr_reverse"
            ).in_("candidate_edge_id", batch).execute().data
            work_scores.extend(s_res)
        work_scores.sort(key=lambda x: x["margin"])
        scores_res = work_scores[:15]

    print(f"Loaded {len(scores_res)} low-margin Machine Learning pairs to probe.")

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN or None)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN or None,
        revision=MODEL_REVISION,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    torch.manual_seed(42)
    model.eval()
    print("[SUCCESS] Model and tokenizer loaded onto GPU.")

    started_at = datetime.now(timezone.utc)
    results = []

    for item in scores_res:
        ce = cand_map.get(item["candidate_edge_id"], {})
        src_concept = concept_map.get(ce.get("src_id"), {})
        dst_concept = concept_map.get(ce.get("dst_id"), {})

        concept_a = src_concept.get("canonical_name", "")
        concept_b = dst_concept.get("canonical_name", "")
        reference = dst_concept.get("definition") or f"The fundamental concept of {concept_b}."

        if not concept_a or not concept_b:
            continue

        print(f"Probing {concept_a} -> {concept_b} (margin: {item.get('margin'):.6f})...")
        d_cdp = compute_d_cdp(model, tokenizer, concept_a, concept_b, reference)
        results.append({
            "candidate_edge_id": item["candidate_edge_id"],
            "model_version_id": item["model_version_id"],
            "concept_a": concept_a,
            "concept_b": concept_b,
            "margin": item.get("margin"),
            "s_lr_forward": item.get("s_lr_forward"),
            "s_lr_reverse": item.get("s_lr_reverse"),
            "d_cdp": float(round(d_cdp, 6)),
            "n_templates": len(TEMPLATES),
        })
        print(f"  Result: D({concept_a} -> {concept_b}) = {d_cdp:+.4f}")

    finished_at = datetime.now(timezone.utc)

    # Save to parquet
    df = pd.DataFrame(results)
    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "artifacts"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cdp_results_ml.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[SUCCESS] Saved {len(results)} ML CDP results to {out_path}")

    # Record cdp_runs in Supabase
    if results and scores_res:
        cdp_run_res = sb.table("cdp_runs").insert({
            "model_version_id": scores_res[0]["model_version_id"],
            "kaggle_kernel_ref": "anshulsingh45/lightgap-cdp-probe-qwen",
            "n_pairs_probed": len(results),
            "n_templates": len(TEMPLATES),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "notes": "Automated Qwen2.5-3B CDP probe for Machine Learning domain executed on Kaggle GPU.",
        }).execute()
        print(f"[SUCCESS] Inserted cdp_runs record: {cdp_run_res.data[0]['id']}")

        # Ingest directly into edge_scores
        print("Ingesting d_cdp values into edge_scores...")
        for r in results:
            sb.table("edge_scores").update({
                "d_cdp": r["d_cdp"]
            }).eq("candidate_edge_id", r["candidate_edge_id"]).execute()
        print("[SUCCESS] Ingested all ML CDP scores into edge_scores table.")


if __name__ == "__main__":
    main()
