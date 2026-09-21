"""S0.5 — Batched LLM concept verification (A5).

Runs one batched Groq call per domain over the deduped candidate list,
asking which items are genuine named concepts vs. fragments/citations/
overly-generic phrases. This is a labelling task over a list the
pipeline already produced — it doesn't decide structure.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()


def _get_groq_client():
    """Get Groq client."""
    from groq import Groq
    return Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


def verify_concepts(
    concepts: List[Dict],
    domain_goal: str,
    batch_size: int = 25,
) -> Tuple[List[Dict], List[Dict]]:
    """Run batched LLM verification over a list of candidate concepts.
    
    Args:
        concepts: List of concept dicts with at least 'name' key.
        domain_goal: The goal concept / domain name for context.
        batch_size: Number of concepts per LLM call (20-30 range to stay within OTPM limits).
    
    Returns:
        Tuple of (accepted_concepts, rejected_concepts).
        Each rejected concept dict has an added 'reject_reason' key.
    """
    if not concepts:
        return [], []
    
    model = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
    client = _get_groq_client()
    
    accepted = []
    rejected = []
    
    # Process in batches
    for i in range(0, len(concepts), batch_size):
        batch = concepts[i:i + batch_size]
        batch_names = [c["name"] for c in batch]
        
        prompt = f"""You are reviewing a list of candidate concept names extracted from educational documents about "{domain_goal}".

For each item, decide whether it is a **genuine named concept or topic** that a student might need to learn, or whether it is a **fragment, citation artifact, overly generic phrase, or non-concept**.

Candidate list:
{chr(10).join(f'{j+1}. {name}' for j, name in enumerate(batch_names))}

Respond with a JSON array. For each item, provide:
{{"index": <1-based>, "keep": true/false, "reason": "brief explanation if rejected"}}

Rules:
- Keep genuine technical/academic concepts and named topics
- Reject sentence fragments, citation artifacts, author names, overly vague phrases like "important concept" or "basic idea"
- Reject anything that reads like a partial sentence rather than a concept name
- When in doubt, keep it"""
        
        models_to_try = [model]
        for cand in ["groq/compound-mini", "groq/compound"]:
            if cand not in models_to_try:
                models_to_try.append(cand)

        batch_succeeded = False
        for current_model in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=450,
                    timeout=20.0,
                )
                content = response.choices[0].message.content.strip()
                
                # Parse JSON from response
                if "```" in content:
                    m = re.search(r'```(?:json)?\s*(.+?)```', content, re.DOTALL)
                    content = m.group(1).strip() if m else content
                
                try:
                    verdicts = json.loads(content)
                except json.JSONDecodeError:
                    # If JSON parse fails, keep all concepts in this batch
                    accepted.extend(batch)
                    batch_succeeded = True
                    break
                
                if not isinstance(verdicts, list):
                    accepted.extend(batch)
                    batch_succeeded = True
                    break
                
                # Build index -> verdict map
                verdict_map = {}
                for v in verdicts:
                    if isinstance(v, dict) and "index" in v:
                        verdict_map[v["index"]] = v
                
                for j, concept in enumerate(batch):
                    v = verdict_map.get(j + 1, {})
                    if v.get("keep", True):
                        accepted.append(concept)
                    else:
                        concept["reject_reason"] = v.get("reason", "LLM flagged as non-concept")
                        rejected.append(concept)
                batch_succeeded = True
                break
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e):
                    continue
                print(f"LLM verification batch failed with {current_model}: {e}")
                break

        if not batch_succeeded:
            accepted.extend(batch)
    
    return accepted, rejected
