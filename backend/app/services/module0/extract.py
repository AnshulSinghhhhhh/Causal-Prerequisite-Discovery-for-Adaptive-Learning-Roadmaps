"""S0.2 — Concept mention extraction.

Combines spaCy noun-phrase chunking with Groq LLM extraction
to find concept mentions in harvested documents.
"""

from __future__ import annotations

import os
import json
import re
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

def _get_groq_client():
    """Get Groq client singleton or instance."""
    from groq import Groq
    return Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


def _get_nlp():
    """Load spaCy model, with graceful fallback if unavailable."""
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


INVALID_EDGE_WORDS = {
    'a', 'an', 'the', 'this', 'that', 'these', 'those', 'there',
    'and', 'or', 'but', 'nor', 'for', 'yet', 'so',
    'of', 'in', 'to', 'with', 'from', 'at', 'by', 'on', 'as', 'into',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'considered', 'called', 'known', 'using', 'based', 'which', 'who', 'what',
    'e.g.', 'eg', 'i.e.', 'ie', 'hence', 'such', 'including', 'like', 'etc', 'etc.',
}


def is_wellformed(span) -> bool:
    """Check if a noun-phrase span is well-formed per A2 POS rules and NER."""
    text = span.text
    if '\n' in text or '\r' in text:
        return False
    if any(ch in text for ch in '[]{}<>=+'):
        return False
    if span.root.pos_ not in ('NOUN', 'PROPN'):
        return False
    # Reject entities that are persons, dates, times, or organizations
    if span.root.ent_type_ in ('PERSON', 'DATE', 'TIME', 'ORG'):
        return False
    first = span[0]
    if first.pos_ in ('CCONJ', 'SCONJ', 'ADP', 'PUNCT', 'SYM') or first.lemma_.lower() in ('a', 'an', 'the'):
        return False
    if first.text.lower() in INVALID_EDGE_WORDS:
        return False
    last = span[-1]
    if last.pos_ in ('CCONJ', 'SCONJ', 'ADP', 'DET', 'VERB', 'AUX', 'PART', 'PUNCT', 'SYM'):
        return False
    if last.lemma_.lower() in ('a', 'an', 'the', 'as', 'of', 'in', 'at', 'by', 'for', 'with', 'from'):
        return False
    return True



def extract_noun_phrases(text: str, nlp=None) -> List[str]:
    """Extract technical noun phrases from text using spaCy or regex fallback."""
    if nlp is None:
        nlp = _get_nlp()
    
    phrases = set()
    if nlp is not None:
        try:
            doc = nlp(text[:100_000])
            for chunk in doc.noun_chunks:
                if not is_wellformed(chunk):
                    continue
                clean = re.sub(r'\s+', ' ', chunk.text).strip()
                # Strip leading determiners and edge punctuation
                clean = re.sub(r'^(the|a|an)\s+', '', clean, flags=re.IGNORECASE).strip()
                clean = re.sub(r'^[\s\.,;:\-\–\—\(\)\[\]"\'\“\”]+|[\s\.,;:\-\–\—\(\)\[\]"\'\“\”]+$', '', clean).strip()
                if len(clean) <= 2 or '\n' in clean or any(ch in clean for ch in '[]{}<>=+'):
                    continue
                phrases.add(clean)
            return sorted(phrases)
        except Exception:
            pass


    # Pure Python / regex extraction for technical terms & compound noun phrases
    # Match capitalized phrases, hyphenated technical terms, and key domain phrases
    pattern = r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[a-z]+(?:\s+[a-z]+){1,2})\b'
    for match in re.finditer(pattern, text[:50_000]):
        p = match.group(0).strip()
        p = re.sub(r'^(the|a|an)\s+', '', p, flags=re.IGNORECASE).strip()
        p = re.sub(r'^[\s\.,;:\-\–\—\(\)\[\]"\'\“\”]+|[\s\.,;:\-\–\—\(\)\[\]"\'\“\”]+$', '', p).strip()
        words = p.lower().split()

        if len(words) < 1 or len(p) <= 3 or any(ch in p for ch in '[]{}<>=+'):
            continue
        # Reject if first or last word is a conjunction, preposition, or auxiliary verb
        if words[0] in INVALID_EDGE_WORDS or words[-1] in INVALID_EDGE_WORDS:
            continue
        phrases.add(p)
    return sorted(phrases)[:100]



def extract_concepts_llm(
    text: str,
    doc_title: str = "",
    model: str = "",
) -> List[Dict]:
    """Use Groq LLM to extract concept mentions from a text chunk.
    
    This is a labelling task (extracting mentions from visible text),
    not a structural decision — compliant with the governing principle.
    
    Returns list of {"name": str, "definition": str, "span": str}.
    """
    client = _get_groq_client()
    
    # Truncate text to fit context window
    text_chunk = text[:8000]
    
    prompt = f"""Analyze this text and list 5 to 10 key technical concepts it introduces or explains.
For each concept, provide:
1. The canonical name (a short, precise term)
2. A one-sentence definition based on the text
3. The exact span of text where it is first defined or introduced

Text (from "{doc_title}"):
{text_chunk}

Respond in JSON format:
[{{"name": "...", "definition": "...", "span": "..."}}]

Rules:
- Only extract concepts actually present in the text
- Limit to at most 10 concepts
- Keep definitions concise (one sentence)"""

    models_to_try = [model] if model else []
    for cand in [os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"), "groq/compound-mini", "groq/compound"]:
        if cand not in models_to_try:
            models_to_try.append(cand)

    for current_model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=current_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
                timeout=15.0,
            )
            content = response.choices[0].message.content.strip()
            
            # Parse JSON from response (handle markdown code blocks)
            if "```" in content:
                m = re.search(r'```(?:json)?\s*(.+?)```', content, re.DOTALL)
                content = m.group(1).strip() if m else content
            
            try:
                concepts = json.loads(content)
                if isinstance(concepts, list):
                    return concepts
                if isinstance(concepts, dict) and "concepts" in concepts:
                    return concepts["concepts"]
            except Exception:
                # Fallback regex parser for completed concept objects in truncated JSON
                recovered = []
                for block in re.finditer(r'\{[^{}]*?"name"\s*:\s*"([^"]+)"[^{}]*?"definition"\s*:\s*"([^"]+)"[^{}]*?\}', content):
                    recovered.append({
                        "name": block.group(1).strip(),
                        "definition": block.group(2).strip(),
                        "span": block.group(1).strip(),
                    })
                if recovered:
                    return recovered
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e):
                continue
            print(f"LLM extraction failed for '{doc_title}' with {current_model}: {e}")

    return []


def extract_from_documents(
    documents: List[Dict],
    batch_size: int = 5,
) -> List[Dict]:
    """Extract concept mentions from a list of harvested documents.
    
    Combines spaCy noun-phrase extraction (high recall) with
    LLM extraction (high precision, with definitions).
    
    Returns list of {"name", "definition", "source_doc_id", "source_span"}.
    """
    nlp = _get_nlp()
    all_concepts: List[Dict] = []
    
    for doc_idx, doc in enumerate(documents, 1):
        doc_id = doc.get("id", "")
        title = doc.get("title", "")
        text = doc.get("raw_text", "")
        print(f"    [{doc_idx}/{len(documents)}] Processing '{title}'...", flush=True)
        
        if not text:
            continue
        
        from .harvest import _strip_reference_sections
        text = _strip_reference_sections(text)
        if not text:
            continue
        
        # LLM extraction (primary — gives definitions)
        llm_concepts = extract_concepts_llm(text, doc_title=title)
        for c in llm_concepts:
            name = c.get("name", "").strip()
            name = re.sub(r'^(the|a|an)\s+', '', name, flags=re.IGNORECASE).strip()
            name = re.sub(r'^[\s\.,;:\-\–\—\(\)\[\]"\'\“\”]+|[\s\.,;:\-\–\—\(\)\[\]"\'\“\”]+$', '', name).strip()
            name = re.sub(r'\s+', ' ', name).strip()
            if len(name) < 3 or '\n' in name or any(ch in name for ch in '[]{}<>=+'):
                continue
            words = name.lower().split()

            if not words or len(name) < 3:
                continue
            if words[0] in INVALID_EDGE_WORDS or words[-1] in INVALID_EDGE_WORDS:
                continue

            if nlp is not None:
                try:
                    ndoc = nlp(name)
                    if len(ndoc) > 0 and not is_wellformed(ndoc):
                        continue
                except Exception:
                    pass
            all_concepts.append({
                "name": name,
                "definition": c.get("definition", "").strip(),
                "source_doc_id": doc_id,
                "source_span": c.get("span", ""),
            })

        
        # spaCy extraction (supplementary — catches terms LLM might miss)
        noun_phrases = extract_noun_phrases(text, nlp=nlp)
        existing_names = {c["name"].lower() for c in all_concepts}
        for phrase in noun_phrases:
            if phrase.lower() not in existing_names:
                all_concepts.append({
                    "name": phrase,
                    "definition": "",  # no definition from spaCy
                    "source_doc_id": doc_id,
                    "source_span": "",
                })
    
    return all_concepts
