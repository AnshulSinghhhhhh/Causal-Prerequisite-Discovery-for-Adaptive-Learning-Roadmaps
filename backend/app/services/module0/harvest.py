"""S0.1 — Corpus harvester.

Pulls source documents from Wikipedia (category subtree + lead sections)
and the local AL-CPL dataset for overlapping domains.
"""

from __future__ import annotations

import os
import re
import glob
import html
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
import wikipediaapi

from ...db import table

load_dotenv()

DATA_ROOT = Path(__file__).resolve().parents[4] / "data"
AL_CPL_DIR = DATA_ROOT / "external" / "al-cpl" / "data"
AL_CPL_DOMAINS = ["data_mining", "geometry", "physics", "precalculus"]


def _strip_reference_sections(text: str) -> str:
    pattern = r'\n={2,}\s*(References|See also|External links|Notes|Further reading|Bibliography)\s*={2,}\n'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return text[:match.start()]
    return text


def _clean_html_text(html_content: str) -> str:
    """Extract clean textual prose from HTML page, stripping code/nav/styles."""
    # Drop script, style, nav, header, footer
    cleaned = re.sub(r'<(script|style|nav|header|footer)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    # Convert paragraph/break tags to newlines to preserve sequence structure
    cleaned = re.sub(r'<(p|br|div|li|h[1-6])[^>]*>', '\n', cleaned, flags=re.IGNORECASE)
    # Strip all remaining tags
    cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
    # Unescape HTML entities
    cleaned = html.unescape(cleaned)
    # Strip references
    cleaned = _strip_reference_sections(cleaned)
    # Collapse multiple blank lines and intra-line whitespace
    lines = [line.strip() for line in cleaned.splitlines()]
    cleaned_lines = []
    for line in lines:
        collapsed = re.sub(r'\s+', ' ', line).strip()
        if collapsed and len(collapsed) > 1:
            cleaned_lines.append(collapsed)
    text = "\n".join(cleaned_lines)
    return text



def harvest_wikipedia(
    goal_concept: str,
    domain_id: str,
    max_pages: int = 10,
    language: str = "en",
) -> List[Dict]:
    """Fetch Wikipedia pages related to the goal concept.
    
    Uses the Wikipedia API to get the page for the goal concept
    and pages linked from its summary/lead (breadth-first, up to max_pages).
    """
    wiki = wikipediaapi.Wikipedia(
        user_agent="LightGAP/0.1 (research project)",
        language=language,
    )
    documents: List[Dict] = []
    visited = set()
    queue = [goal_concept]
    
    while queue and len(documents) < max_pages:
        title = queue.pop(0)
        if title.lower() in visited:
            continue
        visited.add(title.lower())
        
        page = wiki.page(title)
        if not page.exists():
            continue
        
        # Store the page text
        doc = {
            "domain_id": domain_id,
            "source_type": "wikipedia",
            "source_url": page.fullurl,
            "title": page.title,
            "raw_text": _strip_reference_sections(page.text),
        }
        documents.append(doc)
        
        # Add linked pages to queue that appear in summary/lead
        summary_lower = (page.summary or "").lower()
        for link_title in page.links.keys():
            lt_lower = link_title.lower()
            if (
                lt_lower not in visited
                and len(link_title) > 3
                and not link_title.startswith(("Category:", "Template:", "Help:", "Portal:", "File:", "Wikipedia:"))
                and lt_lower in summary_lower
            ):
                queue.append(link_title)
                if len(queue) >= max_pages * 2:
                    break
    
    return documents


def harvest_al_cpl(
    goal_concept: str,
    domain_id: str,
) -> List[Dict]:
    """Load AL-CPL corpus text for domains overlapping the goal concept.
    
    Checks all four AL-CPL domains and loads text files from any
    that are thematically related.
    """
    goal_lower = goal_concept.lower()
    matching_domains = []
    if any(k in goal_lower for k in ("data", "mining", "machine learning", "ml", "ai", "learning", "neural", "classification", "cluster")):
        matching_domains.append("data_mining")
    if any(k in goal_lower for k in ("geom", "triangle", "polygon", "shape", "euclid")):
        matching_domains.append("geometry")
    if any(k in goal_lower for k in ("physic", "mechanic", "force", "energy", "quantum", "gravity")):
        matching_domains.append("physics")
    if any(k in goal_lower for k in ("calculus", "algebra", "trig", "function", "precalc")):
        matching_domains.append("precalculus")

    if not matching_domains:
        return []

    documents: List[Dict] = []
    for domain_name in matching_domains:
        domain_dir = AL_CPL_DIR / domain_name
        if not domain_dir.exists():
            domain_dir = DATA_ROOT / "external" / "pnpr-gcn" / "Graph_Split" / domain_name
        if not domain_dir.exists():
            continue
        
        # Load any narrative text files in the domain
        for fpath in sorted(domain_dir.rglob("*")):
            if fpath.is_file() and fpath.suffix in (".txt", ".tsv"):
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if len(text.strip()) < 50:
                    continue
                
                doc = {
                    "domain_id": domain_id,
                    "source_type": "al_cpl",
                    "source_url": str(fpath),
                    "title": f"AL-CPL/{domain_name}/{fpath.name}",
                    "raw_text": text,
                }
                documents.append(doc)
    
    return documents


def harvest_syllabus(
    goal_concept: str,
    domain_id: str,
    max_results: int = 5,
) -> List[Dict]:
    """Fetch real course syllabi and curricula for the goal concept.
    
    Performs a general web search for:
    "<goal_concept> syllabus OR course outline OR curriculum"
    and tags results as source_type = 'syllabus'.
    """
    query = f'"{goal_concept}" syllabus OR course outline OR curriculum'
    documents: List[Dict] = []
    
    try:
        data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        req = urllib.request.Request(
            "https://html.duckduckgo.com/html/",
            data=data,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            resp_html = resp.read().decode("utf-8", "ignore")
        
        # Matches for results
        matches = re.findall(
            r'<a[^>]+class="result__snippet[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            resp_html,
            re.DOTALL
        )
        if not matches:
            matches = re.findall(
                r'<a[^>]+class="result__url"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                resp_html,
                re.DOTALL
            )
        
        # Title matches
        title_matches = re.findall(
            r'<a[^>]+class="result__title[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            resp_html,
            re.DOTALL
        )
        url_to_title = {}
        for u, t in title_matches:
            clean_t = re.sub(r'<[^>]+>', '', t).strip()
            clean_t = html.unescape(clean_t)
            url_to_title[u.strip()] = clean_t
        
        seen_urls = set()
        for item in matches:
            if len(documents) >= max_results:
                break
            raw_url = item[0].strip()
            if "uddg=" in raw_url:
                m_uddg = re.search(r'uddg=([^&"\'\s]+)', raw_url)
                if m_uddg:
                    raw_url = urllib.parse.unquote(m_uddg.group(1))
            
            if not raw_url.startswith("http") or raw_url in seen_urls:
                continue
            if raw_url.lower().endswith((".pdf", ".doc", ".docx", ".ppt", ".pptx", ".zip")):
                continue
            
            seen_urls.add(raw_url)
            
            try:
                page_req = urllib.request.Request(
                    raw_url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (LightGAP Syllabus Harvester)"}
                )
                with urllib.request.urlopen(page_req, timeout=8) as page_resp:
                    page_html = page_resp.read().decode("utf-8", "ignore")
                
                clean_body = _clean_html_text(page_html)
                if len(clean_body) < 150:
                    continue
                
                doc_title = url_to_title.get(raw_url) or f"{goal_concept} Course Syllabus"
                documents.append({
                    "domain_id": domain_id,
                    "source_type": "syllabus",
                    "source_url": raw_url,
                    "title": doc_title,
                    "raw_text": clean_body,
                })
            except Exception:
                continue
                
    except Exception as e:
        print(f"[harvest_syllabus] Web search error: {e}")
    
    return documents


def persist_documents(documents: List[Dict]) -> List[Dict]:
    """Insert documents into the corpus_documents table and return with IDs."""
    if not documents:
        return []
    
    result = table("corpus_documents").insert(documents).execute()
    return result.data


def harvest(
    goal_concept: str,
    domain_id: str,
    max_wiki_pages: int = 10,
    max_syllabus_pages: int = 5,
) -> List[Dict]:
    """Run the full harvest pipeline for a goal concept.
    
    Combines Wikipedia, syllabus web searches, and AL-CPL sources,
    deduplicates by title, and persists to the database. Reuses existing rows if present.
    """
    existing = table("corpus_documents").select("*").eq("domain_id", domain_id).execute()
    existing_docs = existing.data or []
    has_syllabus = any(d.get("source_type") == "syllabus" for d in existing_docs)
    if existing_docs and has_syllabus:
        return existing_docs

    if existing_docs and not has_syllabus:
        syllabus_docs = harvest_syllabus(goal_concept, domain_id, max_results=max_syllabus_pages)
        if syllabus_docs:
            persisted = persist_documents(syllabus_docs)
            return existing_docs + persisted
        return existing_docs

    docs: List[Dict] = []
    
    # Wikipedia harvest
    wiki_docs = harvest_wikipedia(goal_concept, domain_id, max_pages=max_wiki_pages)
    docs.extend(wiki_docs)
    
    # Syllabus web search harvest
    syllabus_docs = harvest_syllabus(goal_concept, domain_id, max_results=max_syllabus_pages)
    docs.extend(syllabus_docs)
    
    # AL-CPL harvest
    al_cpl_docs = harvest_al_cpl(goal_concept, domain_id)
    docs.extend(al_cpl_docs)
    
    # Deduplicate by title
    seen_titles = set()
    unique_docs = []
    for doc in docs:
        title_key = (doc["title"] or "").lower().strip()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_docs.append(doc)
    
    # Persist
    persisted = persist_documents(unique_docs)
    return persisted
