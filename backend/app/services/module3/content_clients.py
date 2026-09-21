"""Module 3 -- live external API clients (YouTube/GitHub/arXiv/docs).

These are the *runtime* HTTP clients used when ``config.module3.enabled_apis``
is non-empty and API keys are provided via environment variables. They are
kept separate from the aggregation/re-ranking logic so the offline environment
(which wires ``DeterministicContentClient`` instead) never triggers a network
call during tests. All clients share one contract: ``search(query, limit)``
returns ``List[ContentItem]``.

RATE LIMITS: callers MUST go through ``ResponseCache`` (see ``ContentAggregator``);
never instantiate these and query directly on a per-render basis.
"""

from __future__ import annotations

import os
from typing import List, Optional

from .content_aggregator import ContentAPIClient, ContentItem


class _HTTPClient(ContentAPIClient):
    """Base with a shared ``requests`` import (lazy, so offline is safe)."""

    def _get_json(self, url: str, params: Optional[dict] = None,
                  headers: Optional[dict] = None) -> dict:
        import requests

        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json()


class ArxivClient(_HTTPClient):
    kind = "arxiv"

    def search(self, query: str, limit: int = 5) -> List[ContentItem]:
        # Atom feed query; ArXiv returns Atom XML (not JSON).
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
        }
        return self._parse_atom(self._get(url, params))

    @staticmethod
    def _get(url: str, params: dict) -> str:
        import requests
        return requests.get(url, params=params, timeout=20).text

    def _parse_atom(self, xml: str) -> List[ContentItem]:
        import xml.etree.ElementTree as ET

        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml)
        items: List[ContentItem] = []
        for entry in root.findall("a:entry", ns):
            title = (entry.findtext("a:title", default="", namespaces=ns)
                     .strip().replace("\n", " "))
            link = entry.findtext("a:id", default="", namespaces=ns)
            summary = entry.findtext("a:summary", default="", namespaces=ns)
            items.append(ContentItem(kind=self.kind, url=link, title=title,
                                     description=summary.strip()))
        return items


class GitHubClient(_HTTPClient):
    kind = "github"

    def search(self, query: str, limit: int = 5) -> List[ContentItem]:
        headers = {}
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"token {token}"
        data = self._get_json(
            "https://api.github.com/search/repositories",
            params={"q": query, "per_page": limit}, headers=headers,
        )
        items = []
        for repo in data.get("items", []):
            items.append(ContentItem(
                kind=self.kind, url=repo.get("html_url", ""),
                title=repo.get("full_name", ""),
                description=(repo.get("description") or ""),
                meta={"stars": str(repo.get("stargazers_count", 0))},
            ))
        return items


class YouTubeClient(_HTTPClient):
    kind = "youtube"

    def search(self, query: str, limit: int = 5) -> List[ContentItem]:
        key = os.environ.get("YOUTUBE_API_KEY")
        if not key:
            raise RuntimeError("YOUTUBE_API_KEY not set")
        data = self._get_json(
            "https://www.googleapis.com/youtube/v3/search",
            params={"part": "snippet", "q": query, "maxResults": limit,
                    "type": "video", "key": key},
        )
        items = []
        for item in data.get("items", []):
            vid = item.get("id", {}).get("videoId", "")
            snip = item.get("snippet", {})
            items.append(ContentItem(
                kind=self.kind,
                url=f"https://www.youtube.com/watch?v={vid}",
                title=snip.get("title", ""),
                description=snip.get("description", ""),
                meta={"channel": snip.get("channelTitle", "")},
            ))
        return items


class DocsClient(_HTTPClient):
    kind = "docs"

    def search(self, query: str, limit: int = 5) -> List[ContentItem]:
        # Official docs often expose a JSON search endpoint; this client is a
        # configurable base that fetches from ``DOCS_SEARCH_URL`` if provided.
        url = os.environ.get("DOCS_SEARCH_URL")
        if not url:
            return []
        data = self._get_json(url, params={"q": query})
        items = []
        for doc in data.get("results", [])[:limit]:
            items.append(ContentItem(
                kind=self.kind, url=doc.get("url", ""),
                title=doc.get("title", ""),
                description=doc.get("snippet", ""),
            ))
        return items