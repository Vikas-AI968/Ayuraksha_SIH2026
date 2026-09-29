"""
Document Fetcher Module - Stage 1 of Ingestion.
Supports local files (JSON, TXT, HTML, PDF) and URLs.
"""
import logging
import os
import json
import time
from typing import Dict, Any, Union, Optional
from bs4 import BeautifulSoup
import httpx
import pypdf

from determinism import stable_document_id

logger = logging.getLogger("document_fetcher")


class DocumentFetcher:
    """Fetches raw contents from local files or URLs."""

    def fetch(self, source: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Accepts a file path, raw string content, dictionary payload, or URL reference.
        Returns a standardized raw document payload dictionary.
        """
        if isinstance(source, dict):
            return self._normalize_dict(source)

        if isinstance(source, str):
            if os.path.isfile(source):
                return self._fetch_file(source)
            elif source.startswith("http://") or source.startswith("https://"):
                return self._fetch_url(source)
            elif source.startswith("synthetic://"):
                raise ValueError(f"Synthetic URI string '{source}' must be passed with document content or as a path.")
            else:
                # Treat raw text string
                return {
                    "document_id": stable_document_id(source),
                    "title": "Raw Text Document",
                    "authority": "Unknown Authority",
                    "jurisdiction": "India",
                    "domain": "Patents",
                    "document_type": "Guidance",
                    "effective_date": "2026-01-01",
                    "source_url": "synthetic://raw-text",
                    "content": source,
                }

        raise ValueError(f"Unsupported source format: {type(source)}")

    def _fetch_file(self, file_path: str) -> Dict[str, Any]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return self._normalize_dict(data)

        elif ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            filename = os.path.basename(file_path)
            doc_id = os.path.splitext(filename)[0]
            return {
                "document_id": doc_id,
                "title": doc_id.replace("_", " ").replace("-", " ").title(),
                "authority": "Local File Store",
                "jurisdiction": "India",
                "domain": "Patents",
                "document_type": "Guidance",
                "effective_date": "2026-01-01",
                "source_url": f"file://{os.path.abspath(file_path)}",
                "content": content,
            }

        elif ext == ".html":
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text(separator="\n")
            title = soup.title.string if soup.title else os.path.basename(file_path)
            return {
                "document_id": os.path.splitext(os.path.basename(file_path))[0],
                "title": title,
                "authority": "HTML Import",
                "jurisdiction": "India",
                "domain": "Patents",
                "document_type": "Guidance",
                "effective_date": "2026-01-01",
                "source_url": f"file://{os.path.abspath(file_path)}",
                "content": text,
            }

        elif ext == ".pdf":
            reader = pypdf.PdfReader(file_path)
            text_pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                text_pages.append(f"--- Page {i+1} ---\n{page_text}")
            content = "\n\n".join(text_pages)
            return {
                "document_id": os.path.splitext(os.path.basename(file_path))[0],
                "title": os.path.basename(file_path),
                "authority": "PDF Import",
                "jurisdiction": "India",
                "domain": "Patents",
                "document_type": "Guidance",
                "effective_date": "2026-01-01",
                "source_url": f"file://{os.path.abspath(file_path)}",
                "content": content,
            }

        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return {
                "document_id": os.path.splitext(os.path.basename(file_path))[0],
                "title": os.path.basename(file_path),
                "authority": "Local File Store",
                "jurisdiction": "India",
                "domain": "Patents",
                "document_type": "Guidance",
                "effective_date": "2026-01-01",
                "source_url": f"file://{os.path.abspath(file_path)}",
                "content": content,
            }

    def _fetch_url(self, url: str, timeout: float = 45.0, max_retries: int = 2) -> Dict[str, Any]:
        """Fetches a document body from an http(s) URL with bounded timeouts
        and a small retry budget, so a stalled connection or slow server
        cannot hang ingestion forever."""
        headers = {"User-Agent": "IP-SAKTI-Sahayak/1.0 document-fetch"}
        fetch_timeout = httpx.Timeout(connect=10.0, read=timeout, write=10.0, pool=10.0)
        retryable = (
            httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout,
            httpx.PoolTimeout, httpx.ConnectError, httpx.ReadError,
            httpx.RemoteProtocolError,
        )
        attempt = 0
        while True:
            attempt += 1
            logger.info("Fetching %s (attempt %d/%d)", url, attempt, max_retries + 1)
            try:
                response = httpx.get(url, headers=headers, timeout=fetch_timeout, follow_redirects=True)
                response.raise_for_status()
                break
            except retryable as exc:
                if attempt > max_retries:
                    raise
                logger.warning("Fetch attempt %d for %s failed (%s); retrying", attempt, url, exc)
                time.sleep(2.0 * attempt)

        content_type = response.headers.get("content-type", "")
        raw_bytes = response.content
        is_pdf = "pdf" in content_type.lower() or raw_bytes[:4] == b"%PDF"
        if is_pdf:
            import io
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            text_pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                text_pages.append(f"--- Page {i+1} ---\n{page_text}")
            content = "\n\n".join(text_pages)
        else:
            soup = BeautifulSoup(raw_bytes, "html.parser")
            content = soup.get_text(separator="\n")

        return {
            "document_id": stable_document_id(url),
            "title": url.rsplit("/", 1)[-1] or url,
            "authority": "Unknown Authority",
            "jurisdiction": "India",
            "domain": "Patents",
            "document_type": "Guidance",
            "effective_date": "2026-01-01",
            "source_url": url,
            "content": content,
        }

    def _normalize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if not data.get("document_id"):
            title = data.get("title", "")
            data["document_id"] = stable_document_id(title)
        if not data.get("effective_date"):
            data["effective_date"] = "2026-01-01"
        if not data.get("source_url"):
            data["source_url"] = f"synthetic://{data['document_id']}"
        if not data.get("authority"):
            data["authority"] = "Unknown Authority"
        if not data.get("jurisdiction"):
            data["jurisdiction"] = "India"
        if not data.get("domain"):
            data["domain"] = "Patents"
        if not data.get("document_type"):
            data["document_type"] = "Guidance"
        return data
