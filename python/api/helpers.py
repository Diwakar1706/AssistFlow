"""
Shared helper utilities used by the API routes.

Includes:
  - PDF / web-page element counting (partition stats)
  - HTML-to-readable-text extraction (no extra deps)
  - URL safety check (SSRF protection)
  - Chunk-to-source mapping for the frontend pipeline modal
"""
import ipaddress
from html.parser import HTMLParser
from urllib.parse import urlparse

from fastapi import HTTPException
from pypdf import PdfReader

try:
    import fitz
except ImportError:
    try:
        import pymupdf as fitz   # type: ignore
    except ImportError:
        fitz = None


# ── Partition stats ────────────────────────────────────────────────────

def partition_stats(pdf_path: str):
    """Count structural elements in a PDF (text sections, titles, images,
    tables) so the UI can display the Partitioning step details.
    Returns (stats_dict, pages_with_images_set)."""
    reader = PdfReader(pdf_path)
    text_sections, titles, images = 0, 0, 0
    pages_with_images = set()
    for pno, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        for block in text.split("\n\n"):
            if block.strip():
                text_sections += 1
        for line in text.splitlines():
            s = line.strip()
            if 3 < len(s) < 80 and not s.endswith(".") and len(s.split()) <= 10:
                titles += 1
        try:
            n_img = len(page.images)
        except Exception:
            n_img = 0
        if n_img:
            pages_with_images.add(pno)
            images += n_img

    tables = 0
    if fitz is not None:
        try:
            fdoc = fitz.open(pdf_path)
            for page in fdoc:
                try:
                    tables += len(page.find_tables().tables)
                except Exception:
                    pass
            fdoc.close()
        except Exception:
            pass

    total = text_sections + titles + images + tables
    stats = {
        "pages":          len(reader.pages),
        "text_sections":  text_sections,
        "titles_headers": titles,
        "images":         images,
        "tables":         tables,
        "other_elements": max(total - text_sections - titles - images, 0),
        "total_elements": total,
    }
    return stats, pages_with_images


def text_stats(text: str, num_segments: int) -> dict:
    """Partitioning-style stats for a web / plain-text source."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    titles = sum(
        1 for line in text.splitlines()
        if 3 < len(line.strip()) < 80
        and not line.strip().endswith(".")
        and len(line.split()) <= 10
    )
    return {
        "text_sections":  num_segments,
        "titles_headers": titles,
        "images":         0,
        "tables":         0,
        "total_elements": len(paragraphs) + titles,
    }


# ── HTML extractor ─────────────────────────────────────────────────────

class _ReadableText(HTMLParser):
    """Minimal readable-text extractor — skips scripts, styles, nav /
    header / footer, and turns block elements into paragraph breaks."""
    SKIP  = {"script", "style", "nav", "header", "footer", "aside",
              "form", "button", "svg", "noscript"}
    BLOCK = {"p", "div", "section", "article", "li", "tr",
              "h1", "h2", "h3", "h4", "h5", "h6",
              "br", "blockquote", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.title, self._skip = [], "", 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str):
    """Extract readable text and page title from raw HTML.
    Returns (title, text)."""
    parser = _ReadableText()
    parser.feed(html)
    raw   = "".join(parser.parts)
    lines = [" ".join(line.split()) for line in raw.splitlines()]
    text  = "\n\n".join(p for p in (line.strip() for line in lines) if p)
    return parser.title.strip(), text


# ── SSRF protection ────────────────────────────────────────────────────

def assert_public_url(url: str):
    """Allow only public http(s) URLs.  Blocks localhost and literal
    private / loopback IP addresses pasted into the URL box."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise HTTPException(400, "Please paste a valid http:// or https:// URL")
    host = p.hostname.lower()
    if host == "localhost" or host.endswith(".local"):
        raise HTTPException(400, "Local addresses are not allowed — paste a public web page.")
    try:
        ip = ipaddress.ip_address(host)   # only when the host IS an IP literal
    except ValueError:
        ip = None
    if ip is not None and (ip.is_private or ip.is_loopback
                           or ip.is_link_local or ip.is_reserved):
        raise HTTPException(400, "Local/private addresses are not allowed — paste a public web page.")


# ── Source-chunk mapping ───────────────────────────────────────────────

def sources_for(doc: dict, used_docs: list) -> list:
    """Map the chunks an answer was built from to their pipeline-modal entries.
    chunk_id matches the numbering in the UI's 'View Chunks' tab."""
    by_content = {}
    for c in doc.get("chunks", []):
        by_content.setdefault(c["content"], c["id"])
    out, seen = [], set()
    for d in used_docs:
        cid = d.metadata.get("chunk_id") or by_content.get(d.page_content)
        if cid is None:
            for c in doc.get("chunks", []):
                if c["page"] == d.metadata.get("page") and (
                        c["content"][:80] in d.page_content
                        or d.page_content[:80] in c["content"]):
                    cid = c["id"]
                    break
        if cid is not None and cid not in seen:
            seen.add(cid)
            out.append({"chunk_id": cid,
                        "page":     d.metadata.get("page"),
                        "preview":  d.page_content[:220]})
    return out
