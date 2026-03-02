from __future__ import annotations

import logging
import re

from babeldoc.format.pdf.document_il import Document
from babeldoc.format.pdf.document_il import Page
from babeldoc.format.pdf.document_il import PdfParagraph

logger = logging.getLogger(__name__)


_EN_REFERENCE_TITLES = {
    "references",
    "reference",
    "bibliography",
    "works cited",
    "literature cited",
}
_ZH_REFERENCE_TITLES = {
    "参考文献",
    "参考资料",
    "參考文獻",
    "參考資料",
}


def _normalize_heading(text: str) -> str:
    text = text.strip().lower()
    # Remove leading numbering like "6. References" / "IV References"
    text = re.sub(r"^[\s\divx\.\-\(\)]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_compact(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).lower()


def _looks_like_reference_heading(text: str, layout_label: str | None) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) > 80:
        return False

    normalized = _normalize_heading(stripped)
    compact = _normalize_compact(normalized)

    if normalized in _EN_REFERENCE_TITLES or normalized in _ZH_REFERENCE_TITLES:
        return True
    if compact in {
        "references",
        "reference",
        "bibliography",
        "workscited",
        "literaturecited",
        "参考文献",
        "参考资料",
        "參考文獻",
        "參考資料",
    }:
        return True

    label = (layout_label or "").lower()
    if label in {"reference", "reference_content", "reference_hybrid"}:
        return True
    return False


class ReferenceSectionSkipper:
    """
    Detect and skip reference/bibliography section near document tail.
    """

    def __init__(self, docs: Document, enabled: bool):
        self.enabled = enabled
        self.anchor_page_number: int | None = None
        self.anchor_render_order: int | None = None
        self.tail_start_page_number: int | None = None

        if enabled:
            self._detect_anchor(docs)

    def _detect_anchor(self, docs: Document):
        pages = docs.page or []
        if not pages:
            return

        tail_start_index = max(0, int(len(pages) * 0.55))
        self.tail_start_page_number = pages[tail_start_index].page_number

        for page in pages[tail_start_index:]:
            for paragraph in page.pdf_paragraph:
                if _looks_like_reference_heading(
                    paragraph.unicode or "",
                    paragraph.layout_label,
                ):
                    self.anchor_page_number = page.page_number
                    self.anchor_render_order = paragraph.render_order
                    logger.info(
                        "Detected reference section anchor at page=%s render_order=%s text=%r",
                        self.anchor_page_number,
                        self.anchor_render_order,
                        (paragraph.unicode or "")[:80],
                    )
                    return

    def is_reference_paragraph(self, page: Page, paragraph: PdfParagraph) -> bool:
        if not self.enabled:
            return False

        label = (paragraph.layout_label or "").lower()
        if label in {"reference", "reference_content", "reference_hybrid"}:
            if self.tail_start_page_number is None or page.page_number is None:
                return True
            return page.page_number >= self.tail_start_page_number

        if self.anchor_page_number is None or page.page_number is None:
            return False

        if page.page_number > self.anchor_page_number:
            return True
        if page.page_number < self.anchor_page_number:
            return False

        if self.anchor_render_order is None or paragraph.render_order is None:
            return True
        return paragraph.render_order >= self.anchor_render_order
