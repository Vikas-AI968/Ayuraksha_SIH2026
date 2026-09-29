"""Reusable legal-document structure extraction for normalized text."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_PAGE_RE = re.compile(r"^\s*---\s*Page\s+(\d+)\s*---\s*$", re.IGNORECASE)
_CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
_SCHEDULE_RE = re.compile(r"^\s*(?:THE\s+)?SCHEDULE(?:\s+([IVXLCDM]+|\d+))?\b", re.IGNORECASE)
_SECTION_WORD_RE = re.compile(r"^\s*(?:SECTION|SEC\.)\s+(\d+[A-Z]?)\b", re.IGNORECASE)
_SECTION_NUMBER_RE = re.compile(r"^\s*(\d+[A-Z]?)\.\s+([A-Z][^\n]{2,})")
_RULE_RE = re.compile(r"^\s*(?:RULE|REGULATION|ARTICLE|PARAGRAPH|CLAUSE)\s+(\d+[A-Z]?)\b", re.IGNORECASE)
_NUMBERED_RULE_RE = re.compile(r"^\s*(\d+[A-Z]?)\.\s+(.{2,})")
_SUBSECTION_RE = re.compile(r"^\s*\(?([a-z]|\d+[A-Z]?)\)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class LegalContext:
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    rule: Optional[str] = None
    sub_rule: Optional[str] = None
    regulation: Optional[str] = None
    sub_regulation: Optional[str] = None
    article: Optional[str] = None
    schedule: Optional[str] = None
    paragraph: Optional[str] = None
    page_number: Optional[int] = None

    def as_dict(self) -> dict[str, object]:
        return {
            "chapter": self.chapter,
            "section": self.section,
            "subsection": self.subsection,
            "rule": self.rule,
            "sub_rule": self.sub_rule,
            "regulation": self.regulation,
            "sub_regulation": self.sub_regulation,
            "article": self.article,
            "schedule": self.schedule,
            "paragraph": self.paragraph,
            "page_number": self.page_number,
        }


def _is_numbered_section(line: str) -> Optional[str]:
    match = _SECTION_NUMBER_RE.match(line)
    if not match:
        return None
    number, title = match.groups()
    lowered = title.lower()
    if any(token in lowered for token in ("omitted", "short title", "definitions", "inventions", "patent", "power", "application", "opposition", "anticipation")):
        return number
    return None


def classify_line(line: str, context: LegalContext) -> tuple[str, str] | None:
    if match := _CHAPTER_RE.match(line):
        return "chapter", match.group(1).upper()
    if match := _SCHEDULE_RE.match(line):
        return "schedule", match.group(1) or "I"
    if match := _SECTION_WORD_RE.match(line):
        return "section", match.group(1)
    if section := _is_numbered_section(line):
        return "section", section
    if match := _RULE_RE.match(line):
        label = line.strip().split(maxsplit=1)[0].lower()
        return label, match.group(1)
    if context.rule and (match := _SUBSECTION_RE.match(line)):
        return "sub_rule", match.group(1)
    if context.section and (match := _SUBSECTION_RE.match(line)):
        return "subsection", match.group(1).lower()
    return None


def iter_structured_segments(content: str) -> list[tuple[str, LegalContext]]:
    """Split at legal boundaries while carrying the active hierarchical context."""
    current = LegalContext()
    lines: list[str] = []
    segment_context = current
    segments: list[tuple[str, LegalContext]] = []

    def flush() -> None:
        nonlocal lines
        text = "\n".join(lines).strip()
        if text:
            segments.append((text, segment_context))
        lines = []

    for line in content.splitlines():
        page_match = _PAGE_RE.match(line)
        if page_match:
            flush()
            current = LegalContext(**{**current.as_dict(), "page_number": int(page_match.group(1))})
            segment_context = current
            lines.append(line)
            continue
        marker = classify_line(line, current)
        if marker:
            kind, value = marker
            boundary = kind in {"chapter", "section", "rule", "regulation", "article", "schedule", "paragraph", "clause", "subsection", "sub_rule"}
            if boundary:
                flush()
                segment_context = current
            values = current.as_dict()
            if kind == "chapter":
                values.update(chapter=value, section=None, subsection=None)
            elif kind == "schedule":
                values.update(schedule=value, section=None, subsection=None)
            elif kind == "section":
                values.update(section=value, subsection=None, rule=None, sub_rule=None, schedule=None)
            elif kind == "rule":
                values.update(rule=value, sub_rule=None)
            elif kind == "regulation":
                values.update(regulation=value, sub_regulation=None)
            elif kind == "article":
                values.update(article=value)
            elif kind == "paragraph":
                values.update(paragraph=value)
            elif kind == "subsection":
                values["subsection"] = value
            elif kind == "sub_rule":
                values["sub_rule"] = value
            current = LegalContext(**values)
            if boundary:
                segment_context = current
        lines.append(line)
    flush()
    return segments
