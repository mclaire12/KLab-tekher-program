"""Exact (short) answers for questions about contact details and the document owner.

Rule-based, fully explainable. Used before the QA model because phone numbers,
e-mails and links have exact formats that patterns find more reliably than a model.

Question intent  ->  pattern
  phone          ->  +250 788 123 456, 0788123456, (250) 788-123-456 ...
  email          ->  name@domain.tld
  website/link   ->  https://..., www..., linkedin.com/..., github.com/...
  owner's name   ->  first line of the document that looks like a person's name
  address        ->  "Address: ..." / "Location: ..." line, else the short
                     "City, Country" line in the document header
  other fields   ->  any "Label: value" line whose label matches the question
                     (e.g. "Nationality: Rwandan", "Date of birth: ...")

Which occurrence?
  * the question names someone that appears in the document (e.g. "phone of
    Twahirwa")  -> the occurrence closest to that name (this wins over pronouns);
  * otherwise ("my / his / the owner's phone") -> the FIRST occurrence: in CVs,
    letters and forms the owner's details come first, before referees/contacts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

INTENT_PATTERNS = {
    "phone": re.compile(r"\b(phone|telephone|tel|mobile|cell|contact number|whatsapp|"
                        r"t[ée]l[ée]phone|num[ée]ro|nimero|telefone)\b", re.I),
    "email": re.compile(r"\b(e-?mail|mail|courriel|imeyili|email address)\b", re.I),
    "url": re.compile(r"\b(website|web site|site|link|url|linkedin|github|portfolio|lien)\b", re.I),
    "name": re.compile(r"\b(name|names|nom|izina|amazina|who)\b", re.I),
    "address": re.compile(r"\b(address|adresse|aderesi|location|located|lives?|living|reside|"
                          r"residence|domicile)\b", re.I),
}
ADDRESS_LABELS = ("address", "adresse", "aderesi", "location", "residence", "domicile", "lives in")
LABEL_LINE = re.compile(r"^[ \t]*([^\W\d_][\w '’/.-]{1,30}?)[ \t]*[:：][ \t]*(\S[^\n]{0,150})$", re.M)
OWNER_WORDS = re.compile(
    r"\b(owner|author|applicant|candidate|holder|writer|student|document|cv|resume|"
    r"my|his|her|their|whose|propri[ée]taire|auteur|nyir\w*)\b", re.I)

PHONE = re.compile(r"\+?\(?\d[\d\s().-]{7,18}\d")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(?:https?://|www\.)\S+|\b(?:linkedin|github)\.com/\S+", re.I)
YEAR_RANGE = re.compile(r"^(19|20)\d\d\s*[-–]\s*(19|20)\d\d$")
NOT_A_NAME = re.compile(r"\b(curriculum|vitae|resume|résumé|cv|profile|contact|summary|education|"
                        r"experience|skills|objective|address|report|letter|page)\b", re.I)

# Words that never identify a specific person in the question
_QUESTION_STOP = {
    "what", "which", "where", "when", "who", "whom", "whose", "how", "the", "and", "for", "are", "is",
    "of", "this", "that", "document", "owner", "number", "phone", "telephone", "email", "mail",
    "address", "name", "names", "his", "her", "its", "their", "my", "give", "tell", "please",
    "about", "contact", "mobile", "website", "link", "author", "applicant", "candidate",
}


@dataclass
class ExactAnswer:
    answer: str
    method: str
    source: str  # text around the answer, shown to the user


def detect_intents(question: str) -> list[str]:
    return [name for name, pattern in INTENT_PATTERNS.items() if pattern.search(question)]


def _phones(text: str) -> list[re.Match]:
    found = []
    for m in PHONE.finditer(text):
        candidate = m.group().strip(" .-")
        digits = re.sub(r"\D", "", candidate)
        if 9 <= len(digits) <= 15 and not YEAR_RANGE.match(candidate):
            found.append(m)
    return found


def _target_positions(question: str, text: str) -> list[int]:
    """Positions in the text of words from the question that name someone/something specific."""
    lowered = text.lower()
    positions = []
    for word in re.findall(r"[^\W\d_]{3,}", question):
        w = word.lower()
        if w in _QUESTION_STOP:
            continue
        pos = lowered.find(w)
        if pos >= 0 and word[0].isupper():
            positions.append(pos)
    return positions


def _choose(matches: list[re.Match], targets: list[int]) -> re.Match | None:
    if not matches:
        return None
    if targets:
        return min(matches, key=lambda m: min(abs(m.start() - t) for t in targets))
    return matches[0]


def _context(text: str, start: int, end: int, width: int = 160) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    return ("…" if left else "") + " ".join(text[left:right].split()) + ("…" if right < len(text) else "")


def owner_name(text: str) -> tuple[str, int] | None:
    """First line among the first 15 that looks like a person's name."""
    offset = 0
    for i, line in enumerate(text.splitlines()):
        if i >= 15:
            break
        stripped = line.strip()
        words = stripped.split()
        if (2 <= len(words) <= 5 and not re.search(r"[\d@:/|]", stripped)
                and all(re.fullmatch(r"[^\W\d_][\w'’.-]*", w) for w in words)
                and all(w[0].isupper() for w in words) and not NOT_A_NAME.search(stripped)):
            name = stripped.title() if stripped.isupper() else stripped
            return name, text.find(line, offset)
        offset += len(line) + 1
    return None


def _labelled_lines(text: str) -> list[re.Match]:
    """'Label: value' lines whose value is real content (not another empty label)."""
    return [m for m in LABEL_LINE.finditer(text)
            if not re.match(r"^[^\W\d_][\w '’-]{0,25}[:：]", m.group(2))]


def labelled_value(question: str, text: str, labels: tuple[str, ...] | None = None) -> re.Match | None:
    """Value of a 'Label: value' line matching the given labels or the question's words."""
    if labels is None:
        words = [w.lower() for w in re.findall(r"[^\W\d_]{4,}", question)
                 if w.lower() not in _QUESTION_STOP]
        labels = tuple(w[:5] for w in words)
    for m in _labelled_lines(text):
        label = m.group(1).lower()
        if any(l in label for l in labels):
            return m
    return None


def owner_address(text: str) -> tuple[str, int] | None:
    """'Address:' style line, else a short 'Place, Place' line among the first 15 lines."""
    m = labelled_value("", text, ADDRESS_LABELS)
    if m:
        return m.group(2).strip(), m.start(2)
    offset = 0
    for i, line in enumerate(text.splitlines()):
        if i >= 15:
            break
        stripped = line.strip()
        if ("," in stripped and len(stripped.split()) <= 6 and not re.search(r"[\d@/|:]", stripped)
                and all(w[0].isupper() for w in stripped.replace(",", " ").split())):
            return stripped, text.find(line, offset)
        offset += len(line) + 1
    return None


def exact_answer(question: str, text: str) -> ExactAnswer | None:
    """Return a short exact answer, or None if the question is not a pattern question."""
    intents = detect_intents(question)
    pattern_intents = [i for i in intents if i in ("phone", "email", "url")]
    about_owner = bool(OWNER_WORDS.search(question))
    targets = _target_positions(question, text)

    parts, spans, methods = [], [], []

    if "name" in intents and (about_owner or pattern_intents) and not targets:
        found = owner_name(text)
        if found:
            name, pos = found
            parts.append(("Name", name))
            spans.append((pos, pos + len(name)))
            methods.append("owner name = first name-like line of the document")
        elif not pattern_intents:
            return None  # let the QA model try

    if "address" in intents and not targets:
        found = owner_address(text)
        if found:
            address, pos = found
            parts.append(("Address", address))
            spans.append((pos, pos + len(address)))
            methods.append("address = 'Address:' line or 'City, Country' line in the header")

    finders = {"phone": _phones, "email": lambda t: list(EMAIL.finditer(t)),
               "url": lambda t: list(URL.finditer(t))}
    labels = {"phone": "Phone", "email": "Email", "url": "Link"}
    for intent in pattern_intents:
        match = _choose(finders[intent](text), targets)
        if match is None:
            parts.append((labels[intent], "not found in the document"))
            continue
        parts.append((labels[intent], match.group().strip(" .,;")))
        spans.append((match.start(), match.end()))
        methods.append(f"{labels[intent].lower()} pattern"
                       + (" (closest to the person named in the question)" if targets
                          else " (first occurrence = document owner)"))

    if not parts:
        # Generic "Label: value" lookup, e.g. "What is her nationality?" -> "Nationality: Rwandan"
        m = labelled_value(question, text)
        if m is None:
            return None
        return ExactAnswer(m.group(2).strip(), f"labelled field '{m.group(1).strip()}:'",
                           _merged_context(text, [(m.start(), m.end())]))
    answer = parts[0][1] if len(parts) == 1 else " · ".join(f"{k}: {v}" for k, v in parts)
    return ExactAnswer(answer, "; ".join(methods) or "pattern search", _merged_context(text, spans))


def _merged_context(text: str, spans: list[tuple[int, int]], width: int = 160) -> str:
    """One excerpt per group of nearby answers, so the same text is not shown twice."""
    groups: list[list[int]] = []
    for start, end in sorted(spans):
        if groups and start - width <= groups[-1][1] + width:
            groups[-1][1] = max(groups[-1][1], end)
        else:
            groups.append([start, end])
    return "\n\n".join(_context(text, a, b, width) for a, b in groups)
