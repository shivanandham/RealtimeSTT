"""
Vocabulary correction: rewrites misheard domain terms in transcripts.

Engine-agnostic. The corrections run on partial and final text after
transcription, so they behave the same for every engine and platform.
"""

import difflib
import json
import re


class Vocabulary:
    """
    Terms a transcript should contain, with the ways ASR tends to write them.

    terms is a list of terms ("SQL", "Kubernetes") or a dict of term to
    aliases ({"SQL": ["sequel"], "Node.js": ["node js", "no js"]}).
    Corrections run in this order, on whole words only, keeping the
    surrounding punctuation and using the term's own spelling and case:

    1. an alias phrase (case-insensitive) becomes the term;
    2. an acronym spelled out as letters ("s q l", "S.Q.L", "s-q-l") becomes
       the term;
    3. a word, or word group for multi-word terms, that is a close misspelling
       of a term (difflib ratio >= fuzzy_threshold, at least min_fuzzy_length
       letters) becomes the term. Acronyms are never fuzzy-matched; short
       words are too easy to confuse ("Jason" stays "Jason" unless listed as
       an alias of JSON).
    """

    _WORD = re.compile(r"[A-Za-z][A-Za-z0-9'.\-]*[A-Za-z0-9]|[A-Za-z]")

    def __init__(self, terms, fuzzy_threshold=0.8, min_fuzzy_length=5):
        """
        Builds the correction rules.
        """
        self.terms = {}
        if isinstance(terms, dict):
            for term, aliases in terms.items():
                self.terms[str(term)] = [str(alias) for alias in (aliases or [])]
        else:
            for term in terms or ():
                self.terms[str(term)] = []
        self.fuzzy_threshold = fuzzy_threshold
        self.min_fuzzy_length = min_fuzzy_length
        self._compile()

    @classmethod
    def coerce(cls, value):
        """
        Returns a Vocabulary from None, a Vocabulary, a list/dict of terms or a JSON file path.
        """
        if value is None or isinstance(value, Vocabulary):
            return value
        if isinstance(value, str):
            return cls.from_json(value)
        return cls(value)

    @classmethod
    def from_json(cls, path):
        """
        Loads terms from a JSON file: a list, a term->aliases dict, or
        {"terms": ..., "fuzzy_threshold": ..., "min_fuzzy_length": ...}.
        """
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "terms" in data:
            return cls(
                data["terms"],
                fuzzy_threshold=data.get("fuzzy_threshold", 0.8),
                min_fuzzy_length=data.get("min_fuzzy_length", 5),
            )
        return cls(data)

    def to_dict(self):
        """
        Returns the vocabulary as plain data (the from_json format).
        """
        return {
            "terms": {term: list(aliases) for term, aliases in self.terms.items()},
            "fuzzy_threshold": self.fuzzy_threshold,
            "min_fuzzy_length": self.min_fuzzy_length,
        }

    @staticmethod
    def _is_acronym(term):
        """
        Reports a term written as 2-6 capital letters, such as SQL or JSON.
        """
        letters = [c for c in term if c.isalpha()]
        return 2 <= len(letters) <= 6 and all(c.isupper() for c in letters) and letters == list(
            re.sub(r"[^A-Za-z]", "", term)
        )

    def _compile(self):
        """
        Prepares alias and acronym regexes and the fuzzy candidate list.
        """
        rules = []
        for term, aliases in self.terms.items():
            for alias in sorted(aliases, key=len, reverse=True):
                words = alias.split()
                if words:
                    pattern = r"\b" + r"[\s\-]+".join(re.escape(word) for word in words) + r"\b"
                    rules.append((re.compile(pattern, re.IGNORECASE), term))
            if self._is_acronym(term):
                letters = [re.escape(c) for c in term if c.isalpha()]
                # Letters with optional dots, spaces or hyphens between them; the last
                # letter keeps a following sentence period intact.
                pattern = r"\b" + r"\.?[\s.\-]*".join(letters) + r"(?![A-Za-z0-9])"
                rules.append((re.compile(pattern, re.IGNORECASE), term))
        self._rules = rules

        self._fuzzy = []
        for term in self.terms:
            if self._is_acronym(term):
                continue
            key = term.lower()
            if len(re.sub(r"[^a-z0-9]", "", key)) >= self.min_fuzzy_length:
                self._fuzzy.append((term, key, len(key.split())))

    def apply(self, text):
        """
        Returns the text with vocabulary corrections applied.
        """
        if not text or not self.terms:
            return text
        for regex, term in self._rules:
            text = regex.sub(term, text)
        if self._fuzzy:
            text = self._apply_fuzzy(text)
        return text

    def _apply_fuzzy(self, text):
        """
        Replaces close misspellings of the fuzzy-eligible terms.
        """
        matches = list(self._WORD.finditer(text))
        lowered = [match.group().lower() for match in matches]
        replacements = []
        i = 0
        while i < len(matches):
            best = None
            for term, key, count in self._fuzzy:
                if i + count > len(matches):
                    continue
                candidate = " ".join(lowered[i:i + count])
                if candidate == key:
                    best = (1.0, count, term)
                    break
                if len(re.sub(r"[^a-z0-9]", "", candidate)) < self.min_fuzzy_length:
                    continue
                matcher = difflib.SequenceMatcher(None, candidate, key)
                if matcher.quick_ratio() < self.fuzzy_threshold:
                    continue
                ratio = matcher.ratio()
                if ratio >= self.fuzzy_threshold and (best is None or ratio > best[0]):
                    best = (ratio, count, term)
            if best is None:
                i += 1
                continue
            _, count, term = best
            replacements.append((matches[i].start(), matches[i + count - 1].end(), term))
            i += count
        for start, end, term in reversed(replacements):
            text = text[:start] + term + text[end:]
        return text


__all__ = ["Vocabulary"]
