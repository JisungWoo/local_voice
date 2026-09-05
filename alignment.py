"""Map recognized word spans to script spelling without inventing boundaries."""
from difflib import SequenceMatcher
import math
import re
from typing import TypedDict


class Word(TypedDict):
    text: str
    start: float
    end: float


class Timing(TypedDict):
    audio_duration: float
    transcript: str
    words: list[Word]


class Difference(TypedDict):
    expected: str
    heard: str
    start: float | None
    end: float | None


class AlignmentResult(TypedDict):
    all_words_match: bool
    alignment_usable: bool
    requires_review: bool
    transcript: str
    differences: list[Difference]
    timing: Timing | None
    reason: str | None


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace("’", "'"))


def map_alignment(script: str, observed: list[Word], duration: float) -> AlignmentResult:
    expected_words = script.split()
    expected = [token for word in expected_words for token in tokens(word)]
    heard = [(token, word) for word in observed for token in tokens(word["text"])]
    transcript = " ".join(word["text"].strip() for word in observed)
    result: AlignmentResult = {"all_words_match": False, "alignment_usable": False,
              "requires_review": True, "transcript": transcript,
              "differences": [], "timing": None, "reason": None}
    if not expected or not heard or any(not tokens(word) for word in expected_words):
        result["reason"] = "The transcription contains no alignable words or the script has standalone punctuation."
        return result
    matcher = SequenceMatcher(None, expected, [token for token, _ in heard], autojunk=False)
    for operation, a, b, c, d in matcher.get_opcodes():
        if operation == "equal":
            continue
        span = heard[c:d]
        result["differences"].append({
            "expected": " ".join(expected[a:b]), "heard": " ".join(token for token, _ in span),
            "start": span[0][1]["start"] if span else None,
            "end": span[-1][1]["end"] if span else None,
        })
        if b-a != d-c:
            result["reason"] = "The checker heard missing or extra words. Generate another take or import corrected timing."
    if result["reason"]:
        return result
    aligned: list[Word] = []
    offset, previous_end = 0, 0.0
    for word in expected_words:
        count = len(tokens(word))
        start, end = heard[offset][1]["start"], heard[offset+count-1][1]["end"]
        if (not math.isfinite(start) or not math.isfinite(end) or not math.isfinite(duration)
                or start < previous_end or end <= start or end > duration
                or round(end*1000) <= round(start*1000) or len(word) > 42):
            result["reason"] = "The checker could not measure usable boundaries for every word. Generate another take."
            return result
        aligned.append({"text": word, "start": start, "end": end})
        offset += count
        previous_end = end
    result.update({"all_words_match": not result["differences"], "alignment_usable": True,
                   "requires_review": bool(result["differences"]),
                   "timing": {"audio_duration": duration, "transcript": " ".join(expected_words), "words": aligned}})
    return result
