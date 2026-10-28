"""Auditable hybrid OCR post-processing.

This module consumes *raw outputs* from Tesseract vie_best and PP-OCRv6.  It
does not pretend to run either heavyweight engine itself: engine invocation is
environment-specific and must export JSONL with the schema documented below.
Every selection and repair is retained in the result for review.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

SAFE_REPLACEMENTS = {
    "yeu cau bao ho": "yêu cầu bảo hộ",
    "Yeu cau bao ho": "Yêu cầu bảo hộ",
}
MARKER = re.compile(r"^\s*(\d{1,3})\s*[.)]\s*")

def normalize(text: str) -> str:
    # Preserve line boundaries: marker consensus is line-position based.
    return "\n".join(re.sub(r"[ \t\f\v]+", " ", line).strip() for line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip())

def choose_text(tesseract: str, ppocrv6: str, tesseract_confidence: float = 0.0, ppocrv6_confidence: float = 0.0) -> tuple[str, str, list[str]]:
    """Use Tesseract as primary; accept PP-OCRv6 only with a clear score lead.

    The threshold is intentionally explicit rather than a hidden confidence
    calibration.  Disagreement is retained for human review.
    """
    t, p = normalize(tesseract), normalize(ppocrv6)
    flags=[]
    if not t and p: return p, "ppocrv6_rescue", ["TESSERACT_EMPTY"]
    if not p: return t, "tesseract_vie_best", flags
    if t == p: return t, "agreed", flags
    flags.append("OCR_ENGINE_DISAGREEMENT")
    if ppocrv6_confidence >= tesseract_confidence + 0.15 and len(p) >= max(8, int(len(t)*0.8)):
        return p, "ppocrv6_score_lead", flags
    return t, "tesseract_vie_best", flags

def apply_safe_corrections(text: str) -> tuple[str, list[dict]]:
    out=text; changes=[]
    for old,new in SAFE_REPLACEMENTS.items():
        if old in out:
            out=out.replace(old,new); changes.append({"from":old,"to":new,"rule":"safe_vi_v4"})
    return out, changes

def repair_markers(text: str, variant_texts: list[str]) -> tuple[str, list[dict], list[str]]:
    """Repair only a missing sequential marker when two Tesseract variants agree.

    No textual content is invented.  Ambiguous sequences remain flagged.
    """
    lines=[x.strip() for x in re.split(r"\n+", text) if x.strip()]
    expected=1; repaired=[]; flags=[]; output=[]
    variant_lines=[[x.strip() for x in re.split(r"\n+", v) if x.strip()] for v in variant_texts if v]
    for pos,line in enumerate(lines):
        m=MARKER.match(line)
        number=int(m.group(1)) if m else None
        if number is None and expected > 1:
            support=[]
            for rows in variant_lines:
                if pos < len(rows):
                    vm=MARKER.match(rows[pos])
                    if vm and int(vm.group(1)) == expected: support.append(rows[pos])
            if len(support) >= 2:
                line=f"{expected}. {line}"; repaired.append({"line":pos+1,"from":None,"to":expected,"method":"consensus_marker_repair_v6_v7"}); number=expected
            else: flags.append("MARKER_MISSING_REVIEW")
        elif number is not None and number != expected:
            flags.append("MARKER_SEQUENCE_REVIEW")
        if number == expected: expected += 1
        output.append(line)
    return "\n".join(output), repaired, flags

def process(record: dict) -> dict:
    text, source, flags=choose_text(record.get("tesseract_vie_best", ""), record.get("ppocrv6", ""), float(record.get("tesseract_confidence",0)), float(record.get("ppocrv6_confidence",0)))
    corrected, changes=apply_safe_corrections(text)
    final, repairs, marker_flags=repair_markers(corrected, record.get("tesseract_variants", []))
    return {"text_vi":final,"ocr_state":"READY" if final else "OCR_REQUIRED","ocr_pipeline":{"primary":"Tesseract vie_best","support":"PP-OCRv6","hybrid_selector":source,"safe_correction":"v4","marker_repair":"v6_v7"},"ocr_raw":{"tesseract_vie_best":record.get("tesseract_vie_best",""),"ppocrv6":record.get("ppocrv6","")},"safe_corrections":changes,"marker_repairs":repairs,"ocr_flags":flags+marker_flags}
