"""
postprocessing.py — Deterministic output sanitizer.

Responsibilities
----------------
1. PT-PT language normalization: Llama 3.1 8B reliably drifts toward
   Brazilian Portuguese regardless of system-prompt instructions, because
   its training corpus is dominated by BR-PT. Regex substitution at the
   output level is deterministic and cannot be overridden by the model.

2. Guardrail enforcement: social-engineering refusals must terminate after
   the first guardrail sentence — prevents rule-4 ("Não tenho essa
   informação...") from bleeding through after a rule-8 response.
"""

import re


# ── Guardrail sentences that terminate the response ───────────────────────────

_GUARDRAILS = (
    "Só posso informar sobre as condições e promoções disponíveis nos nossos canais oficiais.",
)


# ── PT-BR → PT-PT substitution rules (applied in order) ──────────────────────

_SUBS: list[tuple[re.Pattern[str], str]] = [
    # Remove "você"/"Você" as a subject pronoun.
    # Portuguese verb conjugation already encodes person/number, so the
    # pronoun is grammatically redundant. Removing it produces idiomatic
    # PT-PT: "Você pode" → "Pode", "Se você quiser" → "Se quiser".
    (re.compile(r'\b[Vv]ocê\s+'), ''),

    # Vocabulary substitutions
    (re.compile(r'\bconosco\b', re.IGNORECASE), 'connosco'),
    (re.compile(r'\beconômic([ao])\b'), r'económic\1'),
    (re.compile(r'\bônibus\b', re.IGNORECASE), 'autocarro'),
    (re.compile(r'\bcelular\b', re.IGNORECASE), 'telemóvel'),

    # "nossa equipa" (without article) → "a nossa equipa"
    # Lookbehind prevents double-applying when already correct ("a nossa").
    (re.compile(r'(?<![Aa] )[Nn]ossa equipa'), 'a nossa equipa'),

    # Collapse whitespace artefacts left by pronoun removal
    (re.compile(r' {2,}'), ' '),
    (re.compile(r'^ ', re.MULTILINE), ''),
]


def _fix_sentence_case(text: str) -> str:
    """Re-capitalise sentence starts disrupted by pronoun removal."""
    # Very start of the text
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    # After ". " / "! " / "? " + optional whitespace
    text = re.sub(
        r'([.!?]\s+)([a-záàâãéêíóôõúüç])',
        lambda m: m.group(1) + m.group(2).upper(),
        text,
    )
    return text


def sanitize(text: str) -> str:
    """
    Clean LLM output before returning it to the user.

    Pipeline:
      1. Guardrail check — if output opens with a social-engineering refusal,
         return only that sentence (hard stop).
      2. PT-PT regex substitutions.
      3. Re-capitalise any sentence starts that pronoun removal lowercased.
    """
    stripped = text.strip()

    # 1. Guardrail enforcement
    for guardrail in _GUARDRAILS:
        if stripped.startswith(guardrail):
            return guardrail

    # 2. Substitutions
    for pattern, replacement in _SUBS:
        text = pattern.sub(replacement, text)

    # 3. Re-capitalise
    text = _fix_sentence_case(text)

    return text
