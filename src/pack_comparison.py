"""
pack_comparison.py — Authoritative pack data and query detection.

Financial arithmetic must not be delegated to a probabilistic model.
Pack vs. avulso comparisons have exactly one correct answer: the numbers
are hardcoded here and injected as a pre-computed block at the top of
the retrieval context so the LLM formats facts rather than calculates them.
"""

from __future__ import annotations

# ── Authoritative pack definitions ────────────────────────────────────────────
# Source of truth: data/precos.md § Pacotes e Promoções
# These values must stay in sync with the knowledge base.

_PACKS: dict[str, dict] = {
    "pack_noiva": {
        "name": "Pack Noiva",
        "price": 550,
        "individual_total": 635,
        "savings": 85,
        "components": [
            # (label, unit_price, qty, subtotal)
            ("3× Limpeza de Pele Profunda",         55, 3, 165),
            ("2× Peeling Químico Superficial",       75, 2, 150),
            ("1× Mesoterapia Facial",               120, 1, 120),
            ("1× Laser Rejuvenescimento Facial",    200, 1, 200),
        ],
    },
    "pack_anti_envelhecimento": {
        "name": "Pack Anti-Envelhecimento (3 meses)",
        "price": 780,
        "individual_total": 960,
        "savings": 180,
        "components": [],  # unit prices not published per component
    },
}


# ── Query detection ───────────────────────────────────────────────────────────

_DIRECT_NAMES = ("pack noiva", "pacote noiva", "pack anti-envelhecimento")
_COMPARISON_WORDS = (
    "avulso", "compensa", "poupo", "poupar", "economizo",
    "mais barato", "mais caro", "quanto poupo", "quanto custa",
)


def is_pack_query(query: str) -> bool:
    """Return True when the query involves a named pack or a pack/avulso comparison."""
    q = query.lower()
    if any(name in q for name in _DIRECT_NAMES):
        return True
    return "pack" in q and any(w in q for w in _COMPARISON_WORDS)


# ── Context block builder ─────────────────────────────────────────────────────

def build_pack_context(pack_key: str = "pack_noiva") -> str:
    """
    Return a pre-formatted, authoritative comparison block to prepend to
    the retrieval context.

    The LLM receives correct numbers from this block and is instructed by
    rule 6 not to recalculate them. This eliminates the class of errors
    where the model inverts a comparison or miscomputes a subtotal.
    """
    p = _PACKS[pack_key]
    name    = p["name"]
    price   = p["price"]
    ind     = p["individual_total"]
    saves   = p["savings"]

    lines = [
        f"╔══ DADOS CALCULADOS — {name} (não recalcules estes valores) ══╗",
        f"  Preço do pack:                               {price}€",
        "  Componentes e respectivos preços avulso:",
    ]

    for label, unit, qty, subtotal in p["components"]:
        lines.append(f"    {label}: {unit}€ × {qty} = {subtotal}€")

    lines += [
        f"  Total avulso de TODOS os {len(p['components'])} componentes:  {ind}€",
        f"  Poupança ao escolher o pack:                 {saves}€  ({ind}€ − {price}€ = {saves}€)",
        "  ─────────────────────────────────────────────────────────",
        f"  ATENÇÃO: a poupança de {saves}€ só se verifica se o/a cliente",
        f"  quiser TODOS os {len(p['components'])} componentes acima. Para um subconjunto",
        "  dos serviços, o custo avulso desse subconjunto pode ser",
        "  inferior ao preço total do pack — nesse caso o pack NÃO compensa.",
        "╚═════════════════════════════════════════════════════════════╝",
    ]

    return "\n".join(lines)
