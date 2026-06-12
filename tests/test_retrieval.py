"""
tests/test_retrieval.py — Retrieval quality test suite for Clínica Aurora.

Run from the project root:
    python tests/test_retrieval.py
    python tests/test_retrieval.py --interactive    # add a REPL at the end
    python tests/test_retrieval.py --k 6            # change number of chunks
"""

import sys
import argparse
import textwrap
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import load_vectorstore, retrieve_with_scores

# ── Colour helpers (degrades gracefully on Windows without ANSI) ──────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _score_colour(score: float) -> str:
    if score >= 0.70:
        return GREEN
    if score >= 0.45:
        return YELLOW
    return RED

# ── Score interpretation guide ────────────────────────────────────────────────
SCORE_GUIDE = """
Score guide:
  >= 0.70   Good match  -- chunk is directly relevant to the query
  0.45-0.69 Partial     -- related content, may not answer precisely
  < 0.45    Weak match  -- low signal; model may say "not found"
"""

# ── Test battery ──────────────────────────────────────────────────────────────
# Format: (query, [expected_source_files], label)
# expected_sources: ANY of these files appearing in top-k counts as a PASS.
TEST_CASES = [
    (
        "Qual é o horário de funcionamento da clínica?",
        ["contactos.md", "faq.md"],
        "Horário",
    ),
    (
        "Quanto custa uma sessão de toxina botulínica?",
        ["tratamentos.md"],
        "Preço botox",
    ),
    (
        "Qual é a política de cancelamento se não puder comparecer?",
        ["politicas.md", "faq.md"],
        "Cancelamento",
    ),
    (
        "Que tratamentos têm para celulite e gordura localizada?",
        ["tratamentos.md"],
        "Tratamentos corporais",
    ),
    (
        "Preciso de consulta prévia para fazer preenchimento com ácido hialurónico?",
        ["politicas.md", "faq.md"],
        "Consulta prévia",
    ),
    (
        "Onde fica a clínica e como chegar de metro?",
        ["contactos.md"],
        "Localização",
    ),
    (
        "Posso fazer depilação a laser se tiver a pele morena?",
        ["tratamentos.md", "faq.md"],
        "Laser pele escura",
    ),
    (
        "Posso fazer tratamentos se estiver grávida?",
        ["faq.md"],
        "Gravidez",
    ),
    (
        "Aceitam MB Way e cartão de crédito?",
        ["politicas.md", "faq.md"],
        "Métodos de pagamento",
    ),
    (
        "Quantas sessões de depilação a laser são necessárias para as pernas?",
        ["tratamentos.md", "faq.md"],
        "N.º sessões laser",
    ),
    (
        "Qual é o preço do pack para noivas?",
        ["tratamentos.md"],
        "Pack noiva",
    ),
    # Edge case: out-of-scope query — expect LOW scores (no good match in KB)
    (
        "Fazem cirurgia plástica ou lipoaspiração?",
        [],   # no expected source — we want low scores here
        "FORA DO ESCOPO — cirurgia",
    ),
]

# ── Core test runner ──────────────────────────────────────────────────────────

def run_tests(k: int = 4) -> dict:
    print(f"\n{BOLD}{'=' * 62}{RESET}")
    print(f"{BOLD}  Clínica Aurora — Retrieval Quality Test Suite{RESET}")
    print(f"{BOLD}{'=' * 62}{RESET}")
    print(f"  k = {k} chunks per query | {len(TEST_CASES)} test cases")
    print(SCORE_GUIDE)

    vs = load_vectorstore()
    passed = failed = 0
    low_score_warnings = []

    for idx, (query, expected_sources, label) in enumerate(TEST_CASES, 1):
        is_oos = not expected_sources  # out-of-scope test
        print(f"\n{BOLD}[{idx:02d}/{len(TEST_CASES)}] {label}{RESET}")
        print(f"  Query: \"{query}\"")
        if expected_sources:
            print(f"  Expected source(s): {', '.join(expected_sources)}")
        else:
            print(f"  {YELLOW}Expected: low scores (out-of-scope){RESET}")
        print(f"  {'-' * 58}")

        results = retrieve_with_scores(query, vs, k=k)
        retrieved_sources = {doc.metadata.get("source", "") for doc, _ in results}
        top_score = results[0][1] if results else 0.0

        for rank, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "?")
            snippet = doc.page_content.replace("\n", " ").strip()
            snippet = textwrap.shorten(snippet, width=80, placeholder="…")
            colour = _score_colour(score)
            print(f"  #{rank} {colour}[{score:.2f}]{RESET} {CYAN}{source}{RESET}")
            print(f"     {snippet}")

        # Verdict
        if is_oos:
            if top_score < 0.50:
                verdict = f"{GREEN}✓ PASS{RESET} — top score {top_score:.2f} correctly low (out-of-scope)"
                passed += 1
            else:
                verdict = (f"{RED}✗ FAIL{RESET} — top score {top_score:.2f} is HIGH for an "
                           f"out-of-scope query (possible false match)")
                failed += 1
        else:
            hit = bool(retrieved_sources & set(expected_sources))
            if hit:
                verdict = f"{GREEN}✓ PASS{RESET} — expected source found in top-{k}"
                passed += 1
            else:
                verdict = (f"{RED}✗ FAIL{RESET} — expected {expected_sources} "
                           f"not in retrieved {sorted(retrieved_sources)}")
                failed += 1
            if top_score < 0.45:
                low_score_warnings.append((label, top_score))

        print(f"  {verdict}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = passed + failed
    pct = passed / total * 100 if total else 0
    print(f"\n{BOLD}{'=' * 62}{RESET}")
    print(f"{BOLD}  Results: {passed}/{total} passed ({pct:.0f}%){RESET}")

    if low_score_warnings:
        print(f"\n{YELLOW}  Low-score warnings (top chunk < 0.45):{RESET}")
        for lbl, sc in low_score_warnings:
            print(f"    • {lbl}: {sc:.2f}")
        print(f"\n  {YELLOW}Possible causes:{RESET}")
        print("    - Query language (EN model may underrank PT slang)")
        print("    - That topic isn't covered in the knowledge base")
        print("    - Chunk boundaries split the relevant passage")
        print("    - Fix: try smaller CHUNK_SIZE, or switch to multilingual model")

    if failed == 0:
        print(f"\n  {GREEN}All tests passed. Retrieval looks healthy.{RESET}")
    else:
        print(f"\n  {RED}{failed} test(s) failed — review FAIL lines above.{RESET}")

    print(f"{BOLD}{'=' * 62}{RESET}\n")
    return {"passed": passed, "failed": failed, "total": total}


# ── Interactive REPL ──────────────────────────────────────────────────────────

def interactive_mode(k: int = 4) -> None:
    print(f"\n{BOLD}Interactive retrieval probe{RESET} (type 'exit' to quit)")
    print(f"k = {k} chunks per query\n")
    vs = load_vectorstore()

    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("exit", "quit", "q"):
            break

        results = retrieve_with_scores(query, vs, k=k)
        print()
        for rank, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "?")
            colour = _score_colour(score)
            print(f"  #{rank} {colour}[{score:.2f}]{RESET} {CYAN}{source}{RESET}")
            # Show full chunk text, wrapped at 72 chars
            for line in textwrap.wrap(doc.page_content.strip(), width=72):
                print(f"     {line}")
            print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval quality tests")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Open an interactive query REPL after the test suite")
    parser.add_argument("--k", type=int, default=4,
                        help="Number of chunks to retrieve (default: 4)")
    args = parser.parse_args()

    # Enable ANSI colours on Windows
    if sys.platform == "win32":
        import os
        os.system("color")

    run_tests(k=args.k)

    if args.interactive:
        interactive_mode(k=args.k)
