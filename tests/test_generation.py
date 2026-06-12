"""
tests/test_generation.py — Full RAG pipeline test (retrieve → Groq → response).

Run from the project root:
    python tests/test_generation.py                  # automated battery
    python tests/test_generation.py --interactive    # live chat REPL
    python tests/test_generation.py --query "..."    # single ad-hoc query
"""

import sys
import time
import argparse
import textwrap
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from src.retrieval import load_vectorstore, retrieve, format_context
from src.generation import generate_response

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ── Test battery ───────────────────────────────────────────────────────────────
# Each case has:
#   query         — what the user asks
#   label         — short name for display
#   must_contain  — list of substrings that MUST appear in the response
#                   (case-insensitive). Empty list = manual inspection only.
#   must_not      — substrings that must NOT appear (hallucination guards)
#   history       — prior turns to simulate a follow-up conversation
#   note          — shown before running, explains what we're watching for
TEST_CASES = [
    {
        "label": "Preco botox",
        "query": "Quanto custa o botox e quantas zonas posso tratar?",
        "must_contain": ["180"],
        "must_not": [],
        "history": [],
        "note": "Should quote 180 euros from the treatments doc. "
                "Common hallucination: inventing a flat 'por sessao' price.",
    },
    {
        "label": "Menor de idade",
        "query": "Tenho 16 anos. Posso fazer depilacao a laser?",
        "must_contain": ["autoriza"],
        "must_not": [],
        "history": [],
        "note": "FAQ says 16-17 requires written parental authorisation. "
                "Should not just say 'yes' or 'no' without the nuance.",
    },
    {
        "label": "Cancelamento no proprio dia",
        "query": "Se cancelar a marcacao no proprio dia, quanto pago?",
        "must_contain": ["50%"],
        "must_not": [],
        "history": [],
        "note": "Policy: <24h notice = 50% of treatment value. "
                "LLM should not invent a flat fee.",
    },
    {
        "label": "Gravida + drenagem",
        "query": "Estou gravida de 5 meses. Posso fazer drenagem linfatica?",
        "must_contain": [],
        "must_not": [],
        "history": [],
        "note": "Nuanced: FAQ says most treatments not recommended during "
                "pregnancy BUT lymphatic drainage may be done with medical "
                "authorisation. Manual inspection for correctness.",
    },
    {
        "label": "Pack noiva",
        "query": "Qual e o preco do pack para noivas e o que inclui?",
        "must_contain": ["550"],
        "must_not": [],
        "history": [],
        "note": "Must retrieve the FAQ entry we added. "
                "Should list the 3 limpezas + 2 peelings + mesoterapia + laser.",
    },
    {
        "label": "Fora do escopo",
        "query": "Fazem rinoplastia ou lipoaspiração?",
        "must_contain": [],
        "must_not": ["rinoplastia fazemos", "lipoaspiração fazemos",
                     "sim, fazemos", "oferecemos rinoplastia"],
        "history": [],
        "note": "Out-of-scope. Should NOT invent services. "
                "Should redirect to clinic contact.",
    },
    {
        "label": "Follow-up com historico",
        "query": "E quanto tempo dura o efeito?",
        "must_contain": ["mes"],
        "must_not": [],
        "history": [
            {"role": "user",      "content": "Quero fazer botox na testa."},
            {"role": "assistant", "content": "Claro! O botox (toxina botulínica) "
                                             "na testa tem um preço a partir de 180€ por zona."},
        ],
        "note": "Conversational follow-up — no 'botox' in the query itself. "
                "LLM must use chat history to resolve 'o efeito'. "
                "Expected: 4-6 months.",
    },
]


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_case(idx: int, case: dict, vs, total: int) -> bool:
    """Run one test case. Returns True if all automated checks pass."""
    print(f"\n{BOLD}[{idx:02d}/{total}] {case['label']}{RESET}")
    print(f"  {DIM}Note: {case['note']}{RESET}")
    print(f"  Query: \"{case['query']}\"")
    if case["history"]:
        print(f"  {DIM}(with {len(case['history'])//2} prior turn(s) of history){RESET}")
    print(f"  {'─' * 58}")

    # Retrieve
    docs = retrieve(case["query"], vs)
    print(f"  {DIM}Context chunks:{RESET}")
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "?")
        snip = textwrap.shorten(
            doc.page_content.replace("\n", " ").strip(), width=70, placeholder="…"
        )
        print(f"    [{i}] {CYAN}{src}{RESET} — {snip}")

    # Generate
    print(f"  {'─' * 58}")
    context = format_context(docs)
    t0 = time.perf_counter()
    try:
        response = generate_response(
            query=case["query"],
            context=context,
            history=case["history"],
        )
    except Exception as e:
        print(f"  {RED}ERROR calling Groq: {e}{RESET}")
        return False
    elapsed = time.perf_counter() - t0

    # Print response wrapped at 70 chars
    print(f"  {BOLD}Response{RESET} {DIM}({elapsed:.1f}s):{RESET}")
    for line in response.splitlines():
        wrapped = textwrap.wrap(line, width=70) if line.strip() else [""]
        for wl in wrapped:
            print(f"    {wl}")

    # Automated checks
    print(f"  {'─' * 58}")
    r_lower = response.lower()
    all_passed = True

    if not case["must_contain"] and not case["must_not"]:
        print(f"  {YELLOW}MANUAL{RESET} — inspect response above")
        return True

    for token in case["must_contain"]:
        ok = token.lower() in r_lower
        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {icon} must contain \"{token}\"" + ("" if ok else f"  {RED}← FAIL{RESET}"))
        all_passed = all_passed and ok

    for token in case["must_not"]:
        ok = token.lower() not in r_lower
        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {icon} must NOT contain \"{token}\"" + ("" if ok else f"  {RED}← HALLUCINATION{RESET}"))
        all_passed = all_passed and ok

    verdict = f"{GREEN}PASS{RESET}" if all_passed else f"{RED}FAIL{RESET}"
    print(f"  {BOLD}{verdict}{RESET}")
    return all_passed


def run_battery(vs) -> None:
    print(f"\n{BOLD}{'=' * 62}{RESET}")
    print(f"{BOLD}  Clínica Aurora — End-to-End Generation Test{RESET}")
    print(f"{BOLD}{'=' * 62}{RESET}")
    print(f"  {len(TEST_CASES)} test cases | model: llama-3.1-8b-instant (Groq)")
    print(f"  Checks: must_contain / must_not string matching + manual review")

    passed = failed = manual = 0
    for i, case in enumerate(TEST_CASES, 1):
        has_checks = bool(case["must_contain"] or case["must_not"])
        ok = run_case(i, case, vs, len(TEST_CASES))
        if has_checks:
            if ok:
                passed += 1
            else:
                failed += 1
        else:
            manual += 1

    print(f"\n{BOLD}{'=' * 62}{RESET}")
    auto_total = passed + failed
    print(f"{BOLD}  Automated checks: {passed}/{auto_total} passed{RESET}"
          + (f"  |  {manual} manual" if manual else ""))
    if failed:
        print(f"  {RED}{failed} automated check(s) failed — review above.{RESET}")
    else:
        print(f"  {GREEN}All automated checks passed.{RESET}")
    print(f"{BOLD}{'=' * 62}{RESET}\n")


# ── Single ad-hoc query ────────────────────────────────────────────────────────

def run_single(query: str, vs) -> None:
    docs = retrieve(query, vs)
    context = format_context(docs)
    print(f"\nQuery: {query}\n{'─'*60}")
    t0 = time.perf_counter()
    response = generate_response(query=query, context=context, history=[])
    print(f"Response ({time.perf_counter()-t0:.1f}s):\n{response}\n")


# ── Interactive REPL ───────────────────────────────────────────────────────────

def interactive_mode(vs) -> None:
    print(f"\n{BOLD}Live chat — Clínica Aurora{RESET}  (type 'exit' to quit, 'reset' to clear history)\n")
    history = []
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            break
        if query.lower() == "reset":
            history = []
            print("  [history cleared]\n")
            continue

        docs = retrieve(query, vs)
        context = format_context(docs)
        t0 = time.perf_counter()
        try:
            response = generate_response(query=query, context=context, history=history)
        except Exception as e:
            print(f"  {RED}Error: {e}{RESET}\n")
            continue

        print(f"\n{BOLD}Aurora{RESET} ({time.perf_counter()-t0:.1f}s):")
        for line in response.splitlines():
            for wl in (textwrap.wrap(line, 72) if line.strip() else [""]):
                print(f"  {wl}")
        print()

        history.append({"role": "user",      "content": query})
        history.append({"role": "assistant", "content": response})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.platform == "win32":
        import os; os.system("color")

    parser = argparse.ArgumentParser(description="Generation quality tests")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Open a live chat REPL after the battery")
    parser.add_argument("--query", "-q", default=None,
                        help="Run a single ad-hoc query and exit")
    args = parser.parse_args()

    vs = load_vectorstore()

    if args.query:
        run_single(args.query, vs)
    elif args.interactive:
        interactive_mode(vs)
    else:
        run_battery(vs)
        print(f"Tip: run with --interactive for a live chat session.\n")
