#!/usr/bin/env python3
"""HAV CLI — explore Higher Abstraction Vocabularies from the terminal.

Usage:
    python3 cli.py search "memory that fades"
    python3 cli.py explain harmonic-mean-fusion
    python3 cli.py suggest "gradually reduce options until one remains"
    python3 cli.py bridge confidence from uncertainty to memory
    python3 cli.py explore        # show a random term
    python3 cli.py domains        # list all domains with term counts
    python3 cli.py all            # list all terms
    python3 cli.py stats          # show vocabulary statistics
"""

import sys
import random
from vocab import HAV


def cmd_search(hav: HAV, query: str):
    results = hav.search(query)
    if not results:
        print(f"No results for '{query}'. Try broader terms.")
        return
    print(f"Found {len(results)} result(s) for '{query}':\n")
    for ns, term, score in results:
        tag_str = ", ".join(term.tags[:3])
        print(f"  [{score:.2f}] {term.name} ({ns})")
        print(f"         {term.short}")
        if tag_str:
            print(f"         tags: {tag_str}")
        print()


def cmd_explain(hav: HAV, name: str):
    explanation = hav.explain(name)
    print(explanation)


def cmd_suggest(hav: HAV, intent: str):
    results = hav.suggest(intent)
    if not results:
        print("No suggestions found. Try rephrasing your intent.")
        return
    print(f"Suggestions for: \"{intent}\"\n")
    for ns, term, score in results:
        print(f"  [{score:.2f}] {term.name} — {term.short}")
    print()


def cmd_bridge(hav: HAV, term_name: str, from_domain: str = "", to_domain: str = ""):
    bridges = hav.bridge(term_name, from_domain, to_domain)
    if not bridges:
        print(f"No bridges found for '{term_name}'.")
        return
    print(f"Bridges for '{term_name}':\n")
    for ns, term in bridges:
        print(f"  {ns}/{term.name}: {term.short}")
    print()


def cmd_explore(hav: HAV):
    term = hav.random_term()
    if not term:
        print("No terms in vocabulary.")
        return
    print("Random term exploration:\n")
    print(term.explain())
    print("\n---\n")
    # Show related terms
    if term.bridges:
        print("Related terms:")
        for bridge in term.bridges[:5]:
            explanation = hav.explain(bridge)
            first_line = explanation.split("\n")[1] if len(explanation.split("\n")) > 1 else ""
            print(f"  - {bridge}: {first_line.strip()}")


def cmd_domains(hav: HAV):
    stats = hav.stats()
    print(f"HAV — {stats['total_terms']} terms across {stats['namespaces']} domains:\n")
    for name, count in sorted(stats["by_domain"].items(), key=lambda x: -x[1]):
        ns = hav.namespace(name)
        desc = ns.description[:60] + "..." if ns and len(ns.description) > 60 else (ns.description or "")
        print(f"  {name:20s} ({count:2d} terms)  {desc}")
    print()


def cmd_all(hav: HAV):
    stats = hav.stats()
    print(f"HAV — {stats['total_terms']} terms:\n")
    for name in sorted(stats["by_domain"].keys()):
        ns = hav.namespace(name)
        print(f"[{name}]")
        for term in sorted(ns.terms.values(), key=lambda t: t.name):
            print(f"  {term.name:35s} L{term.level.value}  {term.short}")
        print()


def cmd_stats(hav: HAV):
    stats = hav.stats()
    print("HAV Statistics")
    print("=" * 40)
    print(f"  Namespaces:   {stats['namespaces']}")
    print(f"  Total terms:  {stats['total_terms']}")
    print(f"  Domains:")
    for name, count in sorted(stats["by_domain"].items(), key=lambda x: -x[1]):
        print(f"    {name:25s} {count:3d} terms")
    # Level distribution
    level_counts = {}
    for ns in hav._namespaces.values():
        for t in ns.terms.values():
            level_counts[t.level.name] = level_counts.get(t.level.name, 0) + 1
    print(f"  Levels:")
    for level, count in sorted(level_counts.items()):
        print(f"    {level:15s} {count:3d}")


def main():
    hav = HAV()

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "search" and len(sys.argv) > 2:
        cmd_search(hav, " ".join(sys.argv[2:]))
    elif command == "explain" and len(sys.argv) > 2:
        cmd_explain(hav, sys.argv[2])
    elif command == "suggest" and len(sys.argv) > 2:
        cmd_suggest(hav, " ".join(sys.argv[2:]))
    elif command == "bridge" and len(sys.argv) > 2:
        # bridge <term> [from <domain>] [to <domain>]
        term = sys.argv[2]
        from_d = ""
        to_d = ""
        args = sys.argv[3:]
        i = 0
        while i < len(args):
            if args[i] == "from" and i + 1 < len(args):
                from_d = args[i + 1]
                i += 2
            elif args[i] == "to" and i + 1 < len(args):
                to_d = args[i + 1]
                i += 2
            else:
                i += 1
        cmd_bridge(hav, term, from_d, to_d)
    elif command == "explore":
        cmd_explore(hav)
    elif command == "domains":
        cmd_domains(hav)
    elif command == "all":
        cmd_all(hav)
    elif command == "stats":
        cmd_stats(hav)
    else:
        # Fallback: treat as search
        cmd_search(hav, " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
