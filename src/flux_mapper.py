#!/usr/bin/env python3
"""Map HAV vocabulary terms to FLUX bytecode operations.

Vocabulary terms that describe recurring patterns become FLUX opcodes.
This module analyzes the fleet's vocabulary and proposes new opcodes
based on term frequency, abstraction level, and cross-domain bridging.

Usage:
    python3 flux_mapper.py          # analyze and propose
    python3 flux_mapper.py --apply  # generate opcode proposals
"""
import sys
sys.path.insert(0, '.')
from vocab import HAV
from collections import Counter

def analyze_opcode_candidates(hav=None):
    """Find vocabulary terms that should become FLUX opcodes."""
    if not hav:
        hav = HAV()

    candidates = []

    for ns_name, ns in hav._namespaces.items():
        for term_name, term in ns.terms.items():
            score = 0

            # High bridge count = widely applicable = good opcode candidate
            bridge_count = len(term.bridges) if term.bridges else 0
            score += bridge_count * 3

            # Lower abstraction = more concrete = more useful as opcode
            level = term.level.value if hasattr(term.level, 'value') else term.level
            score += (4 - level) * 2

            # Short names are better for opcodes
            if len(term_name) <= 20:
                score += 2

            # Verbs in short description suggest action = opcode material
            verb_starts = ['reduce', 'combine', 'filter', 'merge', 'fold',
                          'map', 'scan', 'push', 'pull', 'dispatch',
                          'route', 'bridge', 'adapt', 'evolve', 'compile']
            if any(term.short.lower().startswith(v) for v in verb_starts):
                score += 5

            # Fleet-relevant domains get a boost
            fleet_domains = ['agent-patterns', 'flux-vm', 'fleet-biology',
                           'coordination', 'composition', 'scaling']
            if ns_name in fleet_domains:
                score += 3

            if score >= 8:
                candidates.append({
                    'term': term_name,
                    'domain': ns_name,
                    'score': score,
                    'short': term.short,
                    'level': level,
                    'bridges': bridge_count,
                    'proposed_opcode': f'HAV_{term_name.upper().replace("-", "_")}',
                })

    candidates.sort(key=lambda x: -x['score'])
    return candidates

def propose_opcodes(candidates, top_n=20):
    """Generate FLUX opcode proposals from top candidates."""
    print(f"Top {top_n} FLUX opcode proposals from HAV:\n")
    for i, c in enumerate(candidates[:top_n], 1):
        print(f"  {i:2d}. {c['proposed_opcode']}")
        print(f"      Score: {c['score']} | Level: L{c['level']} | Bridges: {c['bridges']}")
        print(f"      Def: {c['short']}")
        print()

if __name__ == '__main__':
    candidates = analyze_opcode_candidates()
    print(f"Found {len(candidates)} opcode candidates (score >= 8)\n")
    propose_opcodes(candidates)

    if '--apply' in sys.argv:
        import json
        with open('docs/opcode-proposals.json', 'w') as f:
            json.dump(candidates, f, indent=2)
        print(f"Saved {len(candidates)} proposals to docs/opcode-proposals.json")
