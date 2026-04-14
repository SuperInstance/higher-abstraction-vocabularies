#!/usr/bin/env python3
"""Export HAV vocabulary to JSON for web consumption."""
import json
import sys
sys.path.insert(0, '.')
from vocab import HAV

def export_json(output_path='docs/hav.json'):
    hav = HAV()
    data = {
        'meta': hav.stats(),
        'domains': {}
    }
    for ns_name, ns in hav._namespaces.items():
        domain = {
            'description': ns.description if hasattr(ns, 'description') else '',
            'terms': {}
        }
        for term_name, term in ns.terms.items():
            domain['terms'][term_name] = {
                'short': term.short,
                'description': term.description,
                'level': term.level.value if hasattr(term.level, 'value') else term.level,
                'examples': term.examples,
                'bridges': term.bridges,
                'tags': term.tags,
                'aliases': term.aliases,
                'antonyms': term.antonyms,
            }
        data['domains'][ns_name] = domain

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Exported {data['meta']['total_terms']} terms to {output_path}")
    return data

if __name__ == '__main__':
    export_json(sys.argv[1] if len(sys.argv) > 1 else 'docs/hav.json')
