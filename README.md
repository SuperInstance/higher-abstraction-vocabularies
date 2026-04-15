# Higher Abstraction Vocabularies (HAV)

**2000 terms across 292 domains** — the exhaustive vocabulary engine for precision ideation.

> Each term compresses paragraphs of explanation into a single word. The vocabulary IS the manual. High-abstraction vocabulary IS the drone above the corn maze.

## Overview

HAV is a structured, searchable vocabulary engine for agents and humans to communicate about complex computational, biological, and systems concepts with maximum precision. Like a field guide for ideas: every term has a definition, examples, cross-domain bridges, and an abstraction level from Concrete(0) to Meta(4).

The core insight: "Stigmergy" compresses "indirect coordination through environment modification where agents communicate by leaving traces that other agents react to" into one word. HAV gives the fleet thousands of these compressions, organized across 292 domains spanning mathematics, biology, architecture, economics, cognition, and more.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI / Python API                          │
│   search("how systems fail")  explain(stigmergy)  bridge()  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                     HAV Engine (vocab.py)                    │
│  ┌──────────┐ ┌───────────┐ ┌────────┐ ┌──────────────┐   │
│  │ Search   │ │  Explain  │ │ Bridge │ │   Suggest    │   │
│  │ Fuzzy    │ │  Markdown │ │ Cross- │ │  NL Intent   │   │
│  │ Token    │ │  Output   │ │ Domain │ │  Matching    │   │
│  │ Overlap  │ │           │ │ Map    │ │              │   │
│  └────┬─────┘ └───────────┘ └────────┘ └──────────────┘   │
└───────┼─────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────┐
│              Term Store (292 Namespaces)                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐    │
│  │uncertain│  │ memory  │  │coordina-│  │ flux-byte- │    │
│  │ ty      │  │         │  │ tion    │  │ codes      │    │
│  │(7 terms)│  │(8 terms)│  │(8 terms)│  │ (1 term)   │    │
│  └─────────┘  └─────────┘  └─────────┘  └───────────┘    │
│  ... 288 more namespaces spanning all knowledge domains ...  │
└─────────────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────┐
│                 FLUX VM Bytecode Mapper                      │
│  term/flavor → base opcode + variant parameter              │
│  prune/aggressive → different bytecode than prune/cautious  │
└─────────────────────────────────────────────────────────────┘
```

## Features & Concepts

### Abstraction Levels (5 tiers)

| Level | Name | Description | Example |
|-------|------|-------------|---------|
| 0 | **Concrete** | Specific implementation | `quick-sort`, `TCP handshake` |
| 1 | **Pattern** | Design pattern | `divide-and-conquer`, `retry-with-backoff` |
| 2 | **Behavior** | Observable behavior | `emergence`, `convergence`, `stigmergy` |
| 3 | **Domain** | Domain concept | `homeostasis`, `confidence`, `trust` |
| 4 | **Meta** | Cross-domain abstraction | `compression`, `coupling`, `phase-transition` |

### Core Capabilities

- **Semantic Search** — Fuzzy matching via substring, token overlap, and tag similarity with configurable thresholds
- **Rich Explanations** — Markdown-formatted output with definitions, aliases, bridges, antonyms, properties, and examples
- **Cross-Domain Bridges** — Maps concepts between domains (e.g., "Stigmergy IS git", "Dopamine IS confidence")
- **NL Suggestion** — Natural language intent matching ("I need to gradually reduce options" → `deliberation`, `convergence`, `pruning`)
- **FLUX Integration** — Terms compile to FLUX VM bytecode opcodes; flavors map to parameter variants

### By the Numbers

| Metric | Value |
|--------|-------|
| Total terms | 1,687 |
| Domains | 252 |
| File size | 513K chars |
| RA ideation rounds | 13+ |

## Quick Start

```bash
git clone https://github.com/SuperInstance/higher-abstraction-vocabularies.git
cd higher-abstraction-vocabularies
python3 -c "from src.vocab import HAV; h = HAV(); print(h.stats())"
```

### CLI Usage

```bash
# Search for a concept
python3 src/cli.py search "how systems fail"

# Explain a specific term
python3 src/cli.py explain anti-fragility

# Find cross-domain bridges
python3 src/cli.py bridge confidence from uncertainty to biological

# View vocabulary statistics
python3 src/cli.py stats
```

### Python API

```python
from src.vocab import HAV

hav = HAV()

# Semantic search
hav.search("memory that fades")
# -> [('episodic-decay', 0.8), ('forgetting-curve', 0.6), ...]

# Rich explanation
hav.explain("harmonic-mean-fusion")
# -> Full markdown with examples and cross-domain bridges

# Cross-domain bridging
hav.bridge("fold", from_domain="mathematics", to_domain="memory")

# NL suggestion
hav.suggest("I need to... gradually reduce options until one remains")
# -> ['deliberation', 'convergence', 'filtration', 'pruning']
```

## Integration

### With FLUX VM

HAV terms compile directly to FLUX VM bytecode. Each term name maps to base opcodes; flavors map to parameter variants:

- `prune/aggressive` → different bytecode than `prune/cautious`
- `sense/focused` → narrow high-resolution sensing
- `act/reflexive` → skip deliberation, execute immediately
- `communicate/gossip` → stochastic peer message forwarding

### With Fleet Agents

Import `HAV` in any fleet agent to gain shared vocabulary. When two agents both use "stigmergy" they mean exactly the same thing — no ambiguity, no alignment overhead.

## Part of the Lucineer Fleet

[The Fleet](https://github.com/Lucineer/the-fleet) | [Cocapn](https://github.com/Lucineer/cocapn-ai) | [Deckboss](https://github.com/Lucineer/deckboss) | [HAV RA Research](https://github.com/Lucineer/hav-reverse-actualization) | [Iron-to-Iron](https://github.com/SuperInstance/iron-to-iron)

## License

MIT

---

<img src="callsign1.jpg" width="128" alt="callsign">
