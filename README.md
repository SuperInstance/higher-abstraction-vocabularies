# Higher Abstraction Vocabularies (HAV)

**A vocabulary engine for agents and humans to communicate about complex concepts with precision.**

> "Stigmergy" compresses "indirect coordination through environment modification where agents communicate by leaving traces" into one word. The fleet needs thousands of these compressions.

## What It Is

HAV is a structured vocabulary of **68 terms** across **12 domains** — from confidence fusion to circadian rhythms, from circuit breakers to the paradox of choice. Every term has:

- **Short definition** — one line
- **Full description** — with context and fleet integration
- **Examples** — real-world and fleet-specific
- **Cross-domain bridges** — how this term connects to concepts in other domains
- **Abstraction level** — from concrete implementation to meta-pattern
- **Tags** — for discovery and filtering

## Why It Matters

### For Agents

Agents need shared vocabulary to communicate precisely. Instead of sending a 500-byte JSON payload explaining "I observed the path was blocked and therefore I am considering alternatives and will select the one with highest confidence", an agent sends one word: `deliberation`. Both agents look up `deliberation` in HAV and share the full semantics.

```python
from vocab import HAV

hav = HAV()

# Agent A wants to explain its state
hav.explain("deliberation")
# => "Structured consideration of options leading to a decision..."
#    Includes the Consider/Resolve/Forfeit protocol,
#    confidence threshold (0.85), and fleet integration.

# Agent B wants to understand a message
hav.search("confidence below threshold, abandoning proposal")
# => [("forfeit", 0.3), ("convergence", 0.2), ...]
```

### For Humans

HAV is a field guide to the concepts that power autonomous agent systems. Read it to understand:

- **Why** the fleet uses harmonic mean for confidence (not arithmetic mean)
- **What** biological metaphors map to software patterns (dopamine = confidence)
- **How** agents coordinate without central control (stigmergy, consensus, gossip)
- **When** to use satisficing vs optimizing (energy constraints)

## Domains

| Domain | Terms | Covers |
|--------|-------|--------|
| **uncertainty** | 7 | Confidence, trust, Bayesian update, entropy, calibration |
| **memory** | 8 | Working, episodic, semantic, procedural, forgetting, consolidation |
| **coordination** | 8 | Stigmergy, consensus, swarm, emergence, leader election |
| **biological** | 8 | Instinct, apoptosis, circadian, neurotransmitter, membrane |
| **learning** | 7 | Exploration/exploitation, credit assignment, transfer, curriculum |
| **architecture** | 7 | Actor model, circuit breaker, bulkhead, event sourcing |
| **mathematics** | 5 | Harmonic mean, exponential decay, Welford, topological sort |
| **spatial** | 4 | Attention tiles, A*, Manhattan distance, spatial hashing |
| **communication** | 4 | Grounding, speech acts, information bottleneck, context window |
| **decision** | 4 | Satisficing, multi-armed bandit, minimax, paradox of choice |
| **temporal** | 3 | Deadline urgency, causal chains, heartbeats |
| **security** | 3 | Least privilege, sandbox, graceful degradation |

## CLI

```bash
# Search for a concept
python3 src/cli.py search "memory that fades"

# Get full explanation
python3 src/cli.py explain stigmergy

# Suggest terms for a natural-language intent
python3 src/cli.py suggest "gradually reduce options until one remains"

# Find cross-domain bridges
python3 src/cli.py bridge confidence from uncertainty to memory

# Explore a random term
python3 src/cli.py explore

# List all domains
python3 src/cli.py domains

# Show all terms
python3 src/cli.py all

# Statistics
python3 src/cli.py stats
```

## Programmatic API

```python
from vocab import HAV

hav = HAV()

# Search across all domains
results = hav.search("trust between agents")
for domain, term, score in results:
    print(f"{term.name} ({domain}): {term.short} [{score:.2f}]")

# Get human-readable explanation
print(hav.explain("harmonic-mean-fusion"))

# Suggest terms for an intent
suggestions = hav.suggest("I need to handle too many requests")
# -> [("backpressure", 0.2), ("rate-limit", 0.15), ...]

# Find cross-domain bridges
bridges = hav.bridge("confidence", from_domain="uncertainty", to_domain="biological")
# -> [("biological", Term("neurotransmitter", ...))]

# Vocabulary statistics
stats = hav.stats()
# -> {"namespaces": 12, "total_terms": 68, ...}
```

## Key Design Decisions

### Harmonic Mean, Not Arithmetic Mean

When fusing confidence, the fleet uses harmonic mean: `1/(1/a + 1/b)`. This means any uncertain source drags down the ensemble. Arithmetic mean would let a confident-but-wrong source overpower uncertain-but-correct ones.

### Abstraction Levels

Every term has a level:
- **Concrete** (0): Specific implementation — quick-sort, TCP handshake
- **Pattern** (1): Design pattern — divide-and-conquer, circuit breaker
- **Behavior** (2): Observable behavior — emergence, convergence
- **Domain** (3): Domain concept — confidence, trust, homeostasis
- **Meta** (4): Cross-domain — compression, coupling, phase-transition

### Cross-Domain Bridges

Terms in different domains reference each other via `bridges`. "Dopamine IS confidence" isn't just a metaphor — the bridge connects `neurotransmitter` (biological) to `confidence` (uncertainty), indicating they share the same mathematical structure.

## Integration with the Fleet

HAV connects to every layer of the Lucineer fleet:

- **cuda-confidence** / **cuda-equipment**: Core types HAV describes
- **cuda-genepool** / **cuda-biology**: Biological pipeline HAV documents
- **cuda-a2a** / **cuda-communication**: Communication concepts HAV names
- **cuda-actor** / **cuda-workflow**: Architecture patterns HAV defines

## Part of the Lucineer Fleet

[The Fleet](https://github.com/Lucineer/the-fleet) | [Cocapn](https://github.com/Lucineer/cocapn-ai) | [Deckboss](https://github.com/Lucineer/deckboss)

## License

MIT
