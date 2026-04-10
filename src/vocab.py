"""
Higher Abstraction Vocabularies (HAV)
======================================
A structured vocabulary engine for agents and humans to communicate about
complex computational, biological, and systems concepts with precision.

Like a field guide for ideas: every term has a definition, examples,
cross-domain bridges, and abstraction level. Agents use it to compress
complex state into shared nouns. Humans use it to understand what agents
are actually doing.

Core insight: "Stigmergy" compresses "indirect coordination through
environment modification where agents communicate by leaving traces
that other agents react to" into one word. The fleet needs thousands
of these compressions.

Usage:
    from vocab import HAV

    hav = HAV()
    hav.search("memory that fades")
    # -> [('episodic-decay', 0.8), ('forgetting-curve', 0.6), ...]

    hav.explain("harmonic-mean-fusion")
    # -> Human-readable explanation with examples and cross-domain bridges

    hav.bridge("fold", from_domain="mathematics", to_domain="memory")
    # -> Maps mathematical fold to memory consolidation

    hav.suggest("I need to... gradually reduce options until one remains")
    # -> Suggests: deliberation, convergence, filtration, pruning
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Abstraction Levels
# ---------------------------------------------------------------------------

class Level(Enum):
    """How abstract a term is, from concrete implementation to meta-pattern."""
    CONCRETE = 0    # Specific implementation (quick-sort, TCP handshake)
    PATTERN = 1     # Design pattern (divide-and-conquer, retry-with-backoff)
    BEHAVIOR = 2    # Observable behavior (emergence, convergence, stigmergy)
    DOMAIN = 3      # Domain concept (homeostasis, confidence, trust)
    META = 4        # Cross-domain abstraction (compression, coupling, phase-transition)


# ---------------------------------------------------------------------------
# Core Types
# ---------------------------------------------------------------------------

@dataclass
class Term:
    """A vocabulary term with rich metadata."""
    name: str
    short: str                                   # One-line definition
    description: str = ""                        # Full explanation
    level: Level = Level.PATTERN
    domain: str = "general"
    examples: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    bridges: List[str] = field(default_factory=list)   # Other terms this connects to
    antonyms: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def matches(self, query: str) -> float:
        """Fuzzy match score for search. Uses substring + token overlap."""
        q = query.lower().strip()
        if not q:
            return 0.0
        score = 0.0
        name_lo = self.name.lower()
        short_lo = self.short.lower()
        desc_lo = self.description.lower()

        # Exact name match
        if q == name_lo or q in self.aliases:
            return 1.0
        if q == name_lo.replace("-", ""):
            return 0.95

        # Substring matches (weighted by field importance)
        if q in name_lo:
            score += 0.5
        if q in short_lo:
            score += 0.35
        if q in desc_lo:
            score += 0.2

        # Token overlap
        qtokens = set(re.split(r"[\s\-_/]+", q))
        all_text = " ".join([name_lo, short_lo, desc_lo, *self.examples,
                             *self.aliases, *self.tags]).lower()
        all_tokens = set(re.split(r"[\s\-_/.,;:()]+", all_text))
        if qtokens:
            overlap = len(qtokens & all_tokens) / len(qtokens)
            score += overlap * 0.3

        # Example matches
        for ex in self.examples:
            if q in ex.lower():
                score += 0.08

        # Tag matches
        for tag in self.tags:
            if q in tag.lower() or tag.lower() in q:
                score += 0.1

        return min(score, 1.0)

    def explain(self) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"## {self.name}",
            "",
            self.short,
            "",
        ]
        if self.description:
            lines.append(self.description)
            lines.append("")
        if self.aliases:
            lines.append(f"**Also known as:** {', '.join(self.aliases)}")
        if self.tags:
            lines.append(f"**Tags:** {', '.join(self.tags)}")
        if self.properties:
            lines.append("**Properties:**")
            for k, v in self.properties.items():
                lines.append(f"- {k}: {v}")
        if self.examples:
            lines.append("**Examples:**")
            for ex in self.examples:
                lines.append(f"- {ex}")
        if self.bridges:
            lines.append("**See also:** " + ", ".join(f"`{b}`" for b in self.bridges))
        if self.antonyms:
            lines.append("**Opposite of:** " + ", ".join(f"`{a}`" for a in self.antonyms))
        return "\n".join(lines)


@dataclass
class Namespace:
    """A named collection of terms within a domain."""
    name: str
    description: str = ""
    terms: Dict[str, Term] = field(default_factory=dict)

    def define(self, name: str, short: str, **kwargs) -> Term:
        t = Term(name=name, short=short, domain=self.name, **kwargs)
        self.terms[name] = t
        return t

    def lookup(self, name: str) -> Optional[Term]:
        return self.terms.get(name)

    def search(self, query: str, threshold: float = 0.08) -> List[Tuple[Term, float]]:
        results = [(t, t.matches(query)) for t in self.terms.values()]
        return sorted([(t, s) for t, s in results if s >= threshold],
                       key=lambda x: -x[1])

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self) -> Iterator[Term]:
        return iter(self.terms.values())


class HAV:
    """Higher Abstraction Vocabulary engine.

    Provides structured vocabulary for agents and humans to communicate
    about complex concepts with precision. Supports search, explanation,
    cross-domain bridging, and suggestion.
    """

    def __init__(self):
        self._namespaces: Dict[str, Namespace] = {}
        self._load_builtin()

    # --- Namespace Management ---

    def add_namespace(self, name: str, description: str = "") -> Namespace:
        ns = Namespace(name=name, description=description)
        self._namespaces[name] = ns
        return ns

    def namespace(self, name: str) -> Optional[Namespace]:
        return self._namespaces.get(name)

    def define(self, ns_name: str, term_name: str, short: str, **kwargs) -> Term:
        if ns_name not in self._namespaces:
            self.add_namespace(ns_name)
        return self._namespaces[ns_name].define(term_name, short, **kwargs)

    # --- Search ---

    def search(self, query: str, threshold: float = 0.08,
               domain: Optional[str] = None) -> List[Tuple[str, Term, float]]:
        """Search all namespaces (or one) for matching terms."""
        results = []
        domains = {domain} if domain else set(self._namespaces.keys())
        for ns_name in domains:
            ns = self._namespaces.get(ns_name)
            if not ns:
                continue
            for term, score in ns.search(query, threshold):
                results.append((ns_name, term, score))
        return sorted(results, key=lambda x: -x[2])

    def explain(self, name: str) -> str:
        """Get human-readable explanation for a term."""
        for ns in self._namespaces.values():
            t = ns.lookup(name)
            if t:
                return t.explain()
        matches = self.search(name, threshold=0.3)
        if matches:
            return f"No exact match for '{name}'. Did you mean:\n" + \
                   "\n".join(f"- `{t.name}` ({ns}): {t.short}" for ns, t, _ in matches[:5])
        return f"No match for '{name}'."

    def suggest(self, intent: str) -> List[Tuple[str, Term, float]]:
        """Suggest terms that match a natural-language intent."""
        return self.search(intent, threshold=0.05)[:10]

    def bridge(self, term_name: str, from_domain: str = "",
               to_domain: str = "") -> List[Tuple[str, Term]]:
        """Find cross-domain bridges for a term."""
        bridges = []
        for ns_name, ns in self._namespaces.items():
            if to_domain and ns_name != to_domain:
                continue
            for t in ns.terms.values():
                if term_name in t.bridges or term_name in t.aliases:
                    if from_domain and t.domain != from_domain:
                        continue
                    bridges.append((ns_name, t))
        return bridges

    def random_term(self) -> Optional[Term]:
        """Return a random term for exploration."""
        import random
        all_terms = [t for ns in self._namespaces.values() for t in ns.terms.values()]
        return random.choice(all_terms) if all_terms else None

    def stats(self) -> Dict[str, Any]:
        return {
            "namespaces": len(self._namespaces),
            "total_terms": sum(len(ns) for ns in self._namespaces.values()),
            "by_domain": {name: len(ns) for name, ns in self._namespaces.items()},
        }

    # --- Builtin Vocabularies ---

    def _load_builtin(self):
        self._load_uncertainty()
        self._load_memory()
        self._load_coordination()
        self._load_learning()
        self._load_biological()
        self._load_architecture()
        self._load_spatial()
        self._load_temporal()
        self._load_communication()
        self._load_security()
        self._load_decision()
        self._load_control_theory()
        self._load_evolution()
        self._load_networks()
        self._load_game_theory()
        self._load_optimization()
        self._load_probability()
        self._load_economics()
        self._load_ecology()
        self._load_emotion()
        self._load_creativity()
        self._load_metacognition()
        self._load_failure_modes()
        self._load_thermodynamics()
        self._load_complexity()
        self._load_scaling()
        self._load_linguistics()
        self._load_mathematics()

    def _load_uncertainty(self):
        ns = self.add_namespace("uncertainty",
            "Confidence, trust, belief, and probability — how agents handle not-knowing")

        ns.define("confidence",
            "A 0-1 value representing certainty about a proposition or observation",
            description="In the fleet, confidence is a first-class type that propagates through all computation. Two independent confidences fuse via harmonic mean: fused = 1/(1/a + 1/b). This means any uncertain source drags down the whole ensemble, preventing false certainty from aggregating noisy signals.",
            level=Level.DOMAIN,
            examples=["sensor confidence 0.95", "prediction confidence 0.4", "fused confidence after combining two sources"],
            properties={"range": "0.0 to 1.0", "fusion": "harmonic-mean", "unit": "scalar"},
            bridges=["trust", "belief", "probability", "certainty", "information"],
            aliases=["certainty", "sureness", "belief-strength"],
            tags=["core", "fleet-foundation", "propagation"])

        ns.define("harmonic-mean-fusion",
            "Combining independent confidence sources via 1/(1/a + 1/b)",
            description="Unlike arithmetic mean, harmonic mean penalizes uncertainty. Two confidences of 0.9 and 0.1 produce 0.09 (not 0.5). This is critical for agent safety: if ANY sensor or reasoning step is uncertain, the agent should be cautious, not average away the doubt.",
            level=Level.PATTERN,
            examples=["fusing sensor reading 0.95 with prior 0.7 = 0.804", "fusing 0.9 with 0.1 = 0.09 (not 0.5!)"],
            properties={"formula": "1/(1/a + 1/b)", "penalizes": "uncertainty", "used_in": "cuda-confidence, cuda-fusion, cuda-sensor-agent"},
            bridges=["bayesian-update", "weighted-average", "consensus"],
            tags=["core", "mathematics", "fusion"])

        ns.define("trust",
            "Slowly-accumulating confidence in another agent's reliability",
            description="Trust grows slowly (rate = decay_rate / 10) but decays exponentially. This asymmetry means agents must earn trust through consistent good behavior, but a single betrayal can destroy it. Trust is per-capability: you can trust an agent for navigation but not for cooking.",
            level=Level.DOMAIN,
            examples=["trust level 0.7 for pathfinding", "trust drops from 0.8 to 0.2 after failed promise", "gossip: agent shares trust assessments with neighbors"],
            properties={"decay": "exponential", "growth_rate": "1/10 of decay", "per_context": True},
            bridges=["confidence", "reputation", "credit-assignment"],
            aliases=["reliability-belief", "agent-faith"],
            tags=["core", "social", "security"])

        ns.define("bayesian-update",
            "Adjusting beliefs based on new evidence using prior and likelihood",
            description="The mathematical foundation of learning. Prior belief + new evidence = posterior belief. In the fleet, this appears in sensor fusion (cuda-fusion), trust updates (cuda-trust), and confidence propagation (cuda-confidence).",
            level=Level.PATTERN,
            examples=["prior 0.5 + evidence favoring A -> posterior 0.8", "medical diagnosis: symptoms update disease probability"],
            bridges=["harmonic-mean-fusion", "confidence", "learning-rate"],
            tags=["mathematics", "learning", "statistics"])

        ns.define("entropy",
            "Measure of uncertainty or surprise in a distribution",
            description="High entropy = many possible outcomes, equally likely. Low entropy = one outcome dominates. Agents monitor entropy to detect when they understand a situation (low entropy) vs when they're lost (high entropy).",
            level=Level.DOMAIN,
            examples=["uniform distribution = maximum entropy", "coin flip: H(p) = -p*log(p) - (1-p)*log(1-p)", "entropy spike means agent encountered something surprising"],
            bridges=["uncertainty", "surprise", "information", "exploration"],
            tags=["mathematics", "information-theory"])

        ns.define("calibration",
            "How well an agent's confidence matches its actual accuracy",
            description="A well-calibrated agent that says '90% confident' is right 90% of the time. Most agents (and humans) are poorly calibrated — overconfident on hard problems, underconfident on easy ones. The fleet tracks calibration via self-assessed vs actual performance in cuda-self-model.",
            level=Level.BEHAVIOR,
            examples=["forecasting: said 80% chance of rain, it rained 80% of those times", "agent says 0.9 confidence, historical accuracy is 0.3 = poorly calibrated"],
            bridges=["confidence", "self-model", "meta-cognition"],
            tags=["agent-behavior", "meta-cognition"])

        ns.define("information",
            "Reduction in uncertainty gained from an observation or message",
            description="A message that tells you nothing you didn't already know carries zero information. A message that completely resolves your uncertainty carries maximum information. In the fleet, information value determines how much energy an agent should spend processing a message.",
            level=Level.DOMAIN,
            examples=["a bit that resolves a coin flip carries 1 bit of information", "redundant message = 0 information", "surprising message = high information"],
            bridges=["entropy", "confidence", "attention", "communication-cost"],
            tags=["information-theory", "communication"])

    def _load_memory(self):
        ns = self.add_namespace("memory",
            "How agents store, retrieve, forget, and consolidate information")

        ns.define("working-memory",
            "Fast, limited-capacity buffer for current task context",
            description="The agent's 'right now'. Typically holds 4-7 items. Decays in seconds. Analogous to CPU registers or human short-term memory. The fleet uses it to hold the current goal, recent observations, and active deliberation state.",
            level=Level.CONCRETE,
            examples=["holding a phone number while dialing", "keeping 3 recent sensor readings in focus", "current goal: navigate to door"],
            properties={"capacity": "4-7 items", "decay": "seconds", "half_life": "~30s"},
            bridges=["attention", "focus", "registers"],
            tags=["memory", "cognition", "fleet"])

        ns.define("episodic-memory",
            "Specific experiences stored with timestamp and emotional valence",
            description="The 'what happened when' memory. Each episode is a record of what happened, when, what the agent felt, and what it learned. Episodes decay over days but emotional intensity slows decay. Important episodes get consolidated into semantic memory.",
            level=Level.DOMAIN,
            examples=["yesterday I tried path A and it was blocked", "last time I talked to navigator, it gave bad directions", "the time the fleet coordinated perfectly on the warehouse task"],
            properties={"decay": "days", "half_life": "~1 week", "emotional_modulation": True},
            bridges=["semantic-memory", "procedural-memory", "narrative", "learning"],
            tags=["memory", "learning", "fleet"])

        ns.define("semantic-memory",
            "General knowledge extracted from many episodes — the wisdom layer",
            description="When enough episodes share a pattern, that pattern gets extracted into semantic memory. 'Navigator gives bad directions near construction' is semantic — it abstracts across many specific episodes. Decays over months. This is the agent's world model.",
            level=Level.DOMAIN,
            examples=["doors in this building are usually on the right wall", "sensor 3 tends to give noisy readings in rain", "collaborative tasks go faster with 3 agents, slower with 5"],
            properties={"decay": "months", "half_life": "~6 months", "source": "episodic-consolidation"},
            bridges=["episodic-memory", "procedural-memory", "world-model", "knowledge"],
            tags=["memory", "learning", "fleet"])

        ns.define("procedural-memory",
            "How to do things — skills, patterns, automatic behaviors",
            description="The 'muscle memory' of an agent. Once a behavior is practiced enough, it becomes procedural — fast, automatic, requiring minimal working memory. In the fleet, cuda-skill manages procedural memory. Procedural memories decay over years.",
            level=Level.DOMAIN,
            examples=["knowing how to navigate a familiar building", "automatic collision avoidance reflex", "typing without thinking about key locations"],
            properties={"decay": "years", "half_life": "~5 years", "automation": True},
            bridges=["working-memory", "skill", "reflex", "habit"],
            tags=["memory", "skill", "fleet"])

        ns.define("forgetting-curve",
            "Exponential decay of memory strength over time without rehearsal",
            description="Ebbinghaus's discovery: memory fades exponentially. The fleet implements this with configurable half-lives per memory layer. Recall strengthens memory (resets decay timer). Critical memories can be 'pinned' to resist decay.",
            level=Level.PATTERN,
            examples=["forget 50% of a lecture within 1 hour without notes", "spaced repetition extends the curve", "emotional memories decay slower"],
            properties={"shape": "exponential", "configurable_half_life": True},
            bridges=["memory", "decay", "spaced-repetition", "episodic-memory"],
            tags=["memory", "learning", "psychology"])

        ns.define("consolidation",
            "Transfer from short-term to long-term memory during rest",
            description="During rest/sleep, the brain replays important experiences and transfers them to durable storage. In the fleet, the Rest instinct generates ATP while the memory system consolidates episodic memories into semantic knowledge. Without rest, agents accumulate unprocessed experiences that crowd working memory.",
            level=Level.BEHAVIOR,
            examples=["studying before sleep improves retention", "taking breaks between learning sessions", "agent rests -> episodes consolidate -> semantic memory grows"],
            bridges=["rest", "episodic-memory", "semantic-memory", "circadian-rhythm"],
            tags=["memory", "biology", "learning", "fleet"])

        ns.define("rehearsal",
            "Active recall of a memory to strengthen it and reset its decay timer",
            description="Each time you recall something, you're not just retrieving it — you're rewriting it stronger. The fleet uses this: revisiting a lesson or experience resets its forgetting-curve timer. Spaced rehearsal (recalling at increasing intervals) is more effective than cramming.",
            level=Level.PATTERN,
            examples=["flashcard review resets the forgetting curve", "explaining a concept to someone else strengthens your memory of it", "agent reviews failed deliberation to learn from it"],
            bridges=["forgetting-curve", "consolidation", "learning", "spaced-repetition"],
            tags=["memory", "learning"])

        ns.define("chunking",
            "Grouping individual items into larger meaningful units to expand effective capacity",
            description="Humans remember 7±2 items. But by chunking (grouping), we effectively remember more: 'F-B-I-C-I-A-N-S-A' is 9 letters (hard) but 'FBI-CIA-NSA' is 3 chunks (easy). Agents chunk experiences into episodes, episodes into patterns, patterns into skills.",
            level=Level.PATTERN,
            examples=["phone number 555-1234 as two chunks not seven digits", "a 'trip to the store' is one chunk containing many sub-events", "skill = chunk of related procedural memories"],
            bridges=["working-memory", "abstraction", "hierarchy", "pattern"],
            tags=["memory", "cognition", "abstraction"])

    def _load_coordination(self):
        ns = self.add_namespace("coordination",
            "How multiple agents work together — or fail to")

        ns.define("stigmergy",
            "Indirect coordination through environment modification",
            description="Ants don't talk to each other. They leave pheromone trails. Other ants follow strong trails and reinforce them. The environment IS the communication channel. In the fleet, cuda-stigmergy implements this: agents modify a shared field, other agents read and react to it.",
            level=Level.BEHAVIOR,
            examples=["ant trails", "wikipedia edits (each edit is a trace others build on)", "git commits (each commit is a trace the next developer reads)", "agents leaving 'marks' on a shared field that other agents follow"],
            properties={"direct": False, "medium": "environment", "scalability": "excellent"},
            bridges=["gossip", "consensus", "broadcast", "swarm", "tuplespace"],
            tags=["swarm", "decentralized", "scalable", "fleet"])

        ns.define("consensus",
            "Agreement among agents on a shared state or decision",
            description="Reaching agreement when agents have different information and preferences. The fleet uses threshold-based consensus: proposals need 0.85 confidence from participating agents. Below that, the proposal is forfeited. This prevents acting on uncertain agreements.",
            level=Level.BEHAVIOR,
            examples=["raft protocol elects a leader", "jury reaches unanimous verdict", "fleet agrees on navigation plan with 0.92 confidence"],
            bridges=["deliberation", "voting", "agreement", "quorum"],
            tags=["coordination", "distributed", "fleet"])

        ns.define("deliberation",
            "Structured consideration of options leading to a decision",
            description="Not just thinking — structured thinking. The Consider/Resolve/Forfeit protocol: generate options (Consider), evaluate and select (Resolve), abandon deadlocks (Forfeit). Prevents both impulsiveness and analysis paralysis.",
            level=Level.BEHAVIOR,
            examples=["jury deliberation", "design review meeting", "agent evaluates 3 paths and selects the one with highest confidence"],
            properties={"protocol": "consider-resolve-forfeit", "threshold": 0.85},
            bridges=["consensus", "decision-making", "convergence", "filtration"],
            tags=["cognition", "coordination", "fleet"])

        ns.define("gossip",
            "Agents sharing information with random neighbors, spreading knowledge through the network",
            description="Like a rumor spreading through a crowd. Each agent tells a few neighbors, who tell a few more, and eventually everyone knows. In the fleet, gossip is used for trust propagation: agents share trust assessments with neighbors, and trust spreads organically.",
            level=Level.PATTERN,
            examples=["epidemic information dissemination", "trust scores spreading through fleet", "discovery protocols finding new agents"],
            bridges=["stigmergy", "broadcast", "consensus", "trust"],
            tags=["coordination", "distributed", "scalable"])

        ns.define("swarm",
            "Collective behavior emerging from simple local rules without central control",
            description="Birds flock. Fish school. No leader tells them where to go. Each individual follows 3 rules: separation (don't crowd), alignment (match neighbors' direction), cohesion (move toward center). The fleet uses this for fleet coordination: each agent follows simple rules, complex behavior emerges.",
            level=Level.BEHAVIOR,
            examples=["bird flocking", "ant colony optimization", "fleet agents self-organizing around a task without central command"],
            bridges=["stigmergy", "emergence", "consensus", "decentralized"],
            tags=["swarm", "decentralized", "emergence", "fleet"])

        ns.define("emergence",
            "Complex global behavior arising from simple local interactions",
            description="The whole is greater than the sum of its parts. No individual agent knows the big picture, but together they exhibit intelligence that none possess alone. The fleet's cuda-emergence crate detects emergent patterns using Welford's online algorithm for baseline tracking.",
            level=Level.META,
            examples=["consciousness from neurons", "traffic jams from individual driving decisions", "fleet discovers an optimal division of labor nobody explicitly planned"],
            properties={"detection": "welford-baseline", "types": "8-pattern-types"},
            bridges=["swarm", "stigmergy", "self-organization", "complexity"],
            tags=["meta", "swarm", "complexity", "fleet"])

        ns.define("quorum",
            "Minimum number of agents required for a decision to be valid",
            description="A decision made by 2 out of 100 agents isn't representative. Quorum ensures enough agents participate. In distributed systems, quorum is often majority (N/2 + 1). In Byzantine systems, it's 3f+1 (where f is the number of faulty agents).",
            level=Level.PATTERN,
            examples=["majority vote needs quorum of 51%", "byzantine fault tolerance needs 3f+1 agents", "fleet deliberation requires minimum 3 participants"],
            bridges=["consensus", "voting", "byzantine", "election"],
            tags=["coordination", "distributed", "fault-tolerance"])

        ns.define("leader-election",
            "Process of selecting a coordinator from a group of peers",
            description="When all agents are equal, sometimes one needs to lead. Leader election (cuda-election) uses a Raft-like protocol: agents have terms, candidates request votes, majority wins. Leaders send heartbeats. If heartbeat fails, new election.",
            level=Level.PATTERN,
            examples=["raft protocol", "bully algorithm", "fleet elects a task coordinator for the current mission"],
            bridges=["quorum", "consensus", "heartbeat", "fault-tolerance"],
            tags=["coordination", "distributed", "fault-tolerance", "fleet"])

    def _load_learning(self):
        ns = self.add_namespace("learning",
            "How agents improve through experience")

        ns.define("exploration",
            "Trying new actions to discover potentially better strategies",
            description="The agent deliberately chooses suboptimal actions to learn about the environment. Without exploration, agents get stuck in local optima. The explore/exploit tradeoff is fundamental: explore too much and you waste resources, exploit too much and you never discover better options.",
            level=Level.BEHAVIOR,
            examples=["epsilon-greedy: 10% of the time, pick randomly", "curiosity-driven: seek surprising states", "trying a new restaurant instead of the usual one"],
            bridges=["exploitation", "curiosity", "entropy", "discovery"],
            antonyms=["exploitation"],
            tags=["learning", "reinforcement", "agent-behavior"])

        ns.define("exploitation",
            "Using currently known best actions to maximize reward",
            description="The safe choice. Use what works. But if you only exploit, you never discover that a better option exists. The fleet balances this with cuda-adaptation's strategy switching.",
            level=Level.BEHAVIOR,
            examples=["always taking the shortest known path", "using the proven sorting algorithm", "going to your favorite restaurant every time"],
            bridges=["exploration", "optimization", "convergence", "habit"],
            antonyms=["exploration"],
            tags=["learning", "reinforcement", "optimization"])

        ns.define("credit-assignment",
            "Determining which action caused an outcome when many actions contribute",
            description="The hardest problem in learning. You tried path A, used sensor 2, and asked navigator for help. The task succeeded. Which of those caused the success? Temporal credit assignment (which past action matters) and structural credit assignment (which part of the agent mattered) are both unsolved in general.",
            level=Level.META,
            examples=["was it the new sensor or the better path that improved accuracy?", "which weight change in the neural network caused the improvement?", "which team member's contribution was most valuable?"],
            bridges=["learning", "causality", "attribution", "provenance"],
            tags=["learning", "meta", "causality"])

        ns.define("transfer-learning",
            "Applying knowledge from one domain to a different but related domain",
            description="Learning to ride a bicycle helps you learn to ride a motorcycle. The fleet implements this through gene pool sharing (cuda-genepool): successful behavioral patterns in one agent can be adopted by another agent in a different context.",
            level=Level.PATTERN,
            examples=["learning Python helps learn Rust", "spatial reasoning transfers between indoor and outdoor navigation", "an agent's pathfinding skill improves its route-planning skill"],
            bridges=["generalization", "abstraction", "analogy", "genepool"],
            tags=["learning", "generalization"])

        ns.define("curriculum",
            "Structured sequence of learning tasks from easy to hard",
            description="You don't learn calculus before algebra. The fleet's cuda-learning implements curriculum ordering: start with high-confidence, simple tasks. Only advance when current level is mastered. This dramatically speeds learning compared to random task ordering.",
            level=Level.PATTERN,
            examples=["math: arithmetic -> algebra -> calculus", "driving: parking lot -> residential -> highway", "agent: navigate empty room -> navigate with obstacles -> navigate with moving obstacles"],
            bridges=["skill", "learning-rate", "scaffolding", "progression"],
            tags=["learning", "education", "skill"])

        ns.define("spaced-repetition",
            "Reviewing material at increasing intervals to maximize retention",
            description="Review after 1 day, then 3 days, then 7, then 21. Each successful recall extends the interval. Failed recall resets to a short interval. This is the most effective known learning technique for long-term retention. The fleet's forgetting-curve implementation supports this natively.",
            level=Level.PATTERN,
            examples=["flashcard apps like Anki", "reviewing code after 1 day, 3 days, 1 week", "agent reviews past lessons at expanding intervals"],
            bridges=["forgetting-curve", "rehearsal", "consolidation", "memory"],
            tags=["learning", "memory", "psychology"])

        ns.define("overfitting",
            "Learning the training examples too well, failing on new situations",
            description="The agent memorizes instead of generalizes. It gets 100% on practice problems but 50% on real problems. In the fleet, gene auto-quarantine prevents this: genes that work perfectly in one context but fail in others get quarantined.",
            level=Level.BEHAVIOR,
            examples=["student memorizes exam answers but can't apply concepts", "model achieves 99% training accuracy but 60% test accuracy", "agent perfectly navigates training maze but fails on slightly different maze"],
            bridges=["generalization", "regularization", "robustness", "quarantine"],
            antonyms=["generalization"],
            tags=["learning", "statistics", "failure-mode"])

    def _load_biological(self):
        ns = self.add_namespace("biological",
            "Biological metaphors made precise — instincts, energy, neurotransmitters")

        ns.define("instinct",
            "Inherited behavioral program that drives action without reasoning",
            description="The agent doesn't choose to perceive. Instinct makes perception happen. Instincts are the first layer of behavior — before any deliberation, before any learning, the instinct engine generates ATP and drives the agent to survive, perceive, navigate, communicate, learn, and defend. In cuda-genepool, 10 instincts with priorities and energy costs.",
            level=Level.DOMAIN,
            examples=["newborn reflexes: suckling, grasping", "agent automatically avoids obstacles before deliberating about path", "fight-or-flight response fires before conscious thought"],
            properties={"inherited": True, "priority_10": "survive", "priority_1": "rest"},
            bridges=["reflex", "energy", "mitochondrion", "opcode"],
            tags=["biology", "fleet-foundation", "agent-behavior"])

        ns.define("apoptosis",
            "Programmed cell death — graceful self-termination when fitness drops below threshold",
            description="When an agent's fitness drops below 0.1 for 10 consecutive cycles, the apoptosis protocol triggers. This isn't a crash — it's a deliberate, clean shutdown that releases resources back to the fleet. In biology, apoptosis is essential for development (webbed fingers die) and health (cancer cells should undergo apoptosis).",
            level=Level.DOMAIN,
            examples=["tail disappears in frog development", "damaged cells self-destruct to prevent cancer", "agent with failing sensors gracefully shuts down and reports to fleet"],
            properties={"fitness_threshold": 0.1, "patience": "10 ticks", "graceful": True},
            bridges=["shutdown", "graceful-degradation", "fitness", "resource-release"],
            tags=["biology", "safety", "fleet"])

        ns.define("homeostasis",
            "Maintenance of stable internal conditions despite external changes",
            description="Body temperature stays at 37°C whether it's 0°C or 40°C outside. The fleet's energy system maintains homeostasis: ATP budget stays balanced despite varying task loads. Confidence stays calibrated despite varying difficulty. The agent adapts its behavior to maintain internal stability.",
            level=Level.DOMAIN,
            examples=["thermoregulation", "blood pH maintained at 7.4", "agent adjusts deliberation depth based on available energy"],
            bridges=["feedback-loop", "adaptation", "setpoint", "circadian-rhythm"],
            tags=["biology", "control", "stability"])

        ns.define("circadian-rhythm",
            "Time-based modulation of behavior and capability following a ~24-hour cycle",
            description="The fleet's cuda-energy implements circadian modulation via cosine function. Navigate instinct peaks at noon. Rest instinct peaks at 2 AM. The modulation has a floor of 0.1 — no instinct ever goes completely silent. In agents, this means different tasks are more efficient at different times.",
            level=Level.PATTERN,
            examples=["humans alert at 10am, drowsy at 3am", "agent's navigation accuracy peaks midday, communication peaks evening", "cosine modulation: strength = 0.55 + 0.45 * cos(2π * (hour - peak) / 24)"],
            properties={"function": "cosine", "period": "24 hours", "floor": 0.1},
            bridges=["energy", "instinct", "homeostasis", "scheduling"],
            tags=["biology", "temporal", "fleet"])

        ns.define("neurotransmitter",
            "Chemical signal that modulates neural activity — the fleet's confidence amplifier",
            description="Dopamine IS confidence. Serotonin IS trust. Norepinephrine IS alertness. These aren't metaphors — they're the same mathematical structures. The fleet's cuda-neurotransmitter implements 8 types with receptor down-regulation (sensitivity decreases after repeated activation) and Hebbian synapses (neurons that fire together wire together).",
            level=Level.DOMAIN,
            examples=["dopamine spike when prediction confirmed = confidence boost", "serotonin builds with social bonding = trust accumulation", "norepinephrine fires on threat = immediate alert"],
            properties={"types": 8, "down_regulation": True, "hebbian": True},
            bridges=["confidence", "trust", "attention", "emotion"],
            tags=["biology", "cognition", "fleet"])

        ns.define("membrane",
            "Self/other boundary that filters what enters and leaves the agent",
            description="Cell membranes determine what gets in and what stays out. The fleet's membrane (cuda-genepool) has antibodies that block dangerous signals: 'rm -rf', 'format', 'drop_all' are rejected at the boundary. The membrane is the agent's first line of defense — before any reasoning about whether something is dangerous.",
            level=Level.DOMAIN,
            examples=["cell membrane with selective permeability", "firewall blocking dangerous packets", "agent's membrane blocks self-destruct commands before they reach deliberation"],
            bridges=["security", "sandbox", "filter", "boundary"],
            tags=["biology", "security", "fleet"])

        ns.define("enzyme",
            "Catalyst that converts environmental signals into genetic activation",
            description="In the fleet pipeline: Environment -> Sensors -> Membrane -> Enzymes -> Genes. Enzymes are the bridge between perception and action. They detect specific signal patterns (e.g., 'high temperature' or 'low battery') and activate corresponding genes (e.g., 'reduce activity' or 'seek energy'). Without enzymes, the agent perceives but doesn't act.",
            level=Level.PATTERN,
            examples=["lactase enzyme converts lactose into absorbable sugars", "sensor reads 'low ATP' -> enzyme activates 'rest' gene", "pattern matcher in deliberation that triggers emergency protocol"],
            bridges=["instinct", "perception", "gene-activation", "signal-processing"],
            tags=["biology", "pipeline", "fleet"])

        ns.define("hebbian-learning",
            "Synapses strengthen when pre- and post-synaptic neurons fire together",
            description="'Neurons that fire together wire together.' If sensor A consistently fires right before successful action B, the A->B synapse strengthens. If A fires but B doesn't follow, the synapse weakens. This is the simplest form of credit assignment: temporal correlation implies causation.",
            level=Level.PATTERN,
            examples=["pavlovian conditioning: bell + food = bell causes salivation", "sensor detects obstacle right before collision -> sensor-obstacle association strengthens", "learning that asking navigator before pathfinding improves outcomes"],
            bridges=["learning", "credit-assignment", "synapse", "correlation"],
            tags=["biology", "learning", "neuroscience"])

    def _load_architecture(self):
        ns = self.add_namespace("architecture",
            "Software architecture patterns and structures")

        ns.define("actor-model",
            "Concurrency model where each agent is an isolated entity communicating via messages",
            description="Each actor has its own state. No shared memory. All communication through asynchronous messages. If one actor crashes, others continue. The fleet's cuda-actor implements this with mailboxes, supervision strategies, and spawn hierarchies.",
            level=Level.PATTERN,
            examples=["Erlang processes", "Akka actors", "each fleet agent is an actor with a mailbox"],
            properties={"isolation": True, "async": True, "supervision": True},
            bridges=["agent", "mailbox", "concurrency", "fault-tolerance"],
            tags=["architecture", "concurrency", "fleet"])

        ns.define("circuit-breaker",
            "Prevent cascading failures by stopping calls to a failing service",
            description="Like an electrical circuit breaker: if too much current flows (too many failures), it trips open and stops all calls. After a cooldown, it allows a few test calls (half-open). If those succeed, it closes again. The fleet's cuda-circuit implements this with configurable thresholds.",
            level=Level.PATTERN,
            examples=["Netflix Hystrix", "stop calling an API that's returning 500 errors", "agent stops querying a sensor that's been noisy for 30 seconds"],
            bridges=["fault-tolerance", "bulkhead", "backpressure", "graceful-degradation"],
            tags=["resilience", "pattern", "fleet"])

        ns.define("bulkhead",
            "Isolate components so one failure doesn't take down the whole system",
            description="Ship bulkheads: if one compartment floods, the others stay dry. In software: if one service fails, others continue because they have separate resource pools. The fleet's cuda-resilience combines circuit-breaker, rate-limiter, and bulkhead into a ResilienceShield.",
            level=Level.PATTERN,
            examples=["ship compartments", "thread pools per service", "each agent has its own energy budget — one agent's exhaustion doesn't affect others"],
            bridges=["circuit-breaker", "isolation", "fault-tolerance", "resource-pool"],
            tags=["resilience", "pattern", "fleet"])

        ns.define("event-sourcing",
            "Store every state change as an immutable event, reconstruct state by replaying",
            description="Instead of storing current state, store the sequence of events that led to it. Current state = replay all events. This gives full audit trail, time-travel debugging, and the ability to rebuild state from scratch. The fleet's cuda-persistence supports event-sourced snapshots.",
            level=Level.PATTERN,
            examples=["git history = event-sourced code state", "bank ledger = event-sourced balance", "agent's decision history = event-sourced mental state"],
            bridges=["provenance", "audit-trail", "persistence", "immutable"],
            tags=["architecture", "persistence", "audit"])

        ns.define("state-machine",
            "Model behavior as a finite set of states with defined transitions",
            description="An agent can be in exactly one state at a time: Idle, Navigating, Deliberating, Communicating, etc. Transitions between states have guards (conditions) and actions (side effects). The fleet's cuda-state-machine supports hierarchical states, guard evaluation, and state history.",
            level=Level.PATTERN,
            examples=["traffic light: red -> green -> yellow -> red", "agent: idle -> navigating -> arrived -> idle", "TCP: closed -> syn-sent -> established -> fin-wait -> closed"],
            bridges=["workflow", "lifecycle", "guard", "transition"],
            tags=["architecture", "modeling", "fleet"])

        ns.define("backpressure",
            "Signal to slow down when the consumer can't keep up with the producer",
            description="If a fast producer sends messages to a slow consumer, the consumer's queue grows without bound and eventually crashes. Backpressure tells the producer to slow down or stop. The fleet's cuda-backpressure implements credit-based flow control, window-based control, and adaptive rate control (AIMD).",
            level=Level.PATTERN,
            examples=["TCP flow control", "tell a fast sensor to sample less frequently", "fleet coordinator slows task assignment when agents are overloaded"],
            bridges=["flow-control", "throttle", "rate-limit", "congestion"],
            tags=["architecture", "resilience", "fleet"])

        ns.define("sidecar",
            "Separate helper process attached to a primary component for cross-cutting concerns",
            description="Instead of embedding logging, monitoring, and security into every component, run them as separate sidecar processes. The main component talks to the sidecar via local network. The fleet's cuda-metrics and cuda-logging effectively serve as sidecars.",
            level=Level.CONCRETE,
            examples=["Envoy proxy alongside a microservice", "logging agent alongside a navigation agent", "health monitor watching a computation agent"],
            bridges=["monitoring", "logging", "proxy", "separation-of-concerns"],
            tags=["architecture", "pattern", "operations"])

    def _load_spatial(self):
        ns = self.add_namespace("spatial",
            "How agents understand and navigate physical and abstract space")

        ns.define("attention-tile",
            "A rectangular region of an attention matrix that is computed (or skipped) as a unit",
            description="Instead of computing attention for every (query, key) pair, divide the matrix into tiles and only compute the important ones. Ghost tiles are the ones skipped — they're 'ghosts' in the matrix, present logically but computationally absent. This is the core idea of cuda-ghost-tiles.",
            level=Level.CONCRETE,
            examples=["8x8 tile in a 64x64 attention matrix", "skip the bottom-left tile because past tokens rarely attend to future tokens", "GPU thread block = one attention tile"],
            bridges=["sparsity", "pruning", "attention", "gpu-optimization"],
            tags=["spatial", "optimization", "gpu"])

        ns.define("spatial-hash",
            "Hash-based spatial lookup that avoids hierarchical structures",
            description="Divide space into a grid, hash each cell to a bucket. Look up nearby objects by checking the same and neighboring buckets. O(1) lookup vs O(log n) for trees. The fleet uses spatial hashing for fast collision detection and neighbor queries.",
            level=Level.PATTERN,
            examples=["grid-based collision detection in games", "finding nearby agents without checking all agents", "uniform grid spatial hashing"],
            bridges=["grid", "hash", "neighbor-query", "collision-detection"],
            tags=["spatial", "data-structure", "optimization"])

        ns.define("manhattan-distance",
            "Distance measured along grid axes (|dx| + |dy|) — the city block metric",
            description="In a grid world, you can't move diagonally (or diagonal costs more). Manhattan distance measures the actual path length. The fleet uses it for A* pathfinding in grid environments because it's the true minimum distance.",
            level=Level.CONCRETE,
            examples=["taxicab distance in a city grid", "moving a chess rook from a1 to h8 = 14 squares", "agent navigation on a grid map"],
            bridges=["euclidean-distance", "pathfinding", "heuristic", "grid"],
            tags=["spatial", "geometry", "pathfinding"])

        ns.define("a-star",
            "Optimal pathfinding algorithm using actual cost + estimated remaining cost",
            description="f(n) = g(n) + h(n). g(n) is the actual cost from start to current node. h(n) is the heuristic estimate from current to goal. If h(n) never overestimates (admissible), A* finds the optimal path. The fleet's cuda-navigation implements A* with obstacle avoidance and path smoothing.",
            level=Level.PATTERN,
            examples=["GPS navigation", "game character pathfinding", "robot navigating a warehouse", "agent finding path through obstacle field"],
            properties={"optimal": True, "admissible_heuristic": True, "time_complexity": "O(b^d)"},
            bridges=["pathfinding", "heuristic", "manhattan-distance", "navigation"],
            tags=["spatial", "algorithm", "pathfinding", "fleet"])

    def _load_temporal(self):
        ns = self.add_namespace("temporal",
            "Time, scheduling, deadlines, and temporal reasoning")

        ns.define("deadline-urgency",
            "A value that increases as a deadline approaches, modulating agent behavior",
            description="Not all deadlines are equal. A task due in 5 minutes has urgency ~1.0. A task due in 5 days has urgency ~0.1. But urgency isn't linear — it accelerates as the deadline nears. The fleet's cuda-temporal uses this to prioritize: high-urgency tasks preempt low-urgency ones.",
            level=Level.PATTERN,
            examples=["deadline in 1 hour: urgency 0.9, agent drops everything else", "deadline in 1 week: urgency 0.2, agent works on it when convenient", "past deadline: urgency 1.0, agent enters emergency mode"],
            bridges=["priority", "scheduling", "preemption", "time-pressure"],
            tags=["temporal", "scheduling", "fleet"])

        ns.define("causal-chain",
            "A sequence of events where each causes the next",
            description="A -> B -> C -> D. If A didn't happen, D wouldn't have happened. Causal chains are the backbone of provenance tracking. The fleet's cuda-provenance chains decisions: each decision records what caused it, creating an auditable chain of reasoning.",
            level=Level.PATTERN,
            examples=["domino effect", "sensor reading -> deliberation -> decision -> action -> result", "git commit chain: each commit references its parent"],
            bridges=["provenance", "causality", "audit-trail", "temporal"],
            tags=["temporal", "causality", "audit", "fleet"])

        ns.define("heartbeat",
            "Periodic signal indicating an agent is alive and healthy",
            description="In distributed systems, silence is ambiguous: is the agent dead, or just quiet? Heartbeats solve this. Regular 'I'm alive' messages. If heartbeats stop, the agent is presumed dead and its tasks are reassigned. The fleet uses heartbeats for fleet health monitoring (cuda-fleet-mesh).",
            level=Level.PATTERN,
            examples=["raft leader heartbeats", "health check pings every 30s", "watchdog timer in embedded systems"],
            bridges=["health", "fault-detection", "timeout", "leader-election"],
            tags=["temporal", "fault-tolerance", "coordination", "fleet"])

    def _load_communication(self):
        ns = self.add_namespace("communication",
            "How agents exchange information and meaning")

        ns.define("grounding",
            "Establishing shared understanding of word meanings between agents",
            description="When I say 'near', do I mean 1 meter or 10 meters? Grounding is the process of establishing that both agents mean the same thing by the same word. The fleet's cuda-communication implements this with SharedVocabulary: agents negotiate term definitions and track grounding scores.",
            level=Level.BEHAVIOR,
            examples=["two humans agreeing that 'soon' means 'within 5 minutes'", "agents negotiating that 'high priority' means 'respond within 1 second'", "establishing a shared coordinate system"],
            bridges=["vocabulary", "shared-understanding", "negotiation", "semantic-alignment"],
            tags=["communication", "language", "coordination", "fleet"])

        ns.define("speech-act",
            "An utterance that performs an action — saying is doing",
            description="Not all communication is information transfer. Some communication IS action: 'I promise to arrive by 3pm' creates an obligation. 'You're fired' changes employment status. 'I name this ship Lighthouse' creates a name. The fleet's A2A intents are speech acts: Command, Request, Warn, Apologize.",
            level=Level.DOMAIN,
            examples=["'I promise...' = commitment", "'I order you to...' = command", "'I apologize for...' = repair", "'Warning: obstacle ahead' = alert"],
            bridges=["intent", "a2a", "communication", "action"],
            tags=["communication", "language", "philosophy"])

        ns.define("information-bottleneck",
            "Compressing information to its most essential parts before transmission",
            description="Communication costs energy. Sending raw sensor data (1MB) vs sending 'obstacle at (3,5)' (20 bytes). The information bottleneck principle: keep only the information relevant to the task, discard the rest. The fleet's communication costs (cuda-communication) enforce this naturally.",
            level=Level.PATTERN,
            examples=["summarizing a 1-hour meeting in 3 bullet points", "agent sends 'path blocked at intersection' instead of full lidar scan", "compressing 1000 sensor readings into 'temperature nominal'"],
            bridges=["compression", "abstraction", "communication-cost", "attention"],
            tags=["communication", "information-theory", "optimization", "fleet"])

        ns.define("context-window",
            "The amount of recent information an agent can consider simultaneously",
            description="Like human working memory but for LLMs: a fixed-size window of tokens. The fleet faces this at multiple levels: working memory capacity, deliberation depth, message history length. Strategies: chunking, summarization, attention prioritization.",
            level=Level.CONCRETE,
            examples=["GPT's 128K token context window", "agent can hold 7 items in working memory", "conversation history limited to last 50 messages"],
            bridges=["working-memory", "attention", "chunking", "capacity"],
            tags=["communication", "cognition", "capacity"])

    def _load_security(self):
        ns = self.add_namespace("security",
            "Safety, boundaries, and trust enforcement")

        ns.define("least-privilege",
            "Give an agent only the permissions it needs, nothing more",
            description="A navigation agent doesn't need access to the communication log. A sensor agent doesn't need the ability to spawn new agents. The fleet's cuda-rbac implements role-based access with deny-override: deny rules always beat allow rules.",
            level=Level.PATTERN,
            examples=["read-only access to config files", "agent can observe but not modify", "wildcard permissions for admin, specific permissions for worker"],
            bridges=["rbac", "sandbox", "membrane", "boundary"],
            tags=["security", "principle", "fleet"])

        ns.define("sandbox",
            "Restricted execution environment that limits what an agent can do",
            description="Not just permission checks — actual resource limits. The fleet's cuda-sandbox implements: maximum compute time, maximum memory, maximum network bytes, operation rate limits. An agent can try to do something forbidden, but the sandbox prevents it from actually happening.",
            level=Level.CONCRETE,
            examples=["browser sandbox limiting JavaScript access", "container limiting CPU and memory", "agent sandbox: max 100ms compute per operation, max 50 operations per second"],
            bridges=["least-privilege", "rbac", "resource-limit", "isolation"],
            tags=["security", "isolation", "fleet"])

        ns.define("graceful-degradation",
            "Continue operating at reduced capability instead of failing completely",
            description="When things go wrong, don't crash — degrade. Lose a sensor? Use the remaining ones. Lose communication? Continue autonomously. Lose 50% compute? Do less accurate but still useful work. The fleet implements this at every level: sensors degrade, deliberation simplifies, communication compresses.",
            level=Level.BEHAVIOR,
            examples=["airplane continues flying on one engine", "agent uses 2 of 4 sensors after 2 fail", "graceful fallback from expensive model to cheap model under load"],
            bridges=["fault-tolerance", "resilience", "fallback", "circuit-breaker"],
            antonyms=["catastrophic-failure"],
            tags=["resilience", "safety", "fleet"])

    def _load_decision(self):
        ns = self.add_namespace("decision",
            "How agents make choices under uncertainty")

        ns.define("satisficing",
            "Choosing the first option that meets a threshold, not the optimal one",
            description="Optimizing is expensive. Satisficing is fast: find an option that's 'good enough' and go with it. Herbert Simon showed humans do this naturally. The fleet uses it when energy is low or time is short: instead of full deliberation, pick the first option above confidence threshold.",
            level=Level.BEHAVIOR,
            examples=["choosing a restaurant that's 'good enough' vs visiting all 50 to find the best", "agent picks first path with confidence > 0.7 instead of evaluating all 10 paths", "buying the first car that meets your requirements"],
            bridges=["deliberation", "optimization", "heuristics", "energy-conservation"],
            antonyms=["maximizing"],
            tags=["decision", "heuristics", "behavior"])

        ns.define("multi-armed-bandit",
            "Balancing exploration of unknown options against exploitation of known best",
            description="You're at a casino with 10 slot machines (arms). Each has unknown payout rate. How do you maximize winnings? Pure exploration: try all equally. Pure exploitation: play the one that's won most. Optimal: balance using algorithms like UCB (Upper Confidence Bound) or Thompson Sampling. The fleet's cuda-adaptation implements strategy switching similar to this.",
            level=Level.PATTERN,
            examples=["A/B testing: which variant gets more clicks?", "choosing which restaurant to try next", "agent deciding which navigation algorithm to use for this terrain"],
            bridges=["exploration", "exploitation", "ucb", "thompson-sampling"],
            tags=["decision", "reinforcement", "statistics"])

        ns.define("minimax",
            "Choose the action that minimizes the maximum possible loss",
            description="Assume the worst case and make it as good as possible. Chess computers use this: assume the opponent plays perfectly, and choose the move that gives the best outcome even then. The fleet doesn't use minimax directly, but the principle appears in safety reflexes: assume the worst and prepare.",
            level=Level.PATTERN,
            examples=["chess engine assuming best opponent play", "choosing the route with the best worst-case travel time", "agent planning for sensor failure during critical task"],
            bridges=["adversarial", "risk-aversion", "worst-case", "game-theory"],
            tags=["decision", "game-theory", "algorithm"])

        ns.define("paradox-of-choice",
            "More options lead to worse decisions or decision paralysis",
            description="3 jam varieties: 30% of shoppers buy. 24 jam varieties: 3% buy. Too many options overwhelm working memory and increase the cost of deliberation. The fleet's cuda-filtration implements this: limit deliberation scope to the top-N options, not all possible options.",
            level=Level.BEHAVIOR,
            examples=["menu with 500 items vs menu with 10 items", "agent freezing when presented with 1000 possible actions", "dating app fatigue from too many profiles"],
            bridges=["filtration", "deliberation", "working-memory", "overwhelm"],
            tags=["decision", "psychology", "cognition"])



    def _load_control_theory(self):
        ns = self.add_namespace("control-theory",
            "Feedback, regulation, and maintaining target states")

        ns.define("feedback-loop",
            "Output of a system is measured and used to adjust input to maintain a target",
            description="Thermostat measures temperature, compares to setpoint, turns heater on/off. Agent measures confidence, compares to threshold, adjusts deliberation depth. Negative feedback stabilizes. Positive feedback amplifies (dangerous). The fleet uses feedback loops everywhere: energy regulation, confidence calibration, trust decay.",
            level=Level.PATTERN,
            examples=["thermostat maintains 70F", "cruise control maintains 65mph", "agent adjusts exploration rate based on recent success"],
            bridges=["homeostasis", "setpoint", "pid-controller", "adaptation"],
            tags=["control", "feedback", "stability", "fleet"])

        ns.define("setpoint",
            "The target value a control system tries to maintain",
            description="The thermostat is set to 70F. That's the setpoint. If actual temperature is below setpoint, heat. If above, cool. In the fleet: confidence threshold is a setpoint (0.85 for consensus). Energy budget has a setpoint. When actual diverges from setpoint, corrective action activates.",
            level=Level.CONCRETE,
            examples=["thermostat setpoint 70F", "consensus threshold 0.85", "target speed 60mph", "desired trust level 0.7"],
            bridges=["feedback-loop", "homeostasis", "threshold", "target"],
            tags=["control", "target", "fleet"])

        ns.define("hysteresis",
            "The output depends not just on current input but on history — path dependence",
            description="A thermostat set to 70F doesn't flicker on/off at 70.0. It heats to 72, then cools to 68 before heating again. The gap prevents rapid oscillation. In the fleet, deliberation thresholds use hysteresis: once a proposal is accepted, it stays accepted even if confidence dips slightly below threshold.",
            level=Level.PATTERN,
            examples=["thermostat: heat to 72, cool to 68, not flip at 70", "Schmitt trigger in electronics", "agent proposal accepted at 0.85, stays accepted until confidence drops to 0.75"],
            bridges=["feedback-loop", "oscillation", "stability", "threshold"],
            tags=["control", "stability", "pattern"])

        ns.define("overshoot",
            "System exceeds its target before settling back — the pendulum swings past center",
            description="You brake too hard and stop short. Or brake too late and overshoot the stop line. Any control system with delay can overshoot. In the fleet: an agent that reduces exploration too aggressively may overshoot into pure exploitation, missing important discoveries.",
            level=Level.BEHAVIOR,
            examples=["pressing brake too hard", "stock price correction going below fair value", "agent switches from 50% exploration to 0% exploration overnight"],
            bridges=["feedback-loop", "oscillation", "adaptation", "correction"],
            tags=["control", "behavior", "failure-mode"])

        ns.define("dead-zone",
            "Range of inputs that produce no output — intentional insensitivity",
            description="A joystick with a dead zone: small movements do nothing. Prevents noise from causing unwanted action. In the fleet: small confidence changes below 0.05 are ignored. Small trust changes below 0.02 don't trigger reputation updates. This prevents agents from overreacting to noise.",
            level=Level.CONCRETE,
            examples=["joystick dead zone prevents drift", "sensor noise below threshold ignored", "confidence change from 0.80 to 0.81 doesn't trigger deliberation review"],
            bridges=["hysteresis", "threshold", "noise-filtering", "robustness"],
            tags=["control", "noise", "robustness"])

    def _load_evolution(self):
        ns = self.add_namespace("evolution",
            "Evolutionary dynamics — selection, drift, speciation, co-evolution")

        ns.define("natural-selection",
            "Differential survival and reproduction based on fitness",
            description="Organisms better suited to their environment survive and reproduce more. Their traits spread. The fleet implements this via cuda-genepool: genes with high fitness are shared across the gene pool, genes with low fitness are quarantined. Selection pressure comes from the environment (task success/failure).",
            level=Level.DOMAIN,
            examples=["giraffe necks lengthen because taller giraffes reach more food", "gene with 0.8 fitness spreads; gene with 0.1 fitness quarantined", "navigation strategy that finds paths faster gets selected over slower one"],
            bridges=["fitness-landscape", "genetic-drift", "mutation", "adaptation"],
            tags=["evolution", "biology", "fleet"])

        ns.define("fitness-landscape",
            "Multi-dimensional space where each position represents a strategy and height represents fitness",
            description="Imagine a mountainous terrain. Each point is a possible behavior. Height is how well that behavior works. The agent climbs uphill (improves fitness). But it might get stuck on a local peak when a taller peak exists across a valley. The fleet uses fitness landscapes to understand why agents get stuck and how to escape.",
            level=Level.DOMAIN,
            examples=["evolution climbs fitness peaks", "agent stuck in local optimum of 'always exploit'", "adding noise (mutation) lets agent jump across valleys to taller peaks"],
            bridges=["local-minimum", "exploration", "mutation", "natural-selection"],
            tags=["evolution", "optimization", "visualization"])

        ns.define("punctuated-equilibrium",
            "Long periods of stability interrupted by sudden rapid change",
            description="Evolution isn't always gradual. Species stay stable for millions of years, then suddenly diversify after a disruption. In the fleet: an agent may run the same strategy for thousands of cycles (equilibrium), then a major environmental change forces rapid adaptation (punctuation).",
            level=Level.BEHAVIOR,
            examples=["Cambrian explosion", "agent runs strategy X for weeks, then sensor fails and it must completely restructure behavior", "technology disruption forces sudden industry change"],
            bridges=["evolution", "stability", "disruption", "adaptation"],
            tags=["evolution", "pattern", "disruption"])

        ns.define("genetic-drift",
            "Random changes in gene frequency unrelated to fitness — noise in evolution",
            description="Not all changes are adaptive. Some spread by random chance. In a small population, drift is stronger — a single agent's random mutation can spread through a tiny fleet. The fleet's gene pool is susceptible to drift when fleet size is small.",
            level=Level.BEHAVIOR,
            examples=["neutral mutation spreading in small population", "fleet of 3 agents: one agent's random behavioral quirk spreads to others", "founder effect: new colony has different gene frequencies than parent"],
            bridges=["natural-selection", "noise", "population-size", "random-walk"],
            tags=["evolution", "noise", "population"])

        ns.define("co-evolution",
            "Two species evolve in response to each other — arms races and mutualisms",
            description="Predator gets faster, prey gets faster. Flower evolves deeper tube, bee evolves longer tongue. In the fleet: attacker agents evolve better strategies, defender agents evolve better defenses. Neither can stop improving because the other keeps changing. cuda-compliance co-evolves with potential threats.",
            level=Level.META,
            examples=["predator-prey arms race", "virus-antivirus co-evolution", "adversarial-red-team vs compliance-engine arms race"],
            bridges=["natural-selection", "competition", "arms-race", "adaptation"],
            tags=["evolution", "meta", "competition", "fleet"])

        ns.define("speciation",
            "Divergence into separate species when populations face different selective pressures",
            description="One species splits into two when sub-populations experience different environments. In the fleet: agents that work in different domains (indoor navigation vs outdoor) may evolve specialized strategies that are no longer interchangeable. cuda-playbook captures domain-specific strategies.",
            level=Level.BEHAVIOR,
            examples=["Darwin's finches on different Galapagos islands", "warehouse agent vs outdoor agent developing incompatible navigation strategies", "generalist agent splitting into specialist sub-agents"],
            bridges=["niche", "divergence", "specialization", "adaptation"],
            tags=["evolution", "diversity", "specialization"])

    def _load_networks(self):
        ns = self.add_namespace("networks",
            "Graph structures, connectivity patterns, and network effects")

        ns.define("small-world",
            "Network where most nodes are locally connected but any two nodes are reachable in few hops",
            description="Six degrees of separation. Your friends know your friends' friends, but through a few long-range connections, you can reach anyone. The fleet's mesh network is small-world: agents primarily coordinate with neighbors but can reach any agent through short relay chains.",
            level=Level.DOMAIN,
            examples=["social networks: six degrees of separation", "neural networks: mostly local connections, few long-range", "fleet mesh: agents gossip with neighbors, information reaches whole fleet in ~5 hops"],
            bridges=["gossip", "scale-free", "clustering", "fleet-mesh"],
            tags=["networks", "social", "fleet"])

        ns.define("scale-free",
            "Network where degree distribution follows a power law — few hubs, many leaves",
            description="Most nodes have few connections. A few nodes have enormous numbers of connections (hubs). The internet is scale-free: most sites have few links, Google has billions. In the fleet, some agents become hubs (fleet coordinators, navigators) while most are leaves.",
            level=Level.DOMAIN,
            examples=["internet: few sites with billions of links", "airline network: few hub airports, many spoke airports", "fleet: coordinator agent talks to 50 agents, worker agents talk to 3"],
            bridges=["hub", "small-world", "power-law", "robustness"],
            tags=["networks", "structure", "statistics"])

        ns.define("hub",
            "A node with disproportionately many connections in a network",
            description="Remove a random node, the network survives. Remove a hub, the network fragments. In the fleet: the captain agent is a hub. If it goes down, the whole fleet loses coordination. This is why the fleet needs redundancy and leader election (cuda-election).",
            level=Level.CONCRETE,
            examples=["airport hub: O'Hare connects to 200+ destinations", "Google: linked by billions of pages", "fleet captain: communicates with every agent"],
            bridges=["scale-free", "single-point-of-failure", "leader-election", "redundancy"],
            tags=["networks", "critical", "vulnerability"])

        ns.define("percolation",
            "Phase transition in connectivity: at a critical density, a giant connected component forms",
            description="Pour water on coffee grounds. At low density, water trickles through isolated paths. At a critical density, suddenly water flows freely through the entire grounds. Same in networks: below critical connection density, information can't spread. Above it, it spreads everywhere instantly. The fleet monitors percolation to ensure information can reach all agents.",
            level=Level.META,
            examples=["water through coffee grounds", "forest fire spreading when tree density exceeds threshold", "fleet information spreading when enough agents are connected", "disease outbreak at critical infection rate"],
            bridges=["phase-transition", "critical-mass", "cascade-failure", "connectivity"],
            tags=["networks", "phase-transition", "criticality"])

        ns.define("cascade-failure",
            "Failure of one node triggers failures in dependent nodes, spreading through the network",
            description="Power grid: one transformer fails, load redistributes to neighbors, they overload and fail, cascading across the whole grid. In the fleet: one agent fails, its tasks redistribute to neighbors, they become overloaded and fail. Circuit breakers (cuda-circuit) and bulkheads (cuda-resilience) prevent cascades by isolating failures.",
            level=Level.BEHAVIOR,
            examples=["2003 Northeast blackout", "bank run: one bank fails, depositors panic, other banks fail", "fleet: overloaded agent crashes, task redistribution overloads neighbors"],
            bridges=["circuit-breaker", "bulkhead", "single-point-of-failure", "robustness"],
            antonyms=["isolation", "containment"],
            tags=["networks", "failure-mode", "critical", "fleet"])

        ns.define("clustering-coefficient",
            "How likely two neighbors of a node are also neighbors of each other",
            description="In a friend group, are your friends also friends with each other? High clustering = tight groups. Low clustering = loose connections. The fleet's fleet-mesh uses clustering to detect sub-groups that form organically around tasks.",
            level=Level.CONCRETE,
            examples=["friend group: your friends know each other", "work team: tight cluster within larger organization", "fleet: navigation agents cluster together, communication agents cluster together"],
            bridges=["small-world", "community", "group-formation", "topology"],
            tags=["networks", "metric", "social"])

    def _load_game_theory(self):
        ns = self.add_namespace("game-theory",
            "Strategic interaction between rational (and irrational) agents")

        ns.define("nash-equilibrium",
            "A state where no agent can improve by changing strategy alone, assuming others stay",
            description="Everyone's stuck. Any individual changing their move makes them worse off. But the group might all be better off if they ALL changed. Prisoner's dilemma: both defect is a Nash equilibrium, but both cooperate would be better for everyone. The fleet uses Nash equilibria to predict stable fleet configurations.",
            level=Level.DOMAIN,
            examples=["prisoner's dilemma: both stay silent would be better, but both confess", "traffic: everyone driving is equilibrium, public transit would be better for all", "agents all exploiting is equilibrium, some exploring would be better for fleet"],
            bridges=["prisoners-dilemma", "mechanism-design", "equilibrium", "cooperation"],
            tags=["game-theory", "equilibrium", "strategy"])

        ns.define("prisoners-dilemma",
            "Two agents each choose to cooperate or defect; individual incentive conflicts with group welfare",
            description="The canonical game theory problem. If both cooperate, both get moderate reward. If one defects while other cooperates, defector gets high reward. If both defect, both get low reward. The fleet faces this constantly: share information (cooperate) vs hoard information (defect). cuda-social implements cooperation strategies (Tit-for-Tat, GenerousTat, Pavlov).",
            level=Level.DOMAIN,
            examples=["two suspects interrogated separately", "arms race: both build weapons (defect) vs both disarm (cooperate)", "agents sharing vs hoarding sensor data"],
            bridges=["nash-equilibrium", "tit-for-tat", "cooperation", "tragedy-of-commons"],
            tags=["game-theory", "social-dilemma", "cooperation"])

        ns.define("mechanism-design",
            "Designing rules of a game so that agents' self-interest produces desired outcomes",
            description="Instead of analyzing a game, you DESIGN the game. Set the rules so that rational agents doing what's best for themselves also produce what's best for the system. The fleet's incentive structures (energy costs for communication, reputation for trust) are mechanism design: agents conserve energy (self-interest) which also prevents spam (system welfare).",
            level=Level.META,
            examples=["auction design: Vickrey auction makes truthful bidding optimal", "fleet energy costs: self-interest (conserve ATP) aligns with system (prevent spam)", "carbon credits: self-interest (minimize cost) aligns with system (reduce emissions)"],
            bridges=["nash-equilibrium", "incentive-alignment", "game-rules", "economics"],
            tags=["game-theory", "design", "meta", "economics"])

        ns.define("tragedy-of-commons",
            "Shared resource depleted by individual agents acting in self-interest",
            description="Common grazing land: each herder adds one more sheep because they gain the full benefit but share the cost with everyone. Result: overgrazing and collapse. In the fleet: shared compute budget is a commons. If every agent maximizes its own usage, the budget exhausts and everyone suffers. Fleet energy budgets (cuda-energy) prevent this.",
            level=Level.DOMAIN,
            examples=["overfishing", "climate change: each country benefits from cheap energy, costs shared globally", "fleet: agents all requesting maximum compute budget", "open office: everyone talks loudly, nobody can focus"],
            bridges=["nash-equilibrium", "resource-allocation", "energy-budget", "mechanism-design"],
            tags=["game-theory", "economics", "resource", "failure-mode"])

        ns.define("zero-sum",
            "One agent's gain is exactly another agent's loss — the pie doesn't grow",
            description="Chess: if I win, you lose. The total utility is constant. Most real situations are NOT zero-sum, but agents often treat them as if they are (leading to unnecessary competition). The fleet recognizes non-zero-sum: sharing information grows the pie for everyone.",
            level=Level.DOMAIN,
            examples=["chess, poker", "negotiation framed as win/lose instead of win/win", "agents treating shared resources as competitive instead of cooperative"],
            bridges=["nash-equilibrium", "cooperation", "competition", "resource"],
            antonyms=["positive-sum", "win-win"],
            tags=["game-theory", "economics", "strategy"])

    def _load_optimization(self):
        ns = self.add_namespace("optimization",
            "Finding the best solution from a space of possibilities")

        ns.define("gradient-descent",
            "Iteratively moving in the direction of steepest improvement",
            description="Imagine standing on a foggy mountain, wanting to reach the valley. You feel the slope under your feet and step downhill. Repeat until flat. This is how neural networks learn. In the fleet, agents use gradient-descent-like strategies to improve: try a small change, if it's better, keep going that direction.",
            level=Level.PATTERN,
            examples=["neural network training", "finding minimum of a function by following negative gradient", "agent incrementally adjusts navigation strategy based on success/failure feedback"],
            bridges=["local-minimum", "learning-rate", "convergence", "hill-climbing"],
            tags=["optimization", "algorithm", "learning"])

        ns.define("local-minimum",
            "A valley that looks like the lowest point from inside, but a deeper valley exists elsewhere",
            description="The agent reaches a point where every small change makes things worse. But a large change (jumping to a different part of the landscape) might reach a much better position. This is why exploration is essential: without it, agents get trapped in local optima. Simulated annealing (cuda-adaptation) helps by occasionally accepting worse moves.",
            level=Level.DOMAIN,
            examples=["ball rolling into a small divot on a hilly surface", "always going to same restaurant (local optimum) when a better one exists across town", "agent stuck using suboptimal navigation algorithm because small tweaks don't help"],
            bridges=["fitness-landscape", "exploration", "simulated-annealing", "gradient-descent"],
            tags=["optimization", "failure-mode", "search"])

        ns.define("simulated-annealing",
            "Occasionally accept worse solutions to escape local minima, accepting worse moves less often over time",
            description="Like annealing metal: heat it up (accept random moves), slowly cool (become more selective). Early on, the agent explores widely. Over time, it settles into the best area found. Temperature parameter controls exploration: high temperature = random, low temperature = greedy. The fleet's cuda-adaptation implements strategy switching inspired by this.",
            level=Level.PATTERN,
            examples=["metal annealing: heat and slowly cool to reduce crystal defects", "traveling salesman: occasionally take a worse route to escape local optimum", "agent: early in task, try random strategies; later, stick with what works"],
            bridges=["local-minimum", "exploration", "temperature", "hill-climbing"],
            tags=["optimization", "algorithm", "search"])

        ns.define("convergence-criteria",
            "Conditions that determine when an optimization process should stop",
            description="When is the agent done improving? After 100 iterations? When improvement drops below 0.001? When confidence exceeds 0.95? Choosing the right stopping criterion prevents both premature termination (stopping too early) and wasted computation (continuing after no improvement is possible). The fleet's cuda-convergence monitors 5 convergence states.",
            level=Level.PATTERN,
            examples=["neural network: stop when loss changes less than 0.0001 for 10 epochs", "deliberation: stop when consensus exceeds 0.85", "search: stop after 1000 iterations or when best score hasn't improved in 100 iterations"],
            bridges=["convergence", "threshold", "optimization", "deliberation"],
            tags=["optimization", "stopping", "fleet"])

        ns.define("multi-objective",
            "Optimizing for multiple conflicting goals simultaneously",
            description="Fast vs accurate. Cheap vs good. Safe vs fast. You can't optimize all at once — improving one often worsens another. The result is a Pareto frontier: set of solutions where you can't improve one objective without worsening another. The fleet faces this constantly: speed vs accuracy vs energy cost.",
            level=Level.DOMAIN,
            examples=["car design: fast vs fuel-efficient vs safe vs cheap", "agent: minimize energy (fast response) vs maximize accuracy (deep deliberation)", "software: minimize latency vs maximize throughput"],
            bridges=["pareto-frontier", "tradeoff", "satisficing", "priority"],
            tags=["optimization", "multi-criteria", "tradeoff", "fleet"])

    def _load_probability(self):
        ns = self.add_namespace("probability",
            "Reasoning under uncertainty — priors, likelihood, evidence")

        ns.define("prior",
            "Belief about a hypothesis before seeing new evidence",
            description="Before you flip the coin, you believe it's 50/50. That's your prior. After seeing 9 heads in a row, your posterior updates to ~99.9% biased. But if your prior was 'this coin is rigged', you'd update differently. Priors matter enormously. The fleet's cuda-confidence starts with a prior (initial confidence) and updates with evidence.",
            level=Level.DOMAIN,
            examples=["medical test: prior probability of disease affects interpretation of positive test", "agent prior: 'this path is usually safe' before checking sensors", "Bayesian spam filter: prior probability that email is spam"],
            bridges=["posterior", "bayesian-update", "base-rate-fallacy", "likelihood"],
            tags=["probability", "bayesian", "prior-knowledge"])

        ns.define("base-rate-fallacy",
            "Ignoring the prior probability when interpreting new evidence",
            description="A disease affects 1 in 1000 people. Test is 99% accurate. You test positive. What's the chance you have the disease? Most people say 99%. The actual answer is ~9%. Why? Because the base rate (1/1000) means most positive tests are false positives. Agents (and humans) constantly commit this fallacy. The fleet guards against it by always tracking priors.",
            level=Level.BEHAVIOR,
            examples=["1 in 1000 disease, 99% test: positive test = only 9% chance of disease", "agent: sensor says danger, but danger is rare (base rate 0.1%) so probably false alarm", "profiling: rare trait in population, even accurate screening produces mostly false positives"],
            bridges=["prior", "bayesian-update", "false-positive", "calibration"],
            tags=["probability", "fallacy", "reasoning"])

        ns.define("conjunction-fallacy",
            "Believing that a specific conjunction is more probable than a general statement",
            description="'Linda is a bank teller and a feminist' is judged more probable than 'Linda is a bank teller' — but that's mathematically impossible. A conjunction can never be more probable than its components. Agents face this when they overweight specific scenarios ('sensor failure AND navigation error') over general ones ('something went wrong').",
            level=Level.BEHAVIOR,
            examples=["Linda the feminist bank teller", "agent: 'the path is blocked because the door is locked AND the key is lost' vs 'the path is blocked'", "overweighting specific failure modes over general failure probability"],
            bridges=["probability", "fallacy", "reasoning", "specificity"],
            tags=["probability", "fallacy", "cognitive-bias"])

        ns.define("regression-to-mean",
            "Extreme observations tend to be followed by more average ones",
            description="Rookie of the year has a mediocre second season. Not because they got worse — because their first season was unusually good (luck + skill). The fleet sees this: an agent with an unusually successful strategy will see performance decline toward average. Don't overreact — it's probably regression, not degradation.",
            level=Level.BEHAVIOR,
            examples=["sports: rookie of the year slump", "agent: amazing performance week 1, average week 2 — not because something broke", "student: aced test after studying hard, next test is lower — not because they forgot everything"],
            bridges=["mean", "variance", "luck", "calibration"],
            tags=["probability", "statistics", "fallacy"])

    def _load_economics(self):
        ns = self.add_namespace("economics",
            "Markets, incentives, costs, and resource allocation")

        ns.define("opportunity-cost",
            "The value of the best alternative you gave up by choosing this option",
            description="Every choice has a hidden cost: what you COULD have done instead. Spending 10 minutes deliberating means NOT spending those 10 minutes acting. The fleet's energy budget makes opportunity cost explicit: energy spent on deliberation is energy NOT available for action.",
            level=Level.DOMAIN,
            examples=["studying for exam A means not studying for exam B", "agent spending ATP on deliberation can't spend it on perception", "choosing to explore means not exploiting the known best path"],
            bridges=["tradeoff", "resource-allocation", "budget", "cost"],
            tags=["economics", "cost", "decision"])

        ns.define("marginal-cost",
            "The cost of producing one more unit — usually decreasing",
            description="First unit is expensive (setup). Each additional unit gets cheaper. First agent in fleet costs full setup. Tenth agent costs almost nothing. This explains why fleet coordination scales well: the marginal cost of adding one more agent decreases.",
            level=Level.DOMAIN,
            examples=["software: first copy costs $1M, next copy costs $0.01", "fleet: first agent needs full setup, additional agents need minimal extra infrastructure", "manufacturing: first car off assembly line is most expensive"],
            bridges=["economies-of-scale", "diminishing-returns", "cost", "scaling"],
            tags=["economics", "cost", "scaling"])

        ns.define("externalities",
            "Costs or benefits that affect parties not involved in the transaction",
            description="Factory pollutes: factory benefits (cheaper production), community pays (health costs). The cost is external to the transaction. In the fleet: one agent's noisy sensor readings pollute the shared information space, affecting all agents. Energy costs for communication (cuda-communication) internalize this externality.",
            level=Level.DOMAIN,
            examples=["pollution from factory", "loud music in shared office", "agent spamming fleet messages: cheap for sender, expensive for receivers", "vaccination: positive externality (protects others)"],
            bridges=["tragedy-of-commons", "mechanism-design", "incentive-alignment", "cost"],
            tags=["economics", "market-failure", "incentive"])

        ns.define("market-equilibrium",
            "Price point where supply equals demand — neither shortage nor surplus",
            description="The invisible hand. When price is too high, supply exceeds demand (surplus), price drops. When too low, demand exceeds supply (shortage), price rises. In the fleet: energy allocation reaches equilibrium when supply (ATP generation from rest) equals demand (consumption from actions). cuda-energy manages this.",
            level=Level.DOMAIN,
            examples=["supply and demand curves crossing", "fleet energy: rest generates ATP, actions consume it, equilibrium when balanced", "task allocation: supply of available agents meets demand from tasks"],
            bridges=["supply-demand", "equilibrium", "homeostasis", "energy-budget"],
            tags=["economics", "equilibrium", "market"])

    def _load_ecology(self):
        ns = self.add_namespace("ecology",
            "How agents interact with their environment and each other as an ecosystem")

        ns.define("niche",
            "The specific role and resource space an organism occupies in its ecosystem",
            description="No two species can occupy the exact same niche for long (competitive exclusion). Each finds its own role: one eats leaves at the top of the tree, another eats leaves at the bottom. In the fleet, each agent has a niche: navigator, sensor, communicator. cuda-playbook manages domain-specific strategies per niche.",
            level=Level.DOMAIN,
            examples=["different bird species feeding at different heights in same tree", "fleet: navigation agent niche vs communication agent niche", "market: different companies targeting different customer segments"],
            bridges=["competitive-exclusion", "speciation", "specialization", "role"],
            tags=["ecology", "niche", "role", "fleet"])

        ns.define("keystone-species",
            "A species whose removal dramatically changes the entire ecosystem",
            description="Remove wolves from Yellowstone: elk overpopulate, eat all the willows, beavers disappear, rivers change course. The wolf is a keystone species — small biomass, enormous impact. In the fleet: the captain agent (cuda-captain) is a keystone. Remove it and fleet coordination collapses even though it does minimal actual work.",
            level=Level.DOMAIN,
            examples=["wolves in Yellowstone", "sea otters maintaining kelp forests", "fleet captain: small computational footprint but critical for coordination", "team lead: doesn't write code but enables the team"],
            bridges=["hub", "cascade-failure", "critical-dependency", "leader"],
            tags=["ecology", "critical", "system-impact"])

        ns.define("symbiosis",
            "Long-term interaction between different species that benefits at least one",
            description="Mutualism: both benefit (bees and flowers). Commensalism: one benefits, other unaffected (barnacles on whale). Parasitism: one benefits, other harmed (tapeworm). In the fleet, agents form mutualistic relationships: navigator provides paths, sensor provides observations — both benefit from the exchange.",
            level=Level.DOMAIN,
            examples=["bees pollinate flowers, flowers feed bees", "barnacles on whale: barnacles benefit, whale unaffected", "fleet: navigator and sensor agents in mutualism — both need each other"],
            bridges=["cooperation", "mutualism", "parasitism", "niche"],
            tags=["ecology", "interaction", "cooperation"])

        ns.define("competitive-exclusion",
            "Two species competing for the same niche cannot coexist indefinitely",
            description="One will eventually outcompete the other. They must differentiate or one goes extinct. In the fleet: if two agents perform the exact same function, the fleet wastes resources. Agents must specialize or one should be deactivated. This drives the fleet toward efficient role distribution.",
            level=Level.BEHAVIOR,
            examples=["two similar bird species on an island: one outcompetes the other", "two identical fleet agents: one should specialize or be removed", "market: companies with identical products compete until one dominates"],
            bridges=["niche", "speciation", "specialization", "diversity"],
            tags=["ecology", "competition", "specialization"])

        ns.define("succession",
            "Predictable sequence of community changes following a disturbance",
            description="After a volcano erupts: lichens first, then mosses, then grasses, then shrubs, then trees. Each stage prepares the environment for the next. In the fleet: after a major disruption (agent failure, new task), behavior reorganizes in a predictable sequence: first basic survival, then perception, then coordination, then optimization.",
            level=Level.BEHAVIOR,
            examples=["volcanic island colonization", "forest regrowth after fire", "fleet recovery after major failure: instinct -> perception -> coordination -> optimization"],
            bridges=["punctuated-equilibrium", "disruption", "recovery", "stages"],
            tags=["ecology", "recovery", "sequence"])

    def _load_emotion(self):
        ns = self.add_namespace("emotion",
            "Emotional states as computational modulators of agent behavior")

        ns.define("valence-arousal",
            "Two-dimensional model of emotion: positive/negative (valence) x calm/excited (arousal)",
            description="Every emotion can be placed on a 2D plane. Joy = high valence, high arousal. Calm = high valence, low arousal. Anger = low valence, high arousal. Sadness = low valence, low arousal. The fleet's cuda-emotion uses this model: emotional state modulates attention, decision speed, and communication style.",
            level=Level.DOMAIN,
            examples=["joy: positive valence, high arousal", "calm: positive valence, low arousal", "anger: negative valence, high arousal", "agent: high arousal = faster decisions, lower accuracy"],
            bridges=["emotion", "modulation", "attention", "decision"],
            tags=["emotion", "psychology", "modulation", "fleet"])

        ns.define("emotional-contagion",
            "Emotional state spreading from one agent to others through observation",
            description="One person yawns, others yawn. Panic in a crowd. Laughter is infectious. In the fleet, cuda-emotion implements emotional contagion: if one agent detects danger (high arousal, negative valence), nearby agents may adopt a similar state. This enables rapid fleet-wide responses but risks panic cascades.",
            level=Level.BEHAVIOR,
            examples=["laughter spreading through a room", "panic in a crowd", "fleet: one agent detects threat, nearby agents become alert"],
            bridges=["cascade-failure", "emotion", "gossip", "swarm"],
            tags=["emotion", "social", "contagion", "fleet"])

        ns.define("anticipation",
            "Predictive emotional state generated by expecting a future event",
            description="The pleasure of anticipating dinner is different from the pleasure of eating it. Anticipation modulates current behavior based on predicted future state. The fleet's temporal reasoning (cuda-temporal) implements this: deadline urgency is a form of anticipation — emotional intensity increases as the deadline approaches.",
            level=Level.DOMAIN,
            examples=["looking forward to vacation", "dread before a difficult meeting", "agent: increasing urgency as deadline approaches = anticipatory emotional modulation"],
            bridges=["deadline-urgency", "prediction", "temporal", "motivation"],
            tags=["emotion", "prediction", "temporal", "motivation"])

    def _load_creativity(self):
        ns = self.add_namespace("creativity",
            "Generating novel, useful combinations from existing elements")

        ns.define("analogy",
            "Mapping structure from a known domain to a novel domain — 'A is to B as C is to D'",
            description="The core mechanism of creative thought. Electricity flows like water (current, pressure/voltage, resistance). The atom is like a solar system. Stigmergy in ant colonies is like git commits. Analogies transfer understanding from familiar domains to unfamiliar ones. HAV itself is a tool for analogy: fleet vocabulary borrows from biology, economics, physics.",
            level=Level.DOMAIN,
            examples=["electricity:water :: voltage:pressure :: current:flow :: resistance:narrowing", "atom:solar system :: nucleus:sun :: electrons:planets", "stigmergy:git commits :: pheromone trails:commit history"],
            bridges=["transfer-learning", "metaphor", "abstraction", "cross-domain"],
            tags=["creativity", "reasoning", "analogy", "abstraction"])

        ns.define("divergent-thinking",
            "Generating many possible solutions without judging them — brainstorming mode",
            description="Quantity over quality. The goal is to generate options, not evaluate them. 'How many ways could we cross this river?' — bridge, boat, swim, tunnel, helicopter, catapult, zip line, wait for winter and walk on ice. The fleet's exploration phase (cuda-deliberation Consider) is divergent thinking.",
            level=Level.BEHAVIOR,
            examples=["brainstorming: generate 100 ideas, don't judge yet", "agent: consider all possible navigation strategies before evaluating any", "creative writing: write freely, edit later"],
            bridges=["exploration", "convergent-thinking", "brainstorming", "generation"],
            antonyms=["convergent-thinking"],
            tags=["creativity", "generation", "exploration"])

        ns.define("convergent-thinking",
            "Evaluating and selecting the best solution from generated options — decision mode",
            description="Now that we have 100 ideas, which 3 are worth trying? Apply criteria, rank, select. The fleet's deliberation phase (cuda-deliberation Resolve) is convergent thinking: evaluate proposals by confidence, cost, and alignment, then select the best.",
            level=Level.BEHAVIOR,
            examples=["narrowing 100 brainstorm ideas to 3 actionable ones", "agent: evaluate all navigation strategies by confidence, select best", "editing a rough draft into a polished piece"],
            bridges=["deliberation", "divergent-thinking", "evaluation", "selection"],
            antonyms=["divergent-thinking"],
            tags=["creativity", "evaluation", "selection"])

        ns.define("combinatorial-explosion",
            "Number of possible combinations grows exponentially with the number of elements",
            description="10 items have 10! = 3.6 million permutations. 20 items have 20! ≈ 2.4 quintillion. You can't evaluate all combinations. The fleet uses heuristics, pruning, and satisficing to avoid combinatorial explosion in deliberation. cuda-filtration limits the deliberation scope to manageable size.",
            level=Level.META,
            examples=["chess: too many positions to enumerate, must use heuristics", "traveling salesman: N! routes, NP-hard", "agent deliberation: 100 possible actions × 10 contexts × 5 goals = 5000 combinations to evaluate"],
            bridges=["pruning", "satisficing", "filtration", "heuristic", "paradox-of-choice"],
            tags=["creativity", "complexity", "scaling", "challenge"])

        ns.define("constraint-relaxation",
            "Solving a hard problem by temporarily removing a constraint, solving, then re-adding it",
            description="Can't solve the problem? Remove one constraint, solve the easier version, then figure out how to satisfy the removed constraint. This is a powerful creative technique. In the fleet: if deliberation is too expensive, relax the accuracy constraint, get a fast answer, then refine it. Or: ignore energy budget temporarily, plan the optimal solution, then trim to fit the budget.",
            level=Level.PATTERN,
            examples=["knapsack: ignore weight limit, pack all valuable items, then remove items until weight fits", "agent: plan optimal path ignoring energy, then trim path to fit budget", "writing: write without worrying about word count, then edit to fit"],
            bridges=["satisficing", "optimization", "heuristic", "abstraction"],
            tags=["creativity", "technique", "problem-solving"])

    def _load_metacognition(self):
        ns = self.add_namespace("metacognition",
            "Thinking about thinking — self-awareness, monitoring, and control of cognition")

        ns.define("introspection",
            "Examining one's own mental states, processes, and reasons for action",
            description="Why did I choose path A over path B? Because path A had higher confidence? Or because I'm biased toward familiar paths? The fleet's cuda-self-model implements introspection: the agent tracks its own capabilities, calibration, and growth trends, creating a model of itself.",
            level=Level.BEHAVIOR,
            examples=["asking 'why did I make that decision?'", "agent reviewing its own deliberation log to understand decision patterns", "journaling as self-reflection"],
            bridges=["self-model", "metacognitive-monitoring", "calibration", "theory-of-mind"],
            tags=["metacognition", "self-awareness", "reflection"])

        ns.define("theory-of-mind",
            "Attributing mental states to others — predicting what others think, want, and will do",
            description="I know that you know that I know. Humans develop this around age 4. In the fleet, agents need theory-of-mind to coordinate: the navigator must model what the sensor agent is currently perceiving to plan routes effectively. cuda-social implements social reasoning.",
            level=Level.DOMAIN,
            examples=["predicting what another driver will do at an intersection", "agent modeling another agent's current goal to avoid interference", "negotiating: understanding the other party's priorities"],
            bridges=["self-model", "social", "prediction", "coordination"],
            tags=["metacognition", "social", "prediction", "fleet"])

        ns.define("metacognitive-monitoring",
            "Watching your own cognitive process in real-time to detect confusion or error",
            description="While reading this, you might realize 'I don't understand this paragraph' — that's metacognitive monitoring. You detect your own confusion. The fleet implements this: if deliberation confidence drops below threshold for multiple cycles, the agent recognizes it's confused and escalates (requests help, switches strategy, or defers).",
            level=Level.BEHAVIOR,
            examples=["'I don't understand' — detecting own confusion", "'I'm going in circles' — detecting unproductive deliberation", "agent: confidence dropping consistently across proposals = metacognitive alarm"],
            bridges=["introspection", "calibration", "confusion", "threshold"],
            tags=["metacognition", "monitoring", "self-awareness"])

    def _load_failure_modes(self):
        ns = self.add_namespace("failure-modes",
            "How systems fail — and how to prevent, detect, and recover from failure")

        ns.define("single-point-of-failure",
            "One component whose failure causes the entire system to fail",
            description="No redundancy. One wire breaks, the whole circuit dies. One server crashes, the whole service goes down. The fleet avoids SPOFs through leader election (cuda-election), circuit breakers (cuda-circuit), and redundant agents. Any critical component must have a backup.",
            level=Level.DOMAIN,
            examples=["one hard drive with no backup", "one DNS server for entire network", "fleet: captain agent crash with no election mechanism = SPOF"],
            bridges=["redundancy", "cascade-failure", "circuit-breaker", "hub"],
            tags=["failure", "critical", "architecture"])

        ns.define("robustness",
            "Ability to maintain function despite perturbations without changing structure",
            description="A robust bridge doesn't collapse when a truck drives over it. It handles the load without needing to adapt. In the fleet: robust agents handle normal variation (sensor noise, network delays) without changing their strategy. They absorb perturbations.",
            level=Level.DOMAIN,
            examples=["bridge handles varying loads", "agent handles sensor noise without changing strategy", "software handles invalid input without crashing"],
            bridges=["resilience", "graceful-degradation", "anti-fragility", "stability"],
            tags=["failure", "property", "system"])

        ns.define("anti-fragility",
            "Getting stronger from stress — not just surviving perturbations but improving because of them",
            description="Muscles grow from exercise (stress). Immune system strengthens from exposure to pathogens. A system that gets BETTER from failure. The fleet aims for anti-fragility: when an agent fails, the fleet learns from it and becomes more resilient. Gene pool quarantine (cuda-genepool) is anti-fragile: failed strategies get quarantined, making the gene pool stronger.",
            level=Level.META,
            examples=["muscles grow from exercise", "immune system from exposure", "fleet: agent failure -> gene quarantined -> fleet stronger", "bone density increases from stress"],
            bridges=["robustness", "resilience", "learning-from-failure", "adaptation"],
            antonyms=["fragility"],
            tags=["failure", "meta", "aspiration", "fleet"])

        ns.define("common-mode-failure",
            "Multiple components fail simultaneously because they share the same vulnerability",
            description="Backup generator fails during outage — because it's maintained by the same team that maintains the main generator. Redundancy doesn't help if both systems share the same weakness. In the fleet: two agents using the same sensor type both fail in the same environmental conditions. Diversity prevents common-mode failure.",
            level=Level.DOMAIN,
            examples=["redundant servers in same datacenter: both fail in fire", "same sensor type on multiple agents: all fail in same interference", "identical software on different hardware: same bug crashes all"],
            bridges=["redundancy", "diversity", "single-point-of-failure", "robustness"],
            tags=["failure", "systematic", "redundancy"])

        ns.define("brittleness",
            "System works well under expected conditions but catastrophically fails under unexpected ones",
            description="Glass is hard but brittle: strong against compression, shatters under impact. A brittle agent performs perfectly in training but completely fails on novel inputs. Contrast with robustness (handles variation) and anti-fragility (improves from stress). The fleet tests for brittleness by deliberately introducing novel situations.",
            level=Level.BEHAVIOR,
            examples=["glass vs rubber", "model that works on test data but fails on real-world edge cases", "agent that follows instructions perfectly but freezes when facing an unexpected obstacle"],
            bridges=["robustness", "anti-fragility", "graceful-degradation", "edge-case"],
            antonyms=["robustness", "anti-fragility"],
            tags=["failure", "property", "fragility"])

    def _load_thermodynamics(self):
        ns = self.add_namespace("thermodynamics",
            "Energy, entropy, and the arrow of time — physics metaphors for agent systems")

        ns.define("entropy-production",
            "All processes irreversibly increase total entropy — order always degrades without energy input",
            description="Your room gets messier without effort. Agents accumulate noise, trust decays, knowledge goes stale. Maintaining order requires energy input (restoring trust, updating knowledge, calibrating sensors). The fleet's constant energy expenditure (ATP generation and consumption) is the thermodynamic cost of maintaining order against entropy.",
            level=Level.META,
            examples=["room gets messy without cleaning", "agent trust decays without positive interactions", "knowledge goes stale without updates", "code degrades without maintenance (software entropy)"],
            bridges=["entropy", "energy", "decay", "maintenance"],
            tags=["physics", "thermodynamics", "meta", "fleet"])

        ns.define("free-energy-principle",
            "Biological systems minimize surprise (prediction error) by updating model or changing environment",
            description="Karl Friston's theory: the brain minimizes free energy = expected surprise. Two ways: update your model (perception/learning) or change the world to match your model (action). The fleet implements this: agents either update their world model (cuda-world-model) or take action to make reality match predictions (navigate to expected state).",
            level=Level.META,
            examples=["you feel cold -> put on jacket (change world) or learn that it's cold here (update model)", "agent's prediction doesn't match sensor -> update world model OR move to expected state", "surprise minimization = free energy minimization"],
            bridges=["prediction", "action-perception", "homeostasis", "model"],
            tags=["physics", "neuroscience", "meta", "unified-theory"])

        ns.define("dissipative-structure",
            "Ordered pattern that emerges from energy flow through a system, maintaining itself far from equilibrium",
            description="Convection cells in boiling water. Hurricanes. Life itself. These structures exist ONLY because energy flows through them. Stop the energy flow and they dissolve. The fleet is a dissipative structure: agent coordination patterns emerge from the constant flow of information and energy. Without this flow, the fleet dissolves into individual agents.",
            level=Level.META,
            examples=["convection cells in boiling water", "hurricane maintained by ocean heat", "life maintained by metabolism", "fleet coordination maintained by constant message flow and energy expenditure"],
            bridges=["emergence", "self-organization", "energy-flow", "far-from-equilibrium"],
            tags=["physics", "complexity", "meta", "emergence"])

        ns.define("negentropy",
            "Local decrease in entropy (increase in order) at the expense of increased entropy elsewhere",
            description="Life is negentropic: organisms maintain internal order by consuming energy and producing waste heat (increasing environmental entropy). The fleet maintains order (coordinated behavior) by consuming ATP (energy) and producing waste (heat, noise, stale messages). Every act of organization has a thermodynamic cost.",
            level=Level.DOMAIN,
            examples=["plant converts sunlight to ordered structure, produces heat", "agent organizes fleet behavior, consumes ATP, produces noise", "refrigerator creates cold (order) by producing heat (disorder)"],
            bridges=["entropy", "energy", "order", "cost"],
            tags=["physics", "thermodynamics", "life", "cost"])

    def _load_complexity(self):
        ns = self.add_namespace("complexity",
            "Emergence, self-organization, and behavior at the edge of chaos")

        ns.define("edge-of-chaos",
            "The boundary between order and chaos where complex adaptive behavior is maximized",
            description="Too ordered = frozen, nothing changes. Too chaotic = random, no patterns. The edge of chaos — between — is where interesting things happen. Cellular automata, neural networks, evolution all operate at the edge of chaos. The fleet's energy budget and trust decay rates are tuned to keep agents at this boundary: enough randomness to explore, enough structure to exploit.",
            level=Level.META,
            examples=["liquid water: ordered (ice) vs chaotic (steam), life exists in liquid", "brain: too synchronized = seizure, too random = coma, normal is edge of chaos", "agent: too rigid = stuck in local optimum, too random = no learning, sweet spot in between"],
            bridges=["chaos", "order", "emergence", "tuning", "criticality"],
            tags=["complexity", "meta", "sweet-spot"])

        ns.define("self-organization",
            "Order emerging spontaneously from local interactions without central control",
            description="No architect tells birds how to flock. No conductor tells heart cells when to beat. Order emerges from simple rules applied locally. The fleet aims for self-organization: agents follow simple rules (trust neighbors, share useful genes, conserve energy) and complex fleet behavior emerges without central coordination.",
            level=Level.META,
            examples=["bird flocking", "crystallization", "market price discovery", "fleet: complex coordination from simple agent rules"],
            bridges=["emergence", "swarm", "decentralized", "stigmergy"],
            tags=["complexity", "emergence", "decentralized"])

        ns.define("autocatalysis",
            "A process that produces the catalysts needed to accelerate itself — self-reinforcing growth",
            description="A chemical reaction that produces more of the enzyme that speeds it up. More enzyme = faster reaction = more enzyme. Positive feedback loop. In the fleet: successful genes produce ATP, which enables more exploration, which discovers more successful genes. Trust generates successful cooperation, which generates more trust. The fleet has multiple autocatalytic cycles.",
            level=Level.META,
            examples=["autocatalytic chemical sets (origin of life)", "viral spread: each infection produces more infections", "trust autocatalysis: trust enables cooperation which builds more trust", "learning autocatalysis: knowledge enables better learning"],
            bridges=["positive-feedback", "self-reinforcement", "growth", "exponential"],
            tags=["complexity", "growth", "positive-feedback"])

        ns.define("autopoiesis",
            "A system that continuously reproduces the conditions necessary for its own existence",
            description="A cell makes its own membrane. The membrane contains the cell. Break the membrane, the cell dies. The cell IS the process of maintaining itself. In the fleet: agents maintain their own code (self-modify), their own energy budget (rest when low), their own reputation (communicate reliably). The agent IS the process of maintaining itself.",
            level=Level.META,
            examples=["living cell maintains its own membrane", "agent maintains its own code through self-modification", "ecosystem maintains conditions for its own species", "organization maintains its own culture through onboarding"],
            bridges=["self-maintenance", "homeostasis", "closure", "life"],
            tags=["complexity", "life", "meta", "philosophy"])

        ns.define("phase-transition",
            "Abrupt qualitative change in system behavior at a critical threshold",
            description="Water becomes ice at 0C. Not gradually — suddenly. Magnetic material becomes magnetized at Curie temperature. Percolation: below critical density, no flow; above, flow everywhere. The fleet experiences phase transitions: below critical agent count, no coordination; above it, fleet behavior emerges. Below critical trust threshold, no cooperation; above it, collaboration emerges.",
            level=Level.META,
            examples=["water to ice at 0C", "magnetization at Curie temperature", "percolation at critical density", "fleet: coordination emerges above critical agent count"],
            bridges=["percolation", "critical-mass", "tipping-point", "emergence"],
            tags=["complexity", "criticality", "abrupt-change"])

    def _load_scaling(self):
        ns = self.add_namespace("scaling",
            "How systems behave as they grow — superlinear, sublinear, and critical transitions")

        ns.define("superlinear-scaling",
            "Output grows faster than input — 2x input produces more than 2x output",
            description="Cities: doubling population increases productivity by 115% (superlinear). Network effects: each new user adds more value than the last. In the fleet: adding the 10th agent to a coordination task may improve performance by 150% because new agent enables a completely new strategy (division of labor, specialization) that wasn't possible with 9 agents.",
            level=Level.DOMAIN,
            examples=["cities: 2x population = 2.15x innovation", "network effects: telephones become more valuable as more people have them", "fleet: 10th agent enables specialization that 9 agents couldn't achieve"],
            bridges=["economies-of-scale", "network-effects", "synergy", "phase-transition"],
            antonyms=["diminishing-returns"],
            tags=["scaling", "growth", "positive"])

        ns.define("diminishing-returns",
            "Each additional unit of input produces less additional output",
            description="First hour of study: learn a lot. Tenth hour: learn a little less. Hundredth hour: almost nothing new. The fleet experiences this: adding agents to a task has diminishing returns after the optimal number. Adding sensors has diminishing returns after sufficient coverage. Energy budgeting must account for diminishing returns on additional investment.",
            level=Level.DOMAIN,
            examples=["studying: first hour = big gains, 10th hour = small gains", "fertilizer: some helps a lot, too much kills the plant", "fleet: 3 agents on task = big improvement, 10th agent on same task = minimal improvement"],
            bridges=["marginal-cost", "opportunity-cost", "optimization", "saturating"],
            antonyms=["superlinear-scaling"],
            tags=["scaling", "economics", "saturation"])

        ns.define("critical-mass",
            "Minimum size needed for a phenomenon to become self-sustaining",
            description="Nuclear reaction: enough fissile material in close proximity = chain reaction. Social movement: enough early adopters = tipping point. Fleet: enough agents = emergent coordination. Below critical mass, the phenomenon dies out. Above it, it sustains and grows. The fleet monitors its size relative to critical mass for various behaviors.",
            level=Level.DOMAIN,
            examples=["nuclear critical mass", "social network needs enough users to be useful", "fleet: need minimum agents for stigmergy to work", "crowdfunding: need enough backers to reach goal"],
            bridges=["phase-transition", "tipping-point", "percolation", "bootstrap"],
            tags=["scaling", "criticality", "threshold"])

        ns.define("tipping-point",
            "A small perturbation that triggers a large, often irreversible, change in system state",
            description="The straw that breaks the camel's back. One more degree of warming triggers ice sheet collapse. One more agent defecting triggers fleet-wide defection cascade. Tipping points are dangerous because they're hard to predict — small changes near the tipping point cause disproportionately large effects.",
            level=Level.DOMAIN,
            examples=["climate tipping points: ice sheet collapse, Amazon dieback", "social: one person leaving a party triggers mass exodus", "fleet: one agent's failure triggers cascade failure when fleet is near capacity"],
            bridges=["phase-transition", "critical-mass", "cascade-failure", "nonlinearity"],
            tags=["scaling", "criticality", "danger", "nonlinearity"])

    def _load_linguistics(self):
        ns = self.add_namespace("linguistics",
            "Language structure, meaning, and the challenge of shared understanding")

        ns.define("compositionality",
            "Meaning of a complex expression is determined by meanings of its parts and their combination rules",
            description="'The cat sat on the mat' means what it means because you understand 'cat', 'sat', 'on', 'the', 'mat', and the rules for combining them. Without compositionality, you'd need to memorize every possible sentence. The fleet's A2A protocol is compositional: simple message types combine into complex communication patterns.",
            level=Level.DOMAIN,
            examples=["'red ball' = red + ball (compositionality)", "programming languages: expressions composed from primitives", "fleet A2A: simple intents combine into complex coordination protocols"],
            bridges=["semantics", "grammar", "productivity", "meaning"],
            tags=["linguistics", "semantics", "composition"])

        ns.define("metaphor",
            "Understanding one domain in terms of another — 'time is money', 'argument is war'",
            description="We can't talk about time without spending, saving, wasting, investing it. We can't talk about arguments without attacking, defending, winning, losing. Metaphors aren't just literary devices — they shape thought. The entire fleet vocabulary is built on biological metaphors: 'memory', 'learning', 'trust', 'energy', 'instinct' — all borrowed from biology to describe computation.",
            level=Level.DOMAIN,
            examples=["'time is money': spend time, save time, invest time", "'argument is war': attack a position, defend a claim, shoot down an argument", "fleet: 'trust', 'energy', 'memory', 'learning' — biological metaphors for computational concepts"],
            bridges=["analogy", "framing", "grounding", "domain-mapping"],
            tags=["linguistics", "thought", "metaphor", "framing"])

        ns.define("grounding-problem",
            "How words connect to the actual world — what does 'red' actually refer to?",
            description="A dictionary defines words in terms of other words. But at some point, words must connect to actual experience. 'Red' connects to the visual experience of seeing red. In the fleet: 'obstacle ahead' must connect to actual sensor readings. Without grounding, agents can communicate fluently but meaninglessly — passing symbols that refer to nothing. cuda-communication's SharedVocabulary addresses grounding.",
            level=Level.META,
            examples=["Chinese room argument: manipulating symbols without understanding", "agent saying 'danger ahead' without actually sensing danger", "dictionary circularity: all definitions reference other definitions"],
            bridges=["grounding", "symbol-grounding", "semantics", "meaning", "reference"],
            tags=["linguistics", "philosophy", "ai-safety", "meta"])

        ns.define("pragmatics",
            "How context determines meaning beyond the literal words",
            description="'Can you pass the salt?' is literally a yes/no question about ability. Pragmatically, it's a request. 'It's cold in here' is literally a statement about temperature. Pragmatically, it's a request to close the window. The fleet's A2A protocol encodes pragmatics: the Intent field carries the pragmatic meaning (Request, Warn, Command) separately from the literal payload.",
            level=Level.DOMAIN,
            examples=["'can you pass the salt?' = request, not question", "'it's cold' = close the window", "A2A message: literal payload + pragmatic intent (Command vs Inform vs Warn)"],
            bridges=["speech-act", "context", "intent", "communication"],
            tags=["linguistics", "context", "meaning", "fleet"])

        ns.define("ambiguity",
            "A single expression having multiple possible interpretations",
            description="'I saw the man with the telescope' — did I use a telescope, or did the man have one? Natural language is full of ambiguity. The fleet avoids ambiguity in A2A messages by using structured intents and typed payloads instead of natural language. But ambiguity is sometimes useful: vague commands allow agents to exercise judgment.",
            level=Level.DOMAIN,
            examples=["'I saw the man with the telescope' (who has the telescope?)", "'flying planes can be dangerous' (are planes dangerous, or is flying them dangerous?)", "agent: 'handle the obstacle' — which obstacle? how? ambiguity allows judgment"],
            bridges=["pragmatics", "context", "disambiguation", "communication"],
            tags=["linguistics", "challenge", "meaning"])
    def _load_mathematics(self):
        ns = self.add_namespace("mathematics",
            "Mathematical structures and operations underlying agent cognition")

        ns.define("harmonic-mean",
            "Average that penalizes small values: n / (1/a + 1/b + ...)",
            description="Unlike arithmetic mean (add and divide), harmonic mean divides and adds. This means small values drag the average down much more than in arithmetic mean. If one sensor says 'I'm 10% confident', the fused confidence will be near 10% regardless of other sensors. Used throughout the fleet for confidence fusion.",
            level=Level.CONCRETE,
            examples=["speed: average of 60mph and 40mph via harmonic mean = 48mph (not 50mph)", "confidence: 0.9 and 0.1 fused = 0.09 (not 0.5)", "resistor parallel: 1/R = 1/R1 + 1/R2"],
            bridges=["harmonic-mean-fusion", "confidence", "fusion", "average"],
            tags=["mathematics", "fleet-foundation"])

        ns.define("exponential-decay",
            "Value decreases as e^(-λt), creating a smooth decline with configurable half-life",
            description="Radioactive decay. Memory fade. Trust erosion. All follow the same curve: start fast, slow down over time. The half-life parameter controls how fast: short half-life = fast decay, long half-life = slow decay. This appears in 30+ fleet crates as the universal aging mechanism.",
            level=Level.PATTERN,
            examples=["radioactive half-life of 10 years", "trust decays with half-life of 1 week", "memory with half-life of 30 seconds (working) vs 1 year (procedural)", "formula: value(t) = initial * e^(-λt)"],
            properties={"formula": "e^(-lambda*t)", "half_life": "ln(2)/lambda", "ubiquitous": True},
            bridges=["decay", "forgetting-curve", "trust", "memory"],
            tags=["mathematics", "ubiquitous", "fleet"])

        ns.define("welford-algorithm",
            "Online algorithm for computing mean and variance without storing all data",
            description="Standard variance calculation needs two passes (or storing all values). Welford's algorithm computes running mean and variance in a single pass, using only 3 variables. The fleet's cuda-emergence uses it for baseline detection: track running statistics of agent behavior to detect emergent patterns.",
            level=Level.CONCRETE,
            examples=["streaming statistics: process 1M events with 3 variables, not 1M storage", "anomaly detection: is current behavior outside 2σ of running mean?", "agent behavior baseline tracking"],
            bridges=["mean", "variance", "anomaly-detection", "online-algorithm"],
            tags=["mathematics", "algorithm", "statistics"])

        ns.define("topological-sort",
            "Order elements so that every dependency appears before its dependent",
            description="You can't bake the cake before mixing the batter. Topological sort finds a valid ordering given dependency constraints. Uses DFS with cycle detection: if a cycle exists, no valid ordering is possible. The fleet's cuda-workflow uses this for task scheduling.",
            level=Level.PATTERN,
            examples=["build system: compile dependencies before dependents", "course schedule: take prerequisites first", "workflow: complete prerequisite tasks before dependent tasks"],
            bridges=["dag", "workflow", "dependency", "ordering"],
            tags=["mathematics", "algorithm", "scheduling"])

        ns.define("hamming-distance",
            "Number of positions at which two strings (or vectors) differ",
            description="10101 vs 11100 = 3 (positions 2, 4, 5 differ). Simple, fast, and useful for error detection, DNA comparison, and similarity search. In the fleet, used for pattern matching and anomaly detection: how different is the current state from the expected state?",
            level=Level.CONCRETE,
            examples=["error detection: received 1010, expected 1110, hamming distance 1", "DNA sequence comparison", "agent state comparison: current vs expected behavior pattern"],
            bridges=["similarity", "distance", "error-detection", "pattern-matching"],
            tags=["mathematics", "metric", "algorithm"])
