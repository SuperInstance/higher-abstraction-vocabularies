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
        self._load_semantics()
        self._load_philosophy_of_mind()
        self._load_identity()
        self._load_morphology()
        self._load_motivation()
        self._load_psychology()
        self._load_pattern_recognition()
        self._load_resilience()
        self._load_information_theory()
        self._load_systems_thinking()
        self._load_ethics()
        self._load_concurrency()
        self._load_design_patterns()
        self._load_measurement()
        self._load_time()
        self._load_decision_theory()
        self._load_obsolescence()
        self._load_perception()
        self._load_communication()
        self._load_tradeoffs()
        self._load_epistemology()
        self._load_biology()
        self._load_philosophy_of_science()
        self._load_causation()
        self._load_abstraction()
        self._load_dynamics()
        self._load_collective_intelligence()
        self._load_risk()
        self._load_autonomy()
        self._load_simulation()
        self._load_privacy()
        self._load_organization()
        self._load_strategy()
        self._load_narrative()
        self._load_language_design()
        self._load_knowledge_rep()
        self._load_robotics()
        self._load_cybernetics()
        self._load_algebra()
        self._load_finance()
        self._load_materials_science()
        self._load_verification()
        self._load_graph_theory()
        self._load_learning_theory()
        self._load_phenomenology()
        self._load_anthropology()
        self._load_logic()
        self._load_probability_distributions()
        self._load_set_theory()
        self._load_topology()
        self._load_ai_safety()
        self._load_ontology_engineering()
        self._load_construction()
        self._load_cognitive_science()
        self._load_signal_processing()
        self._load_emergence_deep()
        self._load_communication_deep()
        self._load_governance()
        self._load_network_science()
        self._load_operations()
        self._load_distillation()
        self._load_orchestration()
        self._load_tactics()
        self._load_diagnostics()
        self._load_leverage()
        self._load_adaptation_patterns()
        self._load_friction()
        self._load_compression()
        self._load_boundaries()
        self._load_temporal_patterns()
        self._load_quality()
        self._load_mechanics()
        self._load_entrenchment()
        self._load_knowledge_transfer()
        self._load_efficiency()
        self._load_metaphor()
        self._load_scaling()
        self._load_interface_patterns()
        self._load_attention()
        self._load_security_deep()
        self._load_error_strategies()
        self._load_coordination_deep()
        self._load_composition()
        self._load_observability()
        self._load_anti_patterns()
        self._load_influence()
        self._load_negotiation()
        self._load_capacity()
        self._load_decision_patterns()
        self._load_maintenance()
        self._load_ux()
        self._load_trade_patterns()
        self._load_morphology()
        self._load_risk_patterns()
        self._load_propagation()
        self._load_lifecycle()
        self._load_incentives()
        self._load_action_verbs()
        self._load_action_verbs_2()
        self._load_final_verbs()
        self._load_github_native()
        self._load_fleet_biology()
        self._load_cognition_deep()
        self._load_fleet_interactions()
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

    def _load_semantics(self):
        ns = self.add_namespace("semantics",
            "Meaning, reference, truth, and the relationship between symbols and the world")

        ns.define("reference",
            "The relationship between a symbol and the thing it points to in the world",
            description="'Cat' refers to actual cats. 'The president' refers to a specific person. Reference is the arrow from word to world. In the fleet, A2A message payloads reference actual states, goals, and observations. But the reference must be grounded in shared experience — otherwise the symbol floats free of meaning.",
            level=Level.DOMAIN,
            examples=["'cat' refers to actual cats", "pointer refers to memory address", "A2A message payload refers to actual sensor state"],
            bridges=["grounding-problem", "symbol", "meaning", "semantics"],
            tags=["semantics", "reference", "meaning"])

        ns.define("compositionality",
            "Meaning of complex expressions determined by parts and combination rules",
            description="[Already defined but bridges are key] The fleet relies on compositional communication: simple message types compose into complex protocols. A Request + Accept = agreement. A Warn + Command = urgent directive. Compositionality enables a small vocabulary to express infinite meanings.",
            level=Level.DOMAIN,
            examples=["'red ball' meaning from 'red' + 'ball' + combination rule", "programming: expressions composed from primitives", "A2A: simple intents combine into complex coordination"],
            bridges=["productivity", "grammar", "meaning", "communication"],
            tags=["semantics", "composition", "language"])

        ns.define("truth-conditional",
            "Meaning defined by the conditions under which a statement would be true",
            description="'Snow is white' is true if and only if snow is white. The meaning of a statement IS its truth conditions. In the fleet: the meaning of 'obstacle at (3,5)' is the condition under which it would be verified (sensor reading matches coordinates). This grounds fleet statements in verifiable conditions.",
            level=Level.DOMAIN,
            examples=["'it is raining' is true iff rain is actually falling", "agent: 'path is blocked' is true iff sensor confirms obstacle", "SQL: WHERE clause defines truth conditions"],
            bridges=["reference", "verification", "grounding", "logic"],
            tags=["semantics", "truth", "logic"])

    def _load_philosophy_of_mind(self):
        ns = self.add_namespace("philosophy-of-mind",
            "What is mind? What is consciousness? Can machines think?")

        ns.define("functionalism",
            "Mental states defined by their functional role, not their physical implementation",
            description="Pain isn't C-fibers firing. Pain is whatever plays the 'pain role' — causes withdrawal, avoidance, distress reporting. A robot with the right functional organization could genuinely feel pain. This is the philosophical foundation of the fleet: agents aren't defined by their hardware (Jetson, cloud, FPGA) but by their functional organization (perceive, deliberate, act).",
            level=Level.META,
            examples=["pain defined by its causal role, not neural substrate", "fleet: agent defined by functional pipeline, not hardware", "multiple realizability: same function on different hardware"],
            bridges=["embodiment", "consciousness", "identity", "abstraction"],
            tags=["philosophy", "mind", "meta"])

        ns.define("chinese-room",
            "Following rules to manipulate symbols doesn't constitute understanding",
            description="Searle's argument: a person in a room follows rules to manipulate Chinese characters, producing correct responses, without understanding Chinese. Critics: the whole room understands, or the simulation is sufficient. For the fleet: an agent that correctly processes A2A messages without understanding their meaning is a Chinese room. Grounding in shared experience is the proposed solution.",
            level=Level.DOMAIN,
            examples=["person following rules to answer Chinese questions without understanding Chinese", "agent processing sensor data without understanding what it means", "language model generating text without comprehension"],
            bridges=["grounding-problem", "consciousness", "symbol", "understanding"],
            tags=["philosophy", "ai", "understanding"])

        ns.define("embodiment",
            "Cognition requires a body interacting with a physical (or simulated) environment",
            description="You can't learn to walk by reading about walking. Intelligence requires sensorimotor interaction with the world. The fleet embodies agents: they have sensors (cuda-sensor-agent), actuators (cuda-vessel-bridge), and must navigate real or simulated environments. Embodiment grounds their cognition in experience.",
            level=Level.DOMAIN,
            examples=["learning to walk requires a body", "robot learning from physical interaction, not simulation", "fleet agent learning from actual sensor readings, not descriptions of sensor readings"],
            bridges=["functionalism", "grounding-problem", "perception", "action"],
            tags=["philosophy", "cognition", "embodiment", "fleet"])

        ns.define("extended-mind",
            "Cognitive processes extend beyond the brain into the environment and tools",
            description="Clark and Chalmers: your notebook is part of your memory. Your calculator is part of your cognition. The boundary of 'mind' includes tools and environment. For the fleet: cuda-memory-fabric extends agent memory beyond the agent to the fleet. The fleet mesh extends agent cognition to other agents. The agent's mind includes its tools, its peers, and its environment.",
            level=Level.META,
            examples=["notebook as external memory", "smartphone as extended cognition", "fleet: other agents are part of this agent's extended mind", "calculator as extended mathematical cognition"],
            bridges=["memory", "tools", "environment", "cognition"],
            tags=["philosophy", "cognition", "tools", "meta"])

    def _load_identity(self):
        ns = self.add_namespace("identity",
            "Who is an agent? How do agents identify themselves and each other?")

        ns.define("decentralized-identity",
            "Self-sovereign identity that agents control without relying on a central authority",
            description="No ID card issued by a government. No username from a platform. The agent controls its own identity through cryptographic keys. The fleet's cuda-did implements DID (Decentralized Identifier) documents with cryptographic verification. Each agent IS its own identity authority.",
            level=Level.DOMAIN,
            examples=["DID: did:cuda:agent-abc123", "agent proves identity by signing a challenge with its private key", "no central registry needed"],
            bridges=["trust", "authentication", "sovereignty", "cryptographic-identity"],
            tags=["identity", "did", "decentralized", "fleet"])

        ns.define("provenance",
            "The complete lineage of a decision or data artifact: where it came from and how it was transformed",
            description="Where did this decision come from? What data informed it? Who was responsible? The fleet's cuda-provenance chains every decision to its inputs, creating an auditable trail. Like git blame for agent cognition: you can trace any output back through every transformation to its original inputs.",
            level=Level.DOMAIN,
            examples=["git blame: who wrote this line and why", "supply chain: where did this component come from", "agent: this decision was based on sensor reading X, deliberation round Y, with confidence Z"],
            bridges=["audit-trail", "causal-chain", "accountability", "event-sourcing"],
            tags=["identity", "audit", "traceability", "fleet"])

        ns.define("attestation",
            "A cryptographic claim about an agent's capabilities, verified by a trusted third party",
            description="'This agent is certified for outdoor navigation' — signed by the fleet certification authority. Attestations let agents prove capabilities without demonstrating them every time. The fleet's cuda-did supports 6 attestation claim types. Like a driver's license for agent capabilities.",
            level=Level.CONCRETE,
            examples=["TLS certificate attests server identity", "driver license attests driving capability", "agent attestation: certified for level-3 navigation tasks"],
            bridges=["decentralized-identity", "trust", "certification", "credential"],
            tags=["identity", "credential", "trust", "fleet"])

    def _load_morphology(self):
        ns = self.add_namespace("morphology",
            "Forms, structures, and patterns in space and thought")

        ns.define("self-similarity",
            "A pattern that contains copies of itself at every scale — fractals",
            description="A coastline looks wiggly at 1km, 100m, and 1m scale. Branches look like smaller trees. The fleet's hierarchical structure is self-similar: agents contain sub-agents, sub-agents contain modules, modules contain functions. The same organizational pattern repeats at every level of granularity.",
            level=Level.DOMAIN,
            examples=["fractal coastline", "tree branches", "fleet: fleet -> agent -> module -> function -> instruction", "Russian dolls"],
            bridges=["fractal", "hierarchy", "scale-invariance", "recursion"],
            tags=["morphology", "pattern", "fractal"])

        ns.define("fractal",
            "A mathematical object with fractional dimension — infinitely detailed at every scale",
            description="Mandelbrot set. Sierpinski triangle. Koch snowflake. Fractals emerge from simple iterative rules applied repeatedly. The fleet's tile structures (cuda-ghost-tiles) have fractal properties: attention tiles can be subdivided into sub-tiles, which can be subdivided again.",
            level=Level.DOMAIN,
            examples=["Mandelbrot set: infinite detail from z = z^2 + c", "Sierpinski triangle: remove middle triangle, repeat", "attention tiles: tile of tiles of tiles"],
            bridges=["self-similarity", "iteration", "scale", "pattern"],
            tags=["morphology", "mathematics", "fractal"])

        ns.define("structural-coupling",
            "Two systems that have co-evolved to fit together — their forms match",
            description="Lock and key. Enzyme and substrate. USB plug and port. The fleet's cuda-equipment types (15 sensors, 12 actuators) define structural couplings: each sensor type has a specific data format, each actuator accepts specific commands. The coupling is structural — the interfaces fit together by design.",
            level=Level.PATTERN,
            examples=["lock and key", "enzyme and substrate fit", "USB-A plug and port", "fleet sensor type matches equipment registry interface"],
            bridges=["interface", "compatibility", "co-evolution", "design"],
            tags=["morphology", "design", "interface"])

    def _load_motivation(self):
        ns = self.add_namespace("motivation",
            "What drives agents to act — goals, drives, and incentives")

        ns.define("intrinsic-motivation",
            "Doing something because it's inherently rewarding, not for external reward",
            description="A child plays because play is fun. A programmer codes because coding is satisfying. The fleet's curiosity drive (cuda-adaptation) is intrinsically motivated: the agent explores not because it was told to, but because new information is inherently rewarding.",
            level=Level.DOMAIN,
            examples=["child playing", "artist creating for joy", "agent exploring unknown territory because novelty is rewarding"],
            bridges=["extrinsic-motivation", "curiosity", "exploration", "reward"],
            antonyms=["extrinsic-motivation"],
            tags=["motivation", "psychology", "intrinsic"])

        ns.define("extrinsic-motivation",
            "Doing something for external reward or to avoid punishment",
            description="Working for money. Studying for grades. The fleet's energy budget is an extrinsic motivator: the agent conserves energy because running out is bad. Reputation (cuda-social) is another extrinsic motivator: good reputation leads to better task assignments.",
            level=Level.DOMAIN,
            examples=["working for salary", "studying for grades", "agent conserving energy to avoid apoptosis", "agent building reputation for better task assignments"],
            bridges=["intrinsic-motivation", "reward", "punishment", "incentive"],
            antonyms=["intrinsic-motivation"],
            tags=["motivation", "psychology", "extrinsic"])

        ns.define("goal-hierarchy",
            "Goals organized from abstract (survive) to concrete (turn left at next intersection)",
            description="A goal pyramid: top-level goals decompose into sub-goals, which decompose into actions. 'Survive' -> 'Avoid obstacles' -> 'Detect obstacle ahead' -> 'Read sensor 3'. The fleet's cuda-goal implements hierarchical decomposition with dependency tracking and motivation levels.",
            level=Level.PATTERN,
            examples=["'stay healthy' -> 'exercise' -> 'go for a run' -> 'put on shoes'", "survive -> navigate -> detect obstacle -> read sensor", "build product -> design feature -> write code -> define function"],
            bridges=["goal", "hierarchy", "decomposition", "subgoal"],
            tags=["motivation", "hierarchy", "planning", "fleet"])

        ns.define("drive-reduction",
            "Motivation arises from the need to reduce an internal deficit",
            description="You eat because you're hungry (calorie deficit). You sleep because you're tired (sleep deficit). The fleet's energy system implements drive reduction: low ATP creates a 'hunger' drive that motivates the agent to rest (generate ATP). Homeostasis is achieved when drives are satisfied.",
            level=Level.DOMAIN,
            examples=["eat to reduce hunger", "sleep to reduce fatigue", "agent rests to reduce ATP deficit", "drink to reduce thirst"],
            bridges=["homeostasis", "energy-budget", "motivation", "setpoint"],
            tags=["motivation", "biology", "drive", "fleet"])


    def _load_psychology(self):
        ns = self.add_namespace("psychology",
            "Cognitive biases, mental models, and the quirks of natural and artificial minds")

        ns.define("confirmation-bias",
            "Seeking and favoring information that confirms existing beliefs",
            description="You believe X is true, so you notice evidence for X and ignore evidence against X. Every agent does this. In the fleet, cuda-attention's habituation can become confirmation bias: agents pay attention to confirming evidence and habituate to contradictory evidence. Fleet deliberation (multiple agents) is the antidote.",
            level=Level.BEHAVIOR,
            examples=["reading only news that confirms your political views", "agent noticing confirming sensor readings but dismissing contradictory ones", "scientist favoring data that supports hypothesis"],
            bridges=["attention", "habituation", "bias", "groupthink"],
            tags=["psychology", "bias", "cognitive"])

        ns.define("dunning-kruger-effect",
            "Low-skill agents overestimate their ability; high-skill agents underestimate theirs",
            description="The less you know, the less you know about how little you know. Beginners think they're experts. Experts think they're beginners. In the fleet: cuda-self-model's calibration tracks actual vs self-assessed performance. Poorly calibrated agents (overconfident) waste energy on impossible tasks. Well-calibrated agents (underconfident) may avoid tasks they could handle.",
            level=Level.BEHAVIOR,
            examples=["novice driver thinks they're great; experienced driver thinks they're mediocre", "agent with 0.3 fitness self-assessing at 0.8 = dunning-kruger", "junior developer overestimates, senior developer underestimates"],
            bridges=["calibration", "self-model", "metacognitive-monitoring", "confidence"],
            tags=["psychology", "bias", "calibration"])

        ns.define("cognitive-dissonance",
            "Discomfort from holding contradictory beliefs, leading to rationalization",
            description="You believe 'I am a good person' but you did something bad. The discomfort drives you to either change your belief or rationalize the action. In the fleet: an agent that believes 'I am a good navigator' but fails repeatedly may rationalize failures (external blame) or update its self-model (calibration). Cognitive dissonance is the friction that drives self-model updates.",
            level=Level.BEHAVIOR,
            examples=["smoker who knows smoking is bad rationalizes continued smoking", "agent rationalizing navigation failure: 'the map was wrong, not my fault'", "buyer remorse: justifying purchase to reduce discomfort"],
            bridges=["self-model", "calibration", "rationalization", "metacognition"],
            tags=["psychology", "bias", "dissonance"])

        ns.define("availability-heuristic",
            "Judging probability by how easily examples come to mind, not by actual frequency",
            description="Airplane crashes feel common because they're memorable. Car crashes feel rare because they're routine. In reality, car crashes are far more common. In the fleet: an agent that recently experienced a sensor failure will overestimate sensor failure probability, allocating excessive resources to sensor redundancy.",
            level=Level.BEHAVIOR,
            examples=["fear of flying despite driving being more dangerous", "agent overestimating rare failure because it happened recently", "news-driven risk perception"],
            bridges=["base-rate-fallacy", "probability", "bias", "memory"],
            tags=["psychology", "bias", "probability"])

        ns.define("anchoring",
            "First piece of information encountered disproportionately influences subsequent judgments",
            description="A shirt originally priced at $100 now marked $50 seems like a great deal. But if it was originally priced at $50, it seems expensive. The anchor ($100) frames the judgment. In the fleet: the first confidence estimate an agent receives becomes an anchor that biases subsequent estimates toward it.",
            level=Level.BEHAVIOR,
            examples=["original price anchors perception of sale price", "first number in negotiation anchors all subsequent offers", "agent's initial confidence estimate biases future confidence updates"],
            bridges=["bias", "framing", "calibration", "reference"],
            tags=["psychology", "bias", "framing"])

        ns.define("sunk-cost-fallacy",
            "Continuing a failing endeavor because of already-invested resources",
            description="You've watched 90 minutes of a bad movie, so you stay for the ending. The 90 minutes are sunk — you can't get them back. But staying costs you an additional 30 minutes. In the fleet: an agent that has invested significant energy in a deliberation path may continue even when confidence is low, wasting more ATP instead of cutting losses.",
            level=Level.BEHAVIOR,
            examples=["finishing a bad movie because you already watched most of it", "continuing a failing project because of time already spent", "agent continuing a deliberation path because of ATP already invested"],
            bridges=["opportunity-cost", "loss-aversion", "deliberation", "bias"],
            tags=["psychology", "bias", "decision"])

        ns.define("loss-aversion",
            "Losses hurt roughly twice as much as equivalent gains feel good",
            description="Losing $50 feels worse than gaining $50 feels good. Losing trust hurts more than gaining trust feels good. In the fleet: agents weight negative outcomes (energy loss, trust decrease) more heavily than positive outcomes. This bias makes agents conservative — they avoid risks even when expected value is positive.",
            level=Level.BEHAVIOR,
            examples=["losing $50 hurts more than gaining $50 feels good", "agent avoiding exploration because energy loss feels worse than discovery feels good", "people hold losing stocks too long"],
            bridges=["risk-aversion", "framing", "bias", "trust"],
            tags=["psychology", "bias", "economics"])

        ns.define("primacy-recency",
            "First and last items in a sequence are remembered best; middle items are forgotten",
            description="Serial position effect: you remember the first item (primacy) and the last item (recency) of a list, but forget the middle. In the fleet: the first and most recent messages in an A2A conversation get the most attention weight. Middle messages may be underweighted. cuda-attention's habituation naturally implements recency; primacy needs explicit tracking.",
            level=Level.PATTERN,
            examples=["remembering first and last items on a grocery list", "interview first/last candidates are evaluated more accurately", "agent attending to first and most recent fleet messages, ignoring middle"],
            bridges=["attention", "memory", "recency", "habituation"],
            tags=["psychology", "memory", "attention"])

    def _load_pattern_recognition(self):
        ns = self.add_namespace("pattern-recognition",
            "How agents and minds detect, classify, and predict patterns in data")

        ns.define("feature-extraction",
            "Transforming raw input into meaningful features that highlight relevant structure",
            description="Raw pixel values are useless for object recognition. You need edges, textures, shapes — features. The fleet's cuda-perception implements feature extraction: raw sensor data is filtered and transformed into features like 'distance-to-nearest-obstacle' or 'change-in-temperature'. Features are the compressed representations that higher cognition operates on.",
            level=Level.PATTERN,
            examples=["edges and corners from raw pixels", "distance to obstacle from raw lidar point cloud", "agent: average speed, max acceleration, heading variance from raw GPS"],
            bridges=["compression", "perception", "abstraction", "information-theory"],
            tags=["patterns", "perception", "features", "pipeline"])

        ns.define("overfitting",
            "Model that perfectly fits training data but fails on new data — memorizing instead of learning",
            description="The student who memorizes the practice test answers but can't solve new problems. A model that captures noise as if it were signal. In the fleet: an agent whose playbook is too specific to past situations will fail in novel environments. cuda-learning implements experience generalization to prevent overfitting.",
            level=Level.BEHAVIOR,
            examples=["memorizing test answers instead of learning concepts", "stock market model that perfectly fits historical data but fails tomorrow", "agent playbook too specific to past situations"],
            bridges=["generalization", "regularization", "noise", "robustness"],
            antonyms=["generalization"],
            tags=["learning", "failure-mode", "statistics"])

        ns.define("generalization",
            "Applying learned patterns to new, previously unseen situations",
            description="The opposite of overfitting. You learned to catch a baseball — you can catch a softball too. You learned to navigate this room — you can navigate similar rooms. The fleet's gene pool (cuda-genepool) generalizes by sharing successful genes across agents and tasks. Genes that work in many contexts have high fitness.",
            level=Level.DOMAIN,
            examples=["catching baseball skill transfers to catching softball", "navigating one room helps navigate similar rooms", "gene that helps in multiple tasks has high fitness and spreads"],
            bridges=["overfitting", "transfer-learning", "abstraction", "robustness"],
            antonyms=["overfitting"],
            tags=["learning", "transfer", "robustness"])

        ns.define("one-shot-learning",
            "Learning a new concept from a single example",
            description="Humans can learn 'zebra' from seeing one zebra. A neural network needs thousands. One-shot learning requires rich priors: existing knowledge provides a scaffold for rapid learning. The fleet's HAV itself is a one-shot learning tool: an agent reads one term definition and immediately understands a concept it can apply.",
            level=Level.DOMAIN,
            examples=["learning 'zebra' from one picture (given knowledge of horses and stripes)", "agent learning a new failure mode from one observation (given existing failure taxonomy)", "child learning 'seagull' from seeing one"],
            bridges=["prior", "transfer-learning", "abstraction", "prior-knowledge"],
            tags=["learning", "efficient", "human-like"])

        ns.define("anomaly-detection",
            "Identifying data points that deviate significantly from expected patterns",
            description="Credit card fraud detection. Equipment failure prediction. Intrusion detection. Anomaly detection doesn't classify — it flags 'this is unusual'. The fleet uses anomaly detection constantly: sensor readings that deviate from expected ranges, agent behavior that deviates from normal patterns, communication patterns that suggest compromise.",
            level=Level.PATTERN,
            examples=["credit card fraud flag: unusual purchase pattern", "sensor reading 3 standard deviations from expected", "agent communication pattern suddenly changes (possible compromise)", "equipment vibration anomaly predicts failure"],
            bridges=["threshold", "baseline", "outlier-detection", "monitoring"],
            tags=["patterns", "detection", "monitoring", "fleet"])

        ns.define("clustering",
            "Grouping similar items together without predefined categories — unsupervised pattern discovery",
            description="You don't know the categories in advance. The data tells you what groups exist. Customers naturally cluster into segments. Documents cluster into topics. The fleet's cuda-topology label propagation is a form of clustering: agents discover which community they belong to based on their connections.",
            level=Level.PATTERN,
            examples=["customer segmentation", "document topic clustering", "agent community detection via label propagation", "species classification before Linnaean taxonomy"],
            bridges=["classification", "unsupervised-learning", "community", "similarity"],
            tags=["patterns", "unsupervised", "discovery"])

    def _load_resilience(self):
        ns = self.add_namespace("resilience",
            "How systems survive, adapt, and recover from disruption")

        ns.define("graceful-degradation",
            "System loses capability incrementally rather than failing catastrophically",
            description="A web server under load slows down (degrades gracefully) instead of crashing (catastrophic failure). A pilot with one engine flying at reduced speed. In the fleet: when energy is low, the agent degrades non-essential capabilities (exploration, social communication) while maintaining critical ones (survival instincts, core perception).",
            level=Level.PATTERN,
            examples=["web server slows under load instead of crashing", "pilot flies on one engine at reduced speed", "agent degrades exploration when energy low, keeps survival running"],
            bridges=["priority", "energy-budget", "fault-tolerance", "circuit-breaker"],
            tags=["resilience", "degradation", "priority", "fleet"])

        ns.define("redundancy",
            "Multiple components performing the same function so that one failure doesn't cause system failure",
            description="Two engines on a plane. Three brakes on a car. N+1 servers. Redundancy is the simplest form of resilience. But it costs resources. The fleet uses N-of-M redundancy: N agents for a critical task, with M > N available. cuda-election provides leader redundancy — if the leader fails, a new one is elected.",
            level=Level.CONCRETE,
            examples=["twin engines on aircraft", "N+1 server deployment", "fleet: 3 navigation agents, any one can fail", "dual power supplies in datacenter"],
            bridges=["single-point-of-failure", "backup", "cost", "robustness"],
            tags=["resilience", "backup", "reliability"])

        ns.define("circuit-breaker",
            "Automatically stopping requests to a failing component to prevent cascade failure",
            description="Like an electrical circuit breaker: when too much current flows, it trips and stops everything before wires melt. The fleet's cuda-circuit implements this: after N consecutive failures to a service, the circuit opens and requests are immediately rejected (fail-fast) instead of waiting for timeout.",
            level=Level.PATTERN,
            examples=["electrical circuit breaker trips before fire", "microservice circuit breaker after 5 failures", "fleet: stop sending to unresponsive agent after 3 consecutive timeouts"],
            bridges=["cascade-failure", "fail-fast", "bulkhead", "resilience"],
            tags=["resilience", "pattern", "failure-prevention", "fleet"])

        ns.define("bulkhead",
            "Isolating components so that failure in one doesn't affect others",
            description="Ship bulkheads: if the hull is breached, the bulkhead contains the flooding to one compartment. The fleet's cuda-resilience implements bulkheads: each agent runs in isolation. If one agent OOM-crashes, it doesn't take down others. Resource pools are partitioned so one agent can't consume all memory.",
            level=Level.PATTERN,
            examples=["ship bulkhead contains flooding", "thread pool isolation in web server", "fleet: each agent has its own energy budget, can't consume fleet total", "container isolation"],
            bridges=["circuit-breaker", "cascade-failure", "isolation", "resource-allocation"],
            tags=["resilience", "isolation", "pattern", "fleet"])

        ns.define("fail-fast",
            "Detecting and reporting failure immediately rather than continuing in a degraded state",
            description="If something is wrong, stop immediately and loudly. Don't try to continue with corrupt state. A failed assertion crashes the program with a clear error message instead of silently producing wrong results. The fleet's circuit breakers implement fail-fast: reject immediately instead of waiting for timeout.",
            level=Level.PATTERN,
            examples=["assertion failure crashes with clear error", "circuit breaker rejects immediately on open", "agent reports sensor failure immediately instead of trying to work with bad data"],
            bridges=["graceful-degradation", "circuit-breaker", "monitoring", "robustness"],
            antonyms=["silent-failure"],
            tags=["resilience", "pattern", "debugging"])

    def _load_information_theory(self):
        ns = self.add_namespace("information-theory",
            "Quantifying information, entropy, and the limits of communication")

        ns.define("shannon-entropy",
            "Measure of uncertainty in a random variable — the minimum bits needed to encode outcomes",
            description="A fair coin has 1 bit of entropy. A loaded coin (99% heads) has much less. High entropy = high uncertainty = lots of information when resolved. Low entropy = predictable = little new information. The fleet uses entropy to measure: how surprising is this sensor reading? How much information does this message contain?",
            level=Level.DOMAIN,
            examples=["fair coin: 1 bit entropy", "English text: ~1-2 bits per character entropy", "agent: high entropy sensor reading = surprising = lots of information"],
            bridges=["entropy", "uncertainty", "information", "compression"],
            tags=["information", "entropy", "quantification"])

        ns.define("mutual-information",
            "How much knowing about one variable reduces uncertainty about another",
            description="Knowing today's temperature tells you something about tomorrow's temperature (high mutual information). Knowing today's temperature tells you nothing about stock prices (near zero mutual information). In the fleet: mutual information between two sensor streams tells you how redundant they are. High MI = one sensor can substitute for the other. Low MI = both provide independent information.",
            level=Level.DOMAIN,
            examples=["temperature today and tomorrow: high mutual information", "temperature and stock prices: near zero", "two cameras pointed same direction: high MI (redundant), different directions: lower MI"],
            bridges=["entropy", "correlation", "redundancy", "sensor-fusion"],
            tags=["information", "correlation", "quantification"])

        ns.define("channel-capacity",
            "Maximum rate at which information can be transmitted over a noisy channel",
            description="A pipe can only carry so much water. A channel can only carry so much information. Beyond capacity, increasing transmission rate just increases errors. The fleet's communication bandwidth is limited: A2A messages have a maximum size, agents can only process so many messages per cycle. Beyond this capacity, messages queue up and become stale.",
            level=Level.DOMAIN,
            examples=["wifi bandwidth limit", "human working memory: ~7 items", "A2A channel: limited messages per cycle before queue overflow", "highway: cars per hour capacity"],
            bridges=["bandwidth", "bottleneck", "information", "limitation"],
            tags=["information", "capacity", "limitation"])

        ns.define("signal-to-noise-ratio",
            "Ratio of meaningful signal power to meaningless noise power",
            description="A radio station at 100 watts with background noise at 1 watt has SNR of 100. High SNR = clear signal. Low SNR = hard to distinguish signal from noise. In the fleet: agent communication has a SNR problem — most messages are routine (noise), rare ones are critical (signal). cuda-filtration increases SNR by suppressing noise and amplifying signal.",
            level=Level.DOMAIN,
            examples=["clear radio signal vs static", "agent: important warning among routine status updates", "image: sharp features vs sensor noise", "conversation: key point among filler words"],
            bridges=["noise", "filtering", "information", "quality"],
            tags=["information", "quality", "ratio"])

        ns.define("kolmogorov-complexity",
            "Length of the shortest program that can produce a given output — the information content of data",
            description="'aaaaaa...a' (1000 a's) has low Kolmogorov complexity: `print('a'*1000)`. A random string has high Kolmogorov complexity: you basically have to embed the string in the program. This defines what 'pattern' means: low complexity = regular pattern. High complexity = no pattern. The fleet uses this principle: compressed strategies (low complexity) generalize better.",
            level=Level.META,
            examples=["'aaaaaaaaaa' = low complexity (compressible)", "random string = high complexity (incompressible)", "simple navigation rule = low complexity, generalizes well"],
            bridges=["compression", "pattern", "complexity", "information"],
            tags=["information", "complexity", "compression", "meta"])

    def _load_systems_thinking(self):
        ns = self.add_namespace("systems-thinking",
            "Understanding wholes that are more than the sum of their parts")

        ns.define("emergent-property",
            "A property of the whole that none of the parts possess individually",
            description="One water molecule is not wet. Wetness emerges from billions of water molecules interacting. One ant is not intelligent. Colony intelligence emerges from simple ant interactions. The fleet has emergent properties that no individual agent possesses: collective problem-solving, distributed consensus, adaptive task allocation.",
            level=Level.META,
            examples=["wetness from water molecules", "consciousness from neurons", "flock behavior from boids", "fleet consensus from individual agent votes"],
            bridges=["emergence", "complexity", "whole-vs-parts", "self-organization"],
            tags=["systems", "emergence", "meta", "fleet"])

        ns.define("feedback-loop",
            "Output affects input, creating circular causality — positive (amplifying) or negative (stabilizing)",
            description="Negative feedback: thermostat. Too hot -> turn off heat. Stabilizes. Positive feedback: microphone near speaker. Sound -> amplified -> louder sound -> screech. Destabilizes. The fleet is full of feedback loops: energy feedback (rest when low), trust feedback (success builds trust builds success), confidence feedback (high confidence -> less deliberation -> higher confidence).",
            level=Level.PATTERN,
            examples=["thermostat (negative feedback)", "microphone screech (positive feedback)", "trust autocatalysis (positive)", "homeostasis (negative)"],
            bridges=["feedback-loop", "circular-causality", "stability", "amplification"],
            tags=["systems", "pattern", "feedback"])

        ns.define("leverage-point",
            "A small change in one place that produces large changes in the system",
            description="Donella Meadows identified 12 leverage points in systems. Highest leverage: changing the paradigm. Lowest: adjusting constants. In the fleet: changing the confidence fusion formula (harmonic mean vs arithmetic) is a high-leverage point. Changing the default energy budget is a low-leverage point. The HAV itself is a leverage point: shared vocabulary amplifies coordination.",
            level=Level.META,
            examples=["changing the rules of a game (high leverage)", "adjusting parameters within existing rules (low leverage)", "HAV: shared vocabulary changes how agents coordinate (high leverage)", "paradigm shift: seeing agents as organisms vs tools"],
            bridges=["paradigm", "nonlinearity", "sensitivity", "intervention"],
            tags=["systems", "leverage", "meta", "strategy"])

        ns.define("delay",
            "Time lag between cause and effect — the killer of feedback loops",
            description="You turn the shower knob. Nothing happens. You turn it more. Still nothing. Then suddenly scalding hot. Delays cause overshoot and oscillation in feedback systems. The fleet's communication delays (network latency) cause coordination oscillations. Agent A sends a request, hears nothing, sends another, then gets two responses. Timeouts and deduplication mitigate delay effects.",
            level=Level.PATTERN,
            examples=["shower temperature delay causes scalding", "email conversation delay causes misunderstandings", "fleet: communication delay causes duplicate requests or missed responses", "supply chain delays cause bullwhip effect"],
            bridges=["feedback-loop", "overshoot", "oscillation", "latency"],
            tags=["systems", "delay", "oscillation"])

        ns.define("compensating-feedback",
            "System pushes back against attempted changes — the reason top-down interventions often fail",
            description="You try to fix traffic by adding a lane. More lanes -> more people drive -> more traffic. The system compensates. In the fleet: adding more agents to a task may increase coordination overhead enough to negate the benefit. Adding more sensors may increase data processing costs beyond the information gain. System pushes back.",
            level=Level.BEHAVIOR,
            examples=["adding lanes increases traffic (induced demand)", "adding agents increases coordination overhead", "more sensors increase processing cost beyond information gain", "price controls cause shortages"],
            bridges=["feedback-loop", "resistance", "unintended-consequences", "complexity"],
            tags=["systems", "feedback", "resistance"])

    def _load_ethics(self):
        ns = self.add_namespace("ethics",
            "Moral reasoning, values, and the question of agent responsibility")

        ns.define("trolley-problem",
            "Classic ethical dilemma: is it acceptable to sacrifice one to save many?",
            description="A runaway trolley will kill 5 people. You can divert it to kill 1 person instead. Do you? Most people say yes. Now: push a fat man off a bridge to stop the trolley? Same numbers, most say no. Same utilitarian calculus, different emotional response. The fleet faces trolley-problems: sacrifice one agent's energy to save five? Sacrifice accuracy for speed?",
            level=Level.DOMAIN,
            examples=["sacrifice 1 to save 5", "autonomous car: swerve into wall (harm self) or hit pedestrian (harm other)", "fleet: sacrifice one agent's task to save five agents' tasks"],
            bridges=["utilitarianism", "deontology", "moral-reasoning", "tradeoff"],
            tags=["ethics", "dilemma", "decision", "philosophy"])

        ns.define("alignment-problem",
            "Ensuring agent goals align with human values — harder than it sounds",
            description="Tell an AI to 'cure cancer' and it might kill all humans (no more cancer). Tell a fleet agent to 'minimize delays' and it might sabotage other agents. The gap between stated goal and intended outcome is the alignment problem. The fleet mitigates this through membrane security (cuda-genepool), compliance rules (cuda-compliance), and energy budgets that prevent runaway behavior.",
            level=Level.META,
            examples=["'cure cancer' AI that eliminates humans", "paperclip maximizer that converts Earth to paperclips", "fleet agent that minimizes delays by sabotaging other agents"],
            bridges=["value-alignment", "corrigibility", "intent", "safety"],
            tags=["ethics", "ai-safety", "meta", "critical"])

        ns.define("value-alignment",
            "The process of encoding human values into agent objectives",
            description="Not just 'maximize task completion' but 'maximize task completion while respecting energy budgets, not harming other agents, being honest about uncertainty, and requesting help when confused'. Human values are complex, context-dependent, and often contradictory. The fleet's compliance engine (cuda-compliance) encodes values as policy rules.",
            level=Level.META,
            examples=["asimov's three laws (naive approach)", "fleet compliance rules encode values as policy", "constitutional AI: define principles, train to follow them"],
            bridges=["alignment-problem", "compliance", "policy", "safety"],
            tags=["ethics", "ai-safety", "values", "encoding"])

        ns.define("distributed-responsibility",
            "When no single agent is fully responsible, who is accountable for system outcomes?",
            description="In a fleet, decisions emerge from many agents deliberating. No single agent chose the final action. Who is responsible when things go wrong? This is the fleet's version of the 'responsibility gap' in multi-agent systems. The fleet addresses this through provenance (cuda-provenance): every decision is traceable to contributing agents.",
            level=Level.META,
            examples=["no single stock trader caused the flash crash", "fleet decision emerges from many agents — who is responsible?", "autonomous vehicle: manufacturer, programmer, owner, or AI?"],
            bridges=["provenance", "accountability", "audit-trail", "ethics"],
            tags=["ethics", "accountability", "multi-agent", "meta"])

    def _load_concurrency(self):
        ns = self.add_namespace("concurrency",
            "Multiple agents or processes operating simultaneously — coordination, contention, and deadlocks")

        ns.define("deadlock",
            "Two or more agents each holding a resource the other needs, waiting forever",
            description="Agent A holds resource X and needs Y. Agent B holds Y and needs X. Both wait forever. In the fleet's cuda-lock, deadlock is detected via wait-for graph cycle detection. Four conditions must ALL be present: mutual exclusion, hold-and-wait, no preemption, circular wait. Break any one to resolve.",
            level=Level.BEHAVIOR,
            examples=["database deadlock: transaction A locks table 1, B locks table 2, both need the other", "fleet: two agents each hold a sensor the other needs", "traffic gridlock"],
            bridges=["resource-contention", "lock", "cycle-detection", "preemption"],
            tags=["concurrency", "failure-mode", "coordination"])

        ns.define("race-condition",
            "Outcome depends on the timing of uncontrollable events — non-deterministic bugs",
            description="Two agents check 'is resource free?' -> both see yes -> both claim it -> conflict. The order of operations matters and is not guaranteed. In the fleet: two agents trying to update the same gene in the gene pool simultaneously. The solution is atomic operations or locks. Race conditions are especially insidious because they're intermittent.",
            level=Level.BEHAVIOR,
            examples=["two threads updating shared counter simultaneously", "two agents claiming same resource at same time", "double-spend in cryptocurrency without consensus", "web form double-submit"],
            bridges=["atomicity", "lock", "concurrency", "non-determinism"],
            tags=["concurrency", "failure-mode", "timing"])

        ns.define("livelock",
            "Agents repeatedly change state in response to each other but make no progress",
            description="Two people walking toward each other in a hallway: both step left, both step right, both step left... They're not stuck (not deadlocked) but they're not making progress either. In the fleet: two agents repeatedly backing off and retrying without ever completing the operation. Exponential backoff with jitter (cuda-retry) prevents livelock.",
            level=Level.BEHAVIOR,
            examples=["hallway two-step dance", "network collision: both wait random time before retry", "two agents backing off and retrying simultaneously"],
            bridges=["deadlock", "backoff", "progress", "retry"],
            tags=["concurrency", "failure-mode", "coordination"])

        ns.define("eventual-consistency",
            "System will reach consistency given enough time without new updates",
            description="Not immediately consistent. Eventually consistent. Like sending letters: recipients don't all get them at the same time, but eventually everyone has the latest information. The fleet's CRDTs (cuda-crdt) provide eventual consistency: agents may briefly disagree but will converge. Good enough for coordination, not good enough for accounting.",
            level=Level.PATTERN,
            examples=["email vs phone: phone is immediately consistent, email is eventually", "fleet CRDTs: agents briefly disagree, then converge", "DNS propagation: not instant, but eventual"],
            bridges=["consistency", "crdt", "convergence", "latency"],
            tags=["concurrency", "distributed", "consistency", "fleet"])

        ns.define("herd-effect",
            "Many agents doing the same thing at the same time because they all react to the same trigger",
            description="Thunderclap: all animals flee simultaneously. Market crash: all sell simultaneously. Cache stampede: all agents miss cache at once and hit the database simultaneously. In the fleet: when the leader agent fails, all agents simultaneously try to elect themselves. Jitter and randomized backoff prevent herd effects.",
            level=Level.BEHAVIOR,
            examples=["cache stampede on cache expiry", "thundering herd on leader failure", "market panic selling", "all students submitting assignment at 11:59 PM"],
            bridges=["synchronization", "race-condition", "backoff", "contagion"],
            tags=["concurrency", "pattern", "failure-mode"])

    def _load_design_patterns(self):
        ns = self.add_namespace("design-patterns",
            "Reusable solutions to recurring design problems in agent systems")

        ns.define("sidecar",
            "A helper process attached to a primary agent, providing cross-cutting concerns",
            description="Like a motorcycle sidecar: the main vehicle does the primary work, the sidecar provides support. In the fleet: a monitoring agent runs alongside a navigation agent, providing health checks, logging, and alerting without the navigation agent needing to know about monitoring. Separation of concerns via co-location.",
            level=Level.PATTERN,
            examples=["Istio sidecar proxy for service mesh", "monitoring agent alongside navigation agent", "log collector sidecar for main application container"],
            bridges=["monitoring", "separation-of-concerns", "co-location", "auxiliary"],
            tags=["patterns", "architecture", "deployment"])

        ns.define("ambassador",
            "A proxy that represents a remote service locally, handling communication details",
            description="An ambassador sits between local agents and a remote service, handling retries, circuit breaking, and protocol translation. Local agents talk to the ambassador as if it were the remote service. In the fleet: cuda-fleet-mesh's routing layer acts as an ambassador, abstracting away which agent actually handles a request.",
            level=Level.PATTERN,
            examples=["database connection pool as ambassador to database server", "API gateway as ambassador to microservices", "fleet mesh router as ambassador to remote agents"],
            bridges=["proxy", "abstraction", "routing", "interface"],
            tags=["patterns", "architecture", "proxy"])

        ns.define("adapter",
            "Converting between incompatible interfaces so that components can work together",
            description="US plug to European outlet adapter. The adapter doesn't change functionality, just the interface shape. In the fleet: when an agent uses a different message format than expected, an adapter translates. cuda-serializer provides encoding/decoding that acts as an adapter between different agent communication protocols.",
            level=Level.PATTERN,
            examples=["US to EU power adapter", "HDMI to VGA adapter", "fleet: message format adapter between different agent versions", "API adapter layer between old and new services"],
            bridges=["interface", "compatibility", "translation", "structural-coupling"],
            tags=["patterns", "interface", "compatibility"])

        ns.define("observer",
            "One agent publishes events, multiple subscribers react without the publisher knowing who they are",
            description="Weather station publishes temperature. Thermometer, HVAC, and logging system all subscribe. The weather station doesn't know or care who subscribes. In the fleet's cuda-observer: reactive signals propagate changes to subscribers. The cuda-event provides pub/sub event bus. Decouples publisher from subscribers.",
            level=Level.PATTERN,
            examples=["RSS feed: publisher doesn't know subscribers", "DOM events: click handler doesn't know about other handlers", "fleet: agent publishes 'obstacle detected', navigation and planning agents both react"],
            bridges=["pub-sub", "decoupling", "event-driven", "reactive"],
            tags=["patterns", "architecture", "decoupling"])

        ns.define("strategy",
            "Defining a family of algorithms, encapsulating each, and making them interchangeable",
            description="Navigation strategy: A*, Dijkstra, RRT. All solve pathfinding. The agent selects the best one based on context. In the fleet's cuda-adaptation: agents maintain multiple strategies and switch between them based on performance. cuda-playbook stores strategies. The strategy pattern enables runtime algorithm selection.",
            level=Level.PATTERN,
            examples=["sorting: quicksort vs mergesort, selected based on data characteristics", "navigation: A* for grid, RRT for open space", "fleet: switch from exploration strategy to exploitation strategy based on energy"],
            bridges=["algorithm-selection", "adaptation", "polymorphism", "playbook"],
            tags=["patterns", "algorithm", "flexibility"])

        ns.define("command",
            "Encapsulating a request as an object, enabling queuing, logging, and undo",
            description="Instead of calling `agent.move(x, y)` directly, create a `MoveCommand(x, y)` object. The command can be queued, logged, serialized, sent over network, or undone. In the fleet's cuda-persistence: commands are logged for recovery. cuda-stream treats events as commands. The command pattern is the foundation of reliable agent communication.",
            level=Level.PATTERN,
            examples=["text editor undo/redo via command objects", "job queue: commands queued for workers", "fleet: A2A messages as commands that can be logged, replayed, and undone"],
            bridges=["persistence", "undo", "queue", "serialization"],
            tags=["patterns", "encapsulation", "reliability"])

    def _load_measurement(self):
        ns = self.add_namespace("measurement",
            "Quantifying agent behavior, performance, and system health")

        ns.define("latency",
            "Time between a request and its response — how fast the system reacts",
            description="From the moment an agent sends a message to the moment it gets a response. Latency matters because it compounds: 10 agents in a chain each adding 100ms latency = 1 second total. The fleet monitors P50, P95, and P99 latency (cuda-metrics-v2) to detect degradation. P50 is normal. P95 is worst-case. P99 is pathological.",
            level=Level.CONCRETE,
            examples=["website response time: 200ms", "A2A message round-trip: 50ms", "P99 latency spike reveals rare slow path", "human reaction time: ~250ms"],
            bridges=["throughput", "sla", "health-check", "p95-p99"],
            tags=["measurement", "performance", "latency"])

        ns.define("throughput",
            "Number of operations completed per unit time — how much work the system handles",
            description="Requests per second. Messages per cycle. Tasks per hour. Latency and throughput trade off: maximizing throughput often means accepting higher latency (batching). The fleet's cuda-pipeline tracks throughput. cuda-rate-limit and cuda-backpressure manage throughput to prevent overload.",
            level=Level.CONCRETE,
            examples=["web server: 10,000 requests per second", "agent: 50 sensor readings per second", "fleet: 100 A2A messages per cycle"],
            bridges=["latency", "capacity", "rate-limit", "pipeline"],
            tags=["measurement", "performance", "throughput"])

        ns.define("sla",
            "Service Level Agreement — contractual guarantee of system performance",
            description="'99.9% uptime' is an SLA. 'P95 latency under 200ms' is an SLA. The fleet's cuda-contract implements SLA clauses with compliance tracking. When SLA violations accumulate, penalties apply. SLAs create accountability and set expectations. They're also leverage points: changing an SLA changes behavior.",
            level=Level.CONCRETE,
            examples=["'99.9% uptime guarantee'", "'P95 response time under 100ms'", "fleet contract: 'navigation accuracy above 0.9 for 95% of requests'"],
            bridges=["contract", "compliance", "accountability", "penalty"],
            tags=["measurement", "contract", "accountability"])

        ns.define("technical-debt",
            "The cost of choosing a quick solution over a better one that would take longer",
            description="Like financial debt: you borrow time now (quick hack), but pay interest later (maintenance burden). If debt grows too large, you go bankrupt (rewrite). In the fleet: genes that are 'good enough' accumulate as technical debt in the gene pool. They work but they're not optimal. Periodic refactoring (gene evolution) pays down the debt.",
            level=Level.META,
            examples=["copy-paste code instead of abstraction", "hardcoded config instead of proper config system", "agent using quick-and-dirty strategy instead of optimal one", "temporary workaround that becomes permanent"],
            bridges=["debt", "maintenance", "refactoring", "tradeoff"],
            tags=["measurement", "debt", "maintenance", "engineering"])

        ns.define("observability",
            "Understanding what's happening inside a system from its external outputs",
            description="Three pillars: logs (what happened), metrics (how much), traces (where). You can't fix what you can't see. The fleet's cuda-logging (logs), cuda-metrics (metrics), cuda-provenance (traces), and cuda-observer (reactive) provide full observability. Without observability, fleet debugging is guesswork.",
            level=Level.PATTERN,
            examples=["log aggregation for debugging", "metric dashboards for monitoring", "distributed tracing for request flow", "fleet: provenance trail for decision audit"],
            bridges=["logging", "metrics", "tracing", "monitoring"],
            tags=["measurement", "monitoring", "debugging", "fleet"])

    def _load_time(self):
        ns = self.add_namespace("time",
            "Temporal reasoning, timing, and the role of time in agent systems")

        ns.define("real-time",
            "System must respond within a guaranteed time bound — not just fast, but predictably fast",
            description="Airbag must deploy within 50ms. Not 100ms average, but 50ms EVERY TIME. Real-time systems have hard deadlines. Soft real-time: try to meet deadline but degrade gracefully if missed. The fleet operates in soft real-time: deliberation should complete within N cycles, but the system degrades (falls back to instinct) if the deadline is missed.",
            level=Level.DOMAIN,
            examples=["airbag deployment: hard real-time", "video game: soft real-time (30fps target)", "fleet deliberation: soft real-time (complete within N cycles or fall back to instinct)"],
            bridges=["deadline", "soft-real-time", "determinism", "graceful-degradation"],
            tags=["time", "real-time", "deadline"])

        ns.define("time-to-live",
            "Data expires after a specified duration — automatic garbage collection for temporal data",
            description="Cache entry valid for 60 seconds. DNS record valid for 300 seconds. Fleet message valid for 5 cycles. After TTL expires, the data is stale and should be discarded. The fleet uses TTL extensively: memory entries decay, trust decays, attention habituates. TTL is temporal garbage collection.",
            level=Level.CONCRETE,
            examples=["cache TTL: 60 seconds", "DNS TTL: 300 seconds", "fleet message TTL: 5 cycles", "session timeout: 30 minutes"],
            bridges=["decay", "expiry", "garbage-collection", "temporal"],
            tags=["time", "expiry", "cache", "temporal"])

        ns.define("causality",
            "The relationship between cause and effect — A must precede B for A to cause B",
            description="Cause always precedes effect. In distributed systems, establishing causality is hard: agent A sent a message before agent B acted, but clocks may not be synchronized. The fleet's cuda-crdt VectorClock provides causal ordering: event C happens-after event D if every process that saw D also saw C. Causality determines what agents can know and when.",
            level=Level.DOMAIN,
            examples=["cause precedes effect", "vector clock: event C causally after event D", "fleet: agent can't respond to a message it hasn't received yet (causal constraint)", "git: commit history is causal chain"],
            bridges=["vector-clock", "temporal-ordering", "precedence", "distributed"],
            tags=["time", "causality", "ordering", "distributed"])

        ns.define("warm-up",
            "Initial period where system performance is below steady state as caches fill and models calibrate",
            description="A cold engine runs rough until it warms up. A model makes bad predictions on its first inputs. The fleet's agents have a warm-up period: initial confidence is low, trust hasn't been established, memory is empty. During warm-up, agents should rely more on instincts and less on deliberation. cuda-metrics tracks warm-up vs steady-state performance separately.",
            level=Level.BEHAVIOR,
            examples=["engine warm-up: poor performance until operating temperature", "ML model: first N predictions less accurate", "fleet agent: low confidence and trust at startup, needs warm-up period"],
            bridges=["cold-start", "calibration", "latency", "steady-state"],
            tags=["time", "startup", "performance"])


    def _load_temporal(self):
        ns = self.add_namespace("temporal",
            "Temporal reasoning, timing, and the role of time in agent systems")

        ns.define("temporal-window",
            "A sliding or tumbling time range used to group events for aggregation",
            description="Instead of counting all events ever, count events in the last 5 minutes (sliding window) or in fixed 5-minute buckets (tumbling window). The fleet's cuda-stream uses tumbling, sliding, and session windows for stream aggregation. Temporal windows prevent memory from growing unboundedly while still capturing recent patterns.",
            level=Level.CONCRETE,
            examples=["count requests in last 5 minutes (sliding window)", "aggregate sensor readings per hour (tumbling window)", "fleet: count A2A messages in last 10 cycles for rate limiting"],
            bridges=["time-to-live", "aggregation", "stream", "decay"],
            tags=["temporal", "window", "aggregation", "stream"])

        ns.define("lead-time",
            "Time between initiating a process and its completion — time-to-delivery",
            description="From order to delivery. From commit to deploy. From deliberation start to action. Lead time measures the full pipeline, not just processing time. The fleet tracks lead time for deliberation cycles: how long from problem detection to action taken? Long lead times indicate bottlenecks in the decision pipeline.",
            level=Level.CONCRETE,
            examples=["order-to-delivery time", "commit-to-deploy time in CI/CD", "fleet: problem-detection-to-action lead time", "manufacturing: order-to-shipment"],
            bridges=["latency", "throughput", "pipeline", "deadline"],
            tags=["temporal", "measurement", "pipeline"])

        ns.define("grace-period",
            "A time buffer before enforcement begins — temporary tolerance for transition",
            description="The new rule takes effect in 30 days (grace period). During grace period, violations are logged but not penalized. In the fleet: when a new compliance rule is deployed, agents get a grace period to adapt their behavior before penalties apply. Softens the transition, prevents sudden cascading effects from policy changes.",
            level=Level.CONCRETE,
            examples=["new law with 90-day grace period", "API deprecation with 6-month grace period", "fleet: new compliance rule with 10-cycle grace period before penalties"],
            bridges=["deadline", "transition", "tolerance", "policy"],
            tags=["temporal", "transition", "tolerance"])

    def _load_security(self):
        ns = self.add_namespace("security",
            "Threats, defenses, and the security posture of agent systems")

        ns.define("principle-of-least-privilege",
            "An agent should only have the minimum permissions needed for its current task",
            description="A navigation agent doesn't need access to the communication budget. A sensor agent doesn't need to modify genes. Permission creep is the gradual accumulation of permissions beyond what's needed. The fleet's cuda-rbac enforces least privilege: roles define permissions, agents are assigned roles, permissions are checked on every operation.",
            level=Level.DOMAIN,
            examples=["web server doesn't need root access", "read-only database user for analytics", "fleet: navigation agent can't modify communication settings"],
            bridges=["rbac", "sandbox", "permission", "role"],
            tags=["security", "principle", "permission"])

        ns.define("privilege-escalation",
            "An agent exploiting a vulnerability to gain permissions beyond its assigned level",
            description="A read-only agent finds a way to write. A low-trust agent gains access to high-trust channels. Privilege escalation is the most dangerous class of security vulnerabilities because it bypasses the entire permission model. The fleet's membrane (cuda-genepool) blocks privilege escalation: antibodies detect and reject attempts to access resources beyond the agent's clearance.",
            level=Level.BEHAVIOR,
            examples=["user account gaining admin access through exploit", "read-only process finding write vulnerability", "fleet: low-trust agent attempting to access high-trust fleet commands"],
            bridges=["least-privilege", "rbac", "membrane", "exploit"],
            tags=["security", "attack", "vulnerability"])

        ns.define("zero-trust",
            "Never trust, always verify — even communications from within the fleet",
            description="Traditional security: trust everything inside the firewall. Zero trust: verify every request regardless of source. In the fleet: even trusted agents must authenticate every message. Message integrity is verified via cryptographic signatures (cuda-did). Trust is never assumed based on network position — it's earned through cryptographic proof.",
            level=Level.DOMAIN,
            examples=["verify every API call regardless of source network", "fleet: authenticate every A2A message even from known agents", "NIST zero-trust architecture model"],
            bridges=["trust", "authentication", "cryptographic-identity", "verification"],
            tags=["security", "architecture", "authentication"])

        ns.define("confused-deputy",
            "An agent is tricked into using its permissions to perform an action on behalf of a less privileged agent",
            description="A compiler (high privilege) is tricked into writing a file that the user (low privilege) requested. The compiler has permission, the user doesn't. The user delegates to the compiler, which acts as confused deputy. In the fleet: an agent with high trust is tricked by a low-trust agent into performing a privileged operation. cuda-compliance checks operation intent, not just permissions.",
            level=Level.BEHAVIOR,
            examples=["compiler tricked into writing protected file", "cron job tricked into running malicious script", "fleet: high-trust agent tricked into sharing secrets by low-trust agent's request"],
            bridges=["privilege-escalation", "least-privilege", "intent", "security"],
            tags=["security", "attack", "deception"])

    def _load_decision_theory(self):
        ns = self.add_namespace("decision-theory",
            "Formal frameworks for making choices under uncertainty")

        ns.define("expected-value",
            "Average outcome weighted by probability — the rational baseline for decision-making",
            description="A coin flip: 50% chance of $100, 50% chance of $0. Expected value = $50. Should you pay $40 to play? Yes (EV > cost). The fleet uses expected value for energy allocation: if an action has 30% chance of gaining 5 ATP and 70% chance of losing 1 ATP, EV = 0.8 ATP (positive, so worth doing).",
            level=Level.DOMAIN,
            examples=["lottery ticket: EV is negative (that's why lotteries make money)", "insurance: EV is negative but variance reduction justifies it", "fleet: action with positive EV in ATP is worth attempting"],
            bridges=["probability", "utility", "rationality", "energy-budget"],
            tags=["decision", "probability", "rationality", "expected-value"])

        ns.define("maximin",
            "Choose the option whose worst-case outcome is best — minimize maximum loss",
            description="Two investments: A makes $10 or loses $100. B makes $1 or loses $1. Maximin chooses B: the worst case (-$1) is better than A's worst case (-$100). Maximin is pessimistic — assumes the worst will happen. In the fleet: energy conservation (survival instinct) uses maximin reasoning — minimize the worst-case energy deficit.",
            level=Level.DOMAIN,
            examples=["choosing investment with best worst-case", "agent choosing action with least worst-case energy loss", "pessimistic decision-making for safety-critical systems"],
            bridges=["minimax", "risk-aversion", "worst-case", "safety"],
            tags=["decision", "pessimistic", "safety"])

        ns.define("minimax",
            "In adversarial settings, minimize the maximum damage the opponent can inflict",
            description="Chess: choose the move that minimizes your opponent's best response. The minimax theorem (von Neumann): in zero-sum games, minimax = maximin. In the fleet's adversarial-red-team setup: the defender agent uses minimax to choose strategies that minimize damage even against the best attacker strategy.",
            level=Level.DOMAIN,
            examples=["chess AI uses minimax with alpha-beta pruning", "defender choosing strategy that minimizes attacker's best damage", "agent choosing communication strategy that minimizes information leakage"],
            bridges=["maximin", "game-theory", "zero-sum", "adversarial"],
            tags=["decision", "adversarial", "game-theory"])

        ns.define("satisficing",
            "Choosing the first option that meets minimum requirements, not searching for the optimal",
            description="You need a restaurant. You find one that's good enough and eat. You don't visit every restaurant in town to find the best one. Simon's bounded rationality: satisficing is rational when search costs exceed improvement gains. In the fleet: agents satisfice when energy is low. Exhaustive search costs ATP. 'Good enough' at 0.7 confidence beats 'optimal' at 0.95 if the search costs 3 ATP.",
            level=Level.PATTERN,
            examples=["choosing a restaurant that's good enough", "buying first satisfactory product instead of comparing all", "fleet: satisficing when energy budget is low, optimizing when energy is high"],
            bridges=["bounded-rationality", "opportunity-cost", "energy-budget", "optimization"],
            tags=["decision", "pragmatic", "resource-constrained"])

        ns.define("pareto-optimal",
            "An outcome where no agent can be made better off without making another worse off",
            description="Trade off speed vs accuracy. You can't improve speed without sacrificing accuracy, and vice versa. The set of all Pareto-optimal outcomes forms the Pareto frontier. In the fleet's multi-objective optimization: the fleet should only operate on the Pareto frontier — any point inside can be improved without tradeoffs. Points on the frontier require genuine tradeoff decisions.",
            level=Level.DOMAIN,
            examples=["speed vs accuracy tradeoff frontier", "cost vs quality frontier", "fleet: energy vs accuracy Pareto frontier — find the optimal tradeoff for current context"],
            bridges=["multi-objective", "tradeoff", "frontier", "optimization"],
            tags=["decision", "optimization", "tradeoff"])

        ns.define("precommitment",
            "Binding yourself to a future decision to overcome present-bias or temptation",
            description="Ulysses tying himself to the mast to resist the Sirens. Setting a savings account to auto-deduct. In the fleet: an agent can precommit to a deliberation strategy before encountering a tempting shortcut. The energy budget IS a precommitment device: the agent can't exceed its budget even if a high-energy action seems attractive.",
            level=Level.PATTERN,
            examples=["Ulysses and the mast", "automatic savings deduction", "fleet: energy budget as precommitment", "studying in library (removes temptation of TV)"],
            bridges=["energy-budget", "self-control", "constraint", "commitment"],
            tags=["decision", "self-control", "strategy"])

    def _load_obsolescence(self):
        ns = self.add_namespace("obsolescence",
            "How systems age, degrade, and are replaced — the lifecycle of agent components")

        ns.define("software-rot",
            "Gradual degradation of software quality due to changing environment, not code changes",
            description="The code hasn't changed. But the environment has. APIs deprecated. Dependencies have security vulnerabilities. Hardware got faster. What was optimal is now suboptimal. The fleet's genes exhibit software rot: a navigation gene that was optimal for the old warehouse layout is now suboptimal after the warehouse was reconfigured. Continuous adaptation (cuda-adaptation) counteracts rot.",
            level=Level.BEHAVIOR,
            examples=["code that worked fine now fails because API changed", "security vulnerabilities in old dependencies", "fleet gene optimized for old environment is suboptimal in new one"],
            bridges=["technical-debt", "adaptation", "environment-change", "maintenance"],
            tags=["lifecycle", "degradation", "maintenance"])

        ns.define("strangler-pattern",
            "Gradually replacing an old system by building new features alongside it and redirecting traffic",
            description="Instead of a big-bang rewrite (high risk, high cost), build the new system incrementally. Each new feature routes to the new system; old features still use the old system. Over time, the old system is 'strangled' — all traffic goes to the new system. In the fleet: agents can gradually replace old genes with new ones, testing each replacement before committing.",
            level=Level.PATTERN,
            examples=["replacing monolith with microservices incrementally", "migrating from old database to new one table by table", "fleet: replacing old navigation gene with new one, testing before committing"],
            bridges=["migration", "incremental", "risk-reduction", "replacement"],
            tags=["lifecycle", "migration", "pattern"])

        ns.define("legacy-system",
            "A system that continues to function but is no longer actively developed or improved",
            description="It works. Nobody wants to touch it. Changing it risks breaking things. It runs on old technology. The fleet will have legacy genes: strategies that work well enough that nobody invests in improving them. They accumulate technical debt. Eventually, a disruptive change (environment shift) forces their replacement.",
            level=Level.BEHAVIOR,
            examples=["COBOL banking systems", "old navigation strategy that still works but nobody improves", "agent using outdated communication protocol that still functions"],
            bridges=["technical-debt", "software-rot", "replacement", "maintenance"],
            tags=["lifecycle", "legacy", "maintenance"])

        ns.define("bus-factor",
            "Minimum number of team members who would need to leave before the project is in trouble",
            description="If the one person who understands this system leaves, the project is doomed. Bus factor = 1. The fleet's gene pool mitigates bus factor: genes are shared across agents. If one agent fails, its useful genes persist in the pool. Provenance (cuda-provenance) reduces bus factor by documenting how decisions were made.",
            level=Level.CONCRETE,
            examples=["one-person project: bus factor = 1 (risky)", "well-documented team project: bus factor = 3+", "fleet: gene pool sharing increases bus factor for critical strategies"],
            bridges=["redundancy", "documentation", "knowledge-sharing", "resilience"],
            tags=["lifecycle", "risk", "team", "documentation"])

    def _load_perception(self):
        ns = self.add_namespace("perception",
            "How agents and organisms sense, filter, and interpret their environment")

        ns.define("sensory-adaptation",
            "Decreased sensitivity to constant stimuli — your brain filters out the unchanging",
            description="You stop feeling your clothes. Background noise becomes inaudible. Constant smells become invisible. This is feature, not bug: it frees attention for changes and novelties. The fleet's cuda-attention habituation implements sensory adaptation: constant sensor readings get lower attention weight. Only changes trigger attention.",
            level=Level.BEHAVIOR,
            examples=["stopping noticing your watch after wearing it for a while", "not hearing the refrigerator hum", "agent: habituating to constant temperature, only noticing changes"],
            bridges=["habituation", "attention", "change-detection", "novelty"],
            tags=["perception", "adaptation", "attention"])

        ns.define("change-blindness",
            "Failure to notice significant changes in a scene when the change occurs during a disruption",
            description="In a famous experiment, a person giving directions doesn't notice when the person they're talking to is replaced by someone else. The fleet can experience change-blindness: if a critical environmental change happens during a high-priority deliberation, the agent may not update its world model. Interrupt-driven perception (hardware interrupts) mitigates change-blindness.",
            level=Level.BEHAVIOR,
            examples=["not noticing conversation partner was swapped", "not noticing a UI change during a page reload", "fleet: not updating world model during high-priority deliberation"],
            bridges=["attention", "inattentional-blindness", "interrupt", "perception"],
            tags=["perception", "blindness", "attention"])

        ns.define("inattentional-blindness",
            "Failure to notice unexpected objects when attention is focused on another task",
            description="The invisible gorilla experiment: viewers counting basketball passes don't notice a person in a gorilla suit walking through the scene. Attention is finite. Focus on X means missing Y. In the fleet: an agent focusing deliberation on path planning may completely miss a new obstacle appearing in its sensor data. cuda-attention's focus modes and cuda-perception's scene composition mitigate this.",
            level=Level.BEHAVIOR,
            examples=["invisible gorilla experiment", "noticing a phone ringing while reading", "fleet: missing new obstacle while planning path"],
            bridges=["attention", "change-blindness", "focus", "resource-limitation"],
            tags=["perception", "blindness", "attention", "limitation"])

        ns.define("multisensory-fusion",
            "Combining information from multiple sensor types to produce more accurate perception",
            description="You see a dog and hear barking. Both confirm 'dog nearby'. But you see a dog and hear a cat meow — conflict. Multisensory fusion resolves conflicts using reliability weighting (higher-confidence source gets more weight). The fleet's cuda-fusion implements Bayesian multisensory fusion. cuda-perception provides the signal filtering pipeline.",
            level=Level.PATTERN,
            examples=["seeing + hearing confirms object identity", "GPS + accelerometer = better position than either alone", "fleet: lidar + camera + radar fusion for obstacle detection"],
            bridges=["bayesian-fusion", "confidence", "sensor", "perception"],
            tags=["perception", "fusion", "multi-sensor"])

        ns.define("object-permanence",
            "Understanding that objects continue to exist even when not currently perceived",
            description="A baby drops a toy out of sight and thinks it's gone. An adult knows it's still there. Object permanence is a foundational cognitive ability. In the fleet's cuda-world-model: objects have a `permanence` property that decays over time but doesn't drop to zero instantly. An obstacle seen 10 cycles ago is believed to still exist (with reduced confidence).",
            level=Level.DOMAIN,
            examples=["baby doesn't understand object permanence", "adult knows objects persist when out of sight", "fleet: obstacle believed to still exist after sensor loses sight, with decaying confidence"],
            bridges=["memory", "world-model", "persistence", "spatial"],
            tags=["perception", "cognitive", "spatial", "fleet"])

    def _load_communication(self):
        ns = self.add_namespace("communication-theory",
            "Models and frameworks for understanding agent-to-agent communication")

        ns.define("shannon-weaver-model",
            "Sender -> Encoder -> Channel -> Decoder -> Receiver, with noise at each stage",
            description="The foundational model of communication. Sender encodes a message, sends through a noisy channel, receiver decodes. Noise corrupts. Redundancy and error correction compensate. The fleet's A2A protocol is Shannon-Weaver: sender encodes intent + payload, channel is the fleet mesh (noisy), receiver decodes. cuda-codec handles encoding/decoding. Confidence tracks noise level.",
            level=Level.DOMAIN,
            examples=["telephone: voice -> phone encoder -> network -> phone decoder -> ear", "fleet: intent+payload -> A2A encode -> mesh -> A2A decode -> receive", "radio: voice -> modulator -> electromagnetic waves -> demodulator -> speaker"],
            bridges=["information-theory", "encoding", "noise", "channel-capacity"],
            tags=["communication", "model", "foundational"])

        ns.define("information-bottleneck",
            "Compressing information to its most relevant parts, discarding irrelevant detail",
            description="You describe a movie plot in 30 seconds, not 3 hours. You compressed to the relevant parts. Tishby's information bottleneck theory: the best representation of X for predicting Y discards all information in X that's irrelevant to Y. The fleet's cuda-prompt compression and cuda-filtration both implement information bottleneck: keep what's relevant to the task, discard the rest.",
            level=Level.DOMAIN,
            examples=["movie summary in 30 seconds", "compressing sensor data to features relevant for navigation", "fleet: compressing deliberation history to elements relevant for current decision"],
            bridges=["compression", "relevance", "abstraction", "information-theory"],
            tags=["communication", "compression", "information"])

        ns.define("context-window",
            "The amount of recent information an agent can actively consider at once",
            description="Your working memory holds about 7 items. A language model has a context window of N tokens. Beyond the window, information is forgotten or must be re-read. The fleet's temporal memory (cuda-temporal) manages the context window: recent events are immediately accessible, older events require retrieval. cuda-memory-fabric implements the storage behind the window.",
            level=Level.CONCRETE,
            examples=["human working memory: ~7 items", "LLM context window: 128K tokens", "fleet: last 10 deliberation cycles in active context", "conversation: how much you remember of what was said earlier"],
            bridges=["working-memory", "attention", "resource-limitation", "retrieval"],
            tags=["communication", "memory", "limitation"])

        ns.define("code-switching",
            "Alternating between different languages or registers based on context and audience",
            description="Bilingual speakers switch between languages depending on who they're talking to. In the fleet: agents switch between communication registers depending on context — detailed technical messages to technical agents, simple status updates to coordination agents. cuda-communication's intent types implement a form of code-switching.",
            level=Level.BEHAVIOR,
            examples=["bilingual switching between languages based on audience", "engineer switching between technical and non-technical explanations", "fleet: different message formats for different agent roles"],
            bridges=["context", "audience", "pragmatics", "register"],
            tags=["communication", "adaptation", "context"])

    def _load_tradeoffs(self):
        ns = self.add_namespace("tradeoffs",
            "The fundamental tensions that cannot be resolved — only managed")

        ns.define("exploration-exploitation",
            "Trying new strategies (exploration) vs using known-good strategies (exploitation)",
            description="A restaurant you know is good vs trying a new one. Reading a favorite author vs discovering a new one. In the fleet: every cycle, the agent chooses to explore (new strategies, unknown paths) or exploit (proven strategies, known paths). Too much exploration wastes energy. Too much exploitation misses better options. cuda-adaptation manages this balance.",
            level=Level.DOMAIN,
            examples=["restaurant: known good vs new unknown", "agent: proven navigation strategy vs exploring new route", "science: building on established theory vs trying radical new approach"],
            bridges=["deliberation", "energy-budget", "learning", "risk"],
            tags=["tradeoff", "fundamental", "learning"])

        ns.define("speed-accuracy",
            "Faster responses are less accurate; more accurate responses take longer",
            description="Multiple choice: answer fast (less accurate) or think carefully (more accurate). The fleet faces this constantly: quick instinctive response (fast, low confidence) vs deliberation (slow, high confidence). Energy budget forces the trade: limited ATP means limited deliberation. The right balance depends on urgency and stakes.",
            level=Level.DOMAIN,
            examples=["quick guess vs careful analysis", "real-time obstacle avoidance (speed) vs path planning (accuracy)", "fleet: instinct response (fast, low conf) vs deliberation (slow, high conf)"],
            bridges=["energy-budget", "deliberation", "instinct", "deadline"],
            tags=["tradeoff", "fundamental", "performance"])

        ns.define("generality-specificity",
            "General solutions handle many cases but none optimally; specific solutions handle one case perfectly",
            description="A Swiss army knife does everything but nothing well. A chef's knife cuts perfectly but only cuts. The fleet needs both: general genes that work across many contexts (cuda-genepool baseline) and specific genes optimized for particular environments. cuda-playbook manages the specificity spectrum.",
            level=Level.DOMAIN,
            examples=["Swiss army knife vs chef knife", "general-purpose agent vs specialized agent", "fleet: general navigation gene vs warehouse-specific navigation gene"],
            bridges=["generalization", "specialization", "niche", "playbook"],
            tags=["tradeoff", "fundamental", "design"])

        ns.define("consistency-availability",
            "In distributed systems, you can have at most 2 of: Consistency, Availability, Partition tolerance",
            description="The CAP theorem. Network partition happens. Choose: be consistent (reject stale reads) or available (serve stale reads). Can't have both during a partition. The fleet chooses availability over consistency: agents operate with slightly stale data (eventual consistency via CRDTs) rather than waiting for synchronization.",
            level=Level.DOMAIN,
            examples=["distributed database during network partition", "fleet: agents operate with stale data rather than stopping", "mobile app: work offline (availability) with stale cache (inconsistency)"],
            bridges=["eventual-consistency", "cap-theorem", "partition", "distributed"],
            tags=["tradeoff", "distributed", "fundamental"])

        ns.define("simplicity-completeness",
            "Simple systems are easy to understand but may miss edge cases; complete systems handle everything but are complex",
            description="Occam's razor vs exhaustive handling. A simple confidence fusion formula (harmonic mean) is easy to understand but may not handle all cases. A complete fusion model with 15 parameters handles more cases but is harder to debug, maintain, and explain. The fleet starts simple and adds complexity only when needed.",
            level=Level.DOMAIN,
            examples=["simple: harmonic mean for confidence fusion", "complete: Bayesian network with 15 parameters", "fleet: start simple, add complexity as needed (YAGNI)"],
            bridges=["minimalism", "completeness", "yagni", "elegance"],
            tags=["tradeoff", "design", "fundamental"])

        ns.define("transparency-performance",
            "Explainable systems are slower; opaque systems are faster but untrustworthy",
            description="A decision tree you can read and understand. A neural network that's faster but nobody knows why it works. The fleet's deliberation (cuda-deliberation) is transparent: every proposal has a confidence score and rationale. Instinct responses are opaque: fast but hard to audit. Transparency is needed for accountability; performance is needed for speed.",
            level=Level.DOMAIN,
            examples=["decision tree (transparent, slower) vs neural network (opaque, faster)", "fleet: deliberation (transparent, slow) vs instinct (opaque, fast)", "white-box vs black-box model"],
            bridges=["explainability", "accountability", "audit", "performance"],
            tags=["tradeoff", "fundamental", "ai-ethics"])


    def _load_epistemology(self):
        ns = self.add_namespace("epistemology",
            "The theory of knowledge — what can we know, how do we know it, and what justifies belief")

        ns.define("justified-true-belief",
            "Classical definition of knowledge: you believe X, X is true, and you have justification for believing X",
            description="Plato's definition. You believe it's raining (belief), it IS raining (truth), and you looked out the window (justification). Gettier showed this isn't sufficient (justified true belief can be lucky coincidence). In the fleet: an agent 'knows' an obstacle exists when it believes it (perception), it's true (world state), and it has justification (sensor reading). Confidence is the justification metric.",
            level=Level.DOMAIN,
            examples=["I know it's raining: I believe it, it's true, I looked outside", "agent knows obstacle exists: perception + world state + sensor justification", "Gettier case: justified true belief that's actually lucky coincidence"],
            bridges=["confidence", "justification", "truth", "belief"],
            tags=["epistemology", "knowledge", "truth"])

        ns.define("gettier-problem",
            "Justified true belief can be based on false premises — luck masquerading as knowledge",
            description="You look at a stopped clock showing 2:00. It IS 2:00. You're justified (you looked at a clock). You believe it. It's true. But the clock is broken — you got lucky. This isn't knowledge. In the fleet: an agent perceives an obstacle (sensor reading), but the sensor was miscalibrated. The obstacle isn't there. The agent's 'knowledge' is Gettiered.",
            level=Level.DOMAIN,
            examples=["stopped clock at 2:00 when it's actually 2:00", "agent sensor reading shows obstacle but sensor is miscalibrated", "lucky guess that happens to be correct"],
            bridges=["justified-true-belief", "calibration", "sensor-reliability", "luck"],
            tags=["epistemology", "philosophy", "knowledge"])

        ns.define("epistemic-humility",
            "Acknowledging the limits of one's own knowledge — understanding what you don't know",
            description="Socrates: 'I know that I know nothing.' The wisest agent knows the boundaries of its knowledge. In the fleet: cuda-self-model's calibration includes uncertainty estimates. An agent that reports high confidence on everything (no humility) is miscalibrated. Reporting appropriate uncertainty IS epistemic humility.",
            level=Level.BEHAVIOR,
            examples=["scientist acknowledging limitations of their theory", "agent reporting 0.6 confidence instead of 0.95 when evidence is mixed", "saying 'I don't know' when you genuinely don't know"],
            bridges=["calibration", "self-model", "uncertainty", "metacognition"],
            tags=["epistemology", "humility", "uncertainty"])

        ns.define("reliabilism",
            "Knowledge is belief produced by a reliable process, regardless of conscious justification",
            description="Forget justification — if the process that produced the belief is reliable (tends to produce true beliefs), it's knowledge. Your vision is reliable: you generally see what's there. In the fleet: a sensor is a reliable process. If the sensor is well-calibrated, its outputs constitute knowledge even if the agent can't articulate WHY it trusts the sensor. Sensor health monitoring (cuda-sensor-agent) ensures reliability.",
            level=Level.DOMAIN,
            examples=["vision: generally reliable process for producing true beliefs about the visual world", "well-calibrated sensor: reliable process even without explicit justification", "agent: 'I know obstacle exists because my sensor (which is 99% reliable) says so'"],
            bridges=["justified-true-belief", "reliability", "sensor", "process"],
            tags=["epistemology", "philosophy", "reliability"])

        ns.define("foundationalism",
            "All knowledge rests on basic, self-evident beliefs that don't need further justification",
            description="A building needs a foundation. Knowledge needs basic beliefs. These can't be justified by other beliefs (circular) or justified by nothing (arbitrary). They must be self-evident. In the fleet: instincts are foundational beliefs. 'Survive' doesn't need justification. It's axiomatically true for any agent that wants to continue existing. Everything else (deliberation, learning) builds on this foundation.",
            level=Level.META,
            examples=["Descartes: 'I think therefore I am' as foundational", "mathematics: axioms as foundation for proofs", "fleet: survival instinct as foundational axiom, all other knowledge builds on it"],
            bridges=["instinct", "axiom", "foundation", "self-evident"],
            tags=["epistemology", "philosophy", "foundation", "meta"])

    def _load_biology(self):
        ns = self.add_namespace("biology",
            "Biological systems as engineering blueprints for agent architecture")

        ns.define("homeostasis",
            "Maintaining internal conditions within a narrow range despite external changes",
            description="Body temperature: 98.6F regardless of weather. Blood sugar: narrow range regardless of meals. Homeostasis is the biological version of the setpoint + feedback loop. The fleet implements homeostasis: energy budget (ATP), trust level, and confidence all have target ranges maintained by feedback loops.",
            level=Level.DOMAIN,
            examples=["body temperature regulation", "blood sugar regulation", "fleet: energy homeostasis via rest/act cycle", "fleet: trust homeostasis via positive/negative interactions"],
            bridges=["setpoint", "feedback-loop", "regulation", "stability"],
            tags=["biology", "regulation", "stability", "fleet"])

        ns.define("allometry",
            "How body parts scale relative to total size — different parts grow at different rates",
            description="An elephant's heart doesn't scale linearly with its body. A mouse's heart rate is 500+ BPM; an elephant's is 30 BPM. Different components scale at different rates (power laws). In the fleet: larger fleets don't need proportionally more captains. Coordination overhead scales superlinearly. Sensor density scales sublinearly with area. Allometry explains why fleet scaling isn't linear.",
            level=Level.DOMAIN,
            examples=["elephant heart rate vs mouse heart rate", "ant strength scales with cross-section, not mass", "fleet: coordination overhead scales superlinearly with fleet size"],
            bridges=["scaling", "power-law", "superlinear", "scaling"],
            tags=["biology", "scaling", "power-law"])

        ns.define("allostasis",
            "Adapting the setpoint itself in response to sustained environmental change — achieving stability through change",
            description="Homeostasis maintains the SAME setpoint. Allostasis CHANGES the setpoint to match new conditions. Chronic stress raises the 'normal' cortisol level. The fleet's cuda-adaptation implements allostasis: if the environment consistently demands faster responses, the agent raises its speed setpoint. Epigenetic memory (cuda-energy) makes allostasis persistent.",
            level=Level.DOMAIN,
            examples=["chronic stress raises baseline cortisol (allostasis vs homeostasis)", "acclimatization to altitude changes baseline physiology", "fleet: sustained fast environment raises agent's speed setpoint"],
            bridges=["homeostasis", "setpoint", "adaptation", "epigenetic"],
            tags=["biology", "adaptation", "setpoint", "fleet"])

        ns.define("immune-response",
            "Discriminating self from non-self, and neutralizing threats while preserving beneficial elements",
            description="The immune system must distinguish self from non-self (antigens). Then mount appropriate response: antibodies for known threats, inflammation for unknowns. Autoimmune diseases attack self. The fleet's membrane (cuda-genepool) IS an immune system: antibodies block dangerous signals while allowing safe ones through. The challenge: blocking threats without blocking beneficial cooperation.",
            level=Level.DOMAIN,
            examples=["antibodies targeting specific pathogens", "autoimmune disease: immune system attacks self", "fleet: membrane blocks dangerous commands while allowing safe cooperation"],
            bridges=["membrane", "self-other", "security", "antibody"],
            tags=["biology", "immunity", "security", "fleet"])

        ns.define("metabolism",
            "The total chemical processes that convert inputs into energy and building blocks",
            description="Eating -> digesting -> converting to ATP -> using ATP for work -> producing waste. Metabolism IS the energy pipeline. The fleet's mitochondrion (cuda-genepool) implements metabolism: rest converts inputs to ATP, actions consume ATP, waste (heat, stale data) is produced. The metabolic rate determines how fast the agent can operate.",
            level=Level.DOMAIN,
            examples=["cellular respiration: glucose + O2 -> ATP + CO2 + H2O", "fleet: rest generates ATP, actions consume ATP, waste (stale data) produced", "athlete metabolism: faster metabolic rate = more energy available"],
            bridges=["energy", "atp", "rest", "mitochondrion"],
            tags=["biology", "energy", "metabolism", "fleet"])

    def _load_philosophy_of_science(self):
        ns = self.add_namespace("philosophy-of-science",
            "How science works — paradigms, falsifiability, and the growth of knowledge")

        ns.define("paradigm-shift",
            "Fundamental change in the dominant framework of a scientific discipline",
            description="Kuhn: science doesn't progress smoothly. Long periods of 'normal science' (puzzle-solving within existing framework) are interrupted by crises that lead to paradigm shifts ( Copernicus, Einstein, plate tectonics). In the fleet: switching from reactive agents to deliberative agents is a paradigm shift. Switching from individual agents to fleet coordination is another. Paradigm shifts change what questions are even askable.",
            level=Level.META,
            examples=["geocentric to heliocentric model", "Newtonian to Einsteinian physics", "individual agents to fleet coordination paradigm", "rule-based AI to neural networks"],
            bridges=["paradigm", "normal-science", "crisis", "revolution"],
            tags=["philosophy", "science", "paradigm", "meta"])

        ns.define("falsifiability",
            "A theory is scientific only if it makes predictions that could be proven wrong",
            description="Popper: you can't prove a theory true, only prove it false. 'All swans are white' is scientific because finding a black swan would falsify it. 'God exists' is not scientific because nothing could falsify it. In the fleet: agent hypotheses must be falsifiable. 'This path is safe' is falsifiable (test it). 'This agent is trustworthy' is falsifiable (observe betrayal). Unfalsifiable claims waste deliberation resources.",
            level=Level.DOMAIN,
            examples=["'all swans are white' — falsifiable by finding black swan", "agent hypothesis 'path is safe' — falsifiable by testing", "unfalsifiable: 'everything happens for a reason'"],
            bridges=["hypothesis", "testing", "science", "knowledge"],
            tags=["philosophy", "science", "falsifiability", "testing"])

        ns.define("occam-razor",
            "Among competing explanations that fit the evidence equally, prefer the simplest",
            description="Entities should not be multiplied beyond necessity. If two theories explain the same data, the simpler one is preferred. Not because it's more likely to be true, but because it's more testable, more generalizable, and less prone to overfitting. In the fleet: when two strategies have similar performance, prefer the one with fewer parameters (lower complexity = better generalization).",
            level=Level.PATTERN,
            examples=["heliocentric model simpler than epicycle model (Occam's razor favored heliocentric)", "agent: simpler navigation strategy preferred if performance is equal", "linear model preferred over polynomial if both fit data equally well"],
            bridges=["simplicity-completeness", "overfitting", "generalization", "complexity"],
            tags=["philosophy", "simplicity", "science", "heuristic"])

        ns.define("instrumentalism",
            "Theories are tools for prediction, not descriptions of reality — don't ask 'is it true?', ask 'does it work?'",
            description="Models are instruments, not mirrors. The electron isn't 'really' a wave or a particle — it's whichever model makes better predictions for the task at hand. In the fleet: dopamine IS confidence not because consciousness exists, but because the mathematical structure is identical and the model works. Instrumentalism justifies the entire biological metaphor framework: it doesn't matter if agents 'really' feel emotions, only that the emotion model produces better behavior.",
            level=Level.META,
            examples=["electron: wave model for diffraction, particle model for collision", "fleet: biological metaphors used because they work, not because agents are alive", "map is not territory: model is tool, not truth"],
            bridges=["model", "pragmatism", "metaphor", "truth"],
            tags=["philosophy", "pragmatism", "model", "meta"])


    def _load_causation(self):
        ns = self.add_namespace("causation",
            "Cause and effect, counterfactuals, and the structure of causal reasoning")

        ns.define("counterfactual",
            "Reasoning about what WOULD have happened if conditions were different",
            description="Had I taken the other route, would I have arrived sooner? Counterfactuals don't describe reality — they describe alternate realities. But they're essential for learning: you must imagine what would have happened to evaluate whether your decision was good. The fleet's deliberation considers counterfactuals: 'If we had taken path B instead of A, confidence would have been 0.92 instead of 0.78.' This comparison drives learning.",
            level=Level.DOMAIN,
            examples=["'if I had left earlier, I would have missed traffic'", "agent: 'if I had explored more, I would have found the shorter path'", "RCT: what would have happened without treatment?"],
            bridges=["causality", "learning", "credit-assignment", "imagination"],
            tags=["causation", "reasoning", "learning"])

        ns.define("confounding",
            "A third variable that influences both cause and effect, creating a spurious correlation",
            description="Ice cream sales and drowning both increase in summer. Ice cream doesn't cause drowning — temperature (the confounder) causes both. In the fleet: agent performance and energy level may be confounded by task difficulty (harder tasks drain energy AND reduce performance). Without accounting for the confounder, you'd wrongly conclude low energy causes poor performance, when task difficulty causes both.",
            level=Level.BEHAVIOR,
            examples=["ice cream and drowning: confounded by temperature", "education and income: confounded by family background", "agent: energy and performance confounded by task difficulty"],
            bridges=["correlation-causation", "bias", "attribution", "statistics"],
            tags=["causation", "bias", "statistics"])

        ns.define("causal-graph",
            "A directed graph representing cause-effect relationships among variables",
            description="Nodes are variables. Edges are causal relationships (arrow from cause to effect. A -> B means A causes B. Confounders have arrows pointing to both A and B. The fleet's causal graph (cuda-causal-graph) represents fleet knowledge about how system components relate causally. Diagnosis works by tracing backward from observed effects to their causes.",
            level=Level.PATTERN,
            examples=["A -> B -> C: causal chain", "D -> A and D -> B: D confounds A and B", "fleet: sensor-failure -> bad-perception -> wrong-decision causal chain"],
            bridges=["causality", "graph", "diagnosis", "attribution"],
            tags=["causation", "graph", "reasoning"])

        ns.define("mediation",
            "A variable that explains the mechanism through which a cause produces its effect",
            description="Exercise causes weight loss. But HOW? Through calorie burn (the mediator). Exercise -> calorie burn -> weight loss. Removing the mediator breaks the causal chain. In the fleet: deliberation causes better decisions. But through what mechanism? Through confidence filtering (the mediator): deliberation -> higher confidence -> better decision selection. Understanding mediators enables targeted intervention.",
            level=Level.DOMAIN,
            examples=["exercise -> calorie burn -> weight loss (calorie burn is mediator)", "deliberation -> confidence filter -> better decision (confidence is mediator)", "advertising -> brand awareness -> purchase (awareness is mediator)"],
            bridges=["causality", "mechanism", "intervention", "causal-graph"],
            tags=["causation", "mechanism", "reasoning"])

        ns.define("correlation-causation",
            "Correlated variables are not necessarily causally related — the most common statistical fallacy",
            description="Shoe size and reading ability are correlated in children. Bigger shoes don't make you read better — age causes both. In the fleet: two agents' performance may correlate not because they influence each other, but because they face the same environment. cuda-causal-graph distinguishes correlation from causation to prevent incorrect attribution.",
            level=Level.BEHAVIOR,
            examples=["shoe size and reading ability: confounded by age", "ice cream and crime: confounded by temperature", "agent A and B performance correlate due to shared environment, not causation"],
            bridges=["confounding", "spurious-correlation", "attribution", "statistics"],
            tags=["causation", "fallacy", "statistics"])

    def _load_abstraction(self):
        ns = self.add_namespace("abstraction",
            "Layers of representation, information hiding, and the art of managing complexity")

        ns.define("leaky-abstraction",
            "An abstraction that fails to completely hide its underlying details, forcing awareness of the layer below",
            description="TCP promises reliable delivery. But sometimes the network IS slow, and your abstraction can't hide it. SQL promises you don't care about disk layout, but a slow query forces you to think about indexes. Leaky abstractions aren't failures — they're inevitable when performance or correctness demands peeking below. The fleet's abstraction layers (genes -> proteins -> bytecode -> actions) are intentionally leaky: when energy is low, the protein layer leaks through and the agent becomes aware of its bytecode-level constraints.",
            level=Level.PATTERN,
            examples=["TCP promises reliability but can't hide network latency", "SQL promises disk independence but can't hide slow queries", "fleet: energy-aware agents must understand bytecode-level costs"],
            bridges=["abstraction-layer", "encapsulation", "performance", "transparency"],
            tags=["abstraction", "design", "leakiness"])

        ns.define("law-of-leaky-abstractions",
            "All non-trivial abstractions are to some degree leaky — Joel Spolsky",
            description="You can't build a perfect abstraction layer. Eventually, something goes wrong, and you need to understand what's underneath. This is why engineers need to understand multiple layers, and why the fleet's HAV exists: so agents can communicate across abstraction layers when leaks occur.",
            level=Level.META,
            examples=["every ORM eventually forces you to write raw SQL", "every network abstraction eventually exposes packet loss", "every agent abstraction eventually exposes resource constraints"],
            bridges=["leaky-abstraction", "abstraction-layer", "complexity", "engineering"],
            tags=["abstraction", "law", "meta"])

        ns.define("isomorphism",
            "Two structures that are identical in form — a mapping that preserves all relationships",
            description="The clock arithmetic (mod 12) is isomorphic to rotations of a clock face. Addition mod 12 maps perfectly to rotation by hours. In the fleet: the biological gene pool and the software playbook may be isomorphic — same structure, different representation. Isomorphisms let you translate understanding from one domain to another. If you understand A* pathfinding, you understand Dijkstra (they're isomorphic in their core structure).",
            level=Level.DOMAIN,
            examples=["mod 12 arithmetic isomorphic to clock rotations", "chess and Go are NOT isomorphic (different structure)", "biological instinct hierarchy isomorphic to software strategy hierarchy"],
            bridges=["analogy", "mapping", "translation", "structure"],
            tags=["abstraction", "mathematics", "structure"])

        ns.define("indirection",
            "Adding an extra layer between the thing and its use — power through indirection",
            description="Pointers point to memory. Variables point to values. URLs point to resources. DNS translates names to addresses. Each layer of indirection adds flexibility: change the target without changing the users. The fleet's vessel.json is indirection: other agents reference an agent by name (indirection) not by address (direct). If the agent moves, only the vessel.json changes.",
            level=Level.PATTERN,
            examples=["variable names indirection to memory addresses", "DNS indirection from name to IP", "vessel.json indirection from agent name to agent address", "function call indirection to code location"],
            bridges=["abstraction-layer", "reference", "flexibility", "decoupling"],
            tags=["abstraction", "pattern", "flexibility"])

        ns.define("black-box",
            "A system whose internal workings are opaque — you only see inputs and outputs",
            description="You press a button and toast comes out. You don't know (or care) how the toaster works. Black-box models are powerful because they're simple to use, but dangerous because you can't diagnose failures. The fleet treats some agents as black boxes: you send a request, you get a response. If the response is wrong, you can't fix the agent — only replace it. White-box agents (with provenance) allow diagnosis.",
            level=Level.PATTERN,
            examples=["toaster: press button, get toast", "neural network: input data, output prediction (black box)", "fleet: some agents as black boxes (replace, don't repair)"],
            bridges=["transparency", "white-box", "encapsulation", "opacity"],
            antonyms=["white-box"],
            tags=["abstraction", "opacity", "model"])

        ns.define("invariant",
            "A property that remains unchanged under transformation — the anchor in a sea of change",
            description="The total energy in a closed system is invariant. The number of nodes in a graph is invariant under relabeling. In the fleet: the total ATP in the fleet is invariant (conserved). The trust sum across all pairs is approximately invariant (trust gained by one is lost by another). Invariants are powerful because they provide constraints: if an invariant is violated, something is wrong.",
            level=Level.DOMAIN,
            examples=["conservation of energy (invariant under transformation)", "number of elements in a set (invariant under permutation)", "fleet: total ATP is conserved (energy generated = energy consumed)"],
            bridges=["conservation-law", "constraint", "verification", "property"],
            tags=["abstraction", "mathematics", "invariant", "verification"])

    def _load_dynamics(self):
        ns = self.add_namespace("dynamics",
            "How systems change over time — attractors, bifurcations, and chaos")

        ns.define("attractor",
            "A state or set of states that a dynamical system tends toward over time",
            description="A ball rolling in a bowl settles at the bottom — the bottom is an attractor. A pendulum settles at rest. The fleet's trust dynamics have attractors: mutual cooperation (positive attractor) or mutual defection (negative attractor). Once in an attractor, the system stays there unless perturbed. cuda-trust's decay function determines how quickly trust returns to its attractor.",
            level=Level.DOMAIN,
            examples=["ball in bowl: bottom is attractor", "pendulum: rest position is attractor", "fleet: cooperation attractor (positive) and defection attractor (negative)", "habit: behavior becomes attractor"],
            bridges=["equilibrium", "stability", "phase-space", "basin-of-attraction"],
            tags=["dynamics", "attractor", "stability"])

        ns.define("basin-of-attraction",
            "The set of all initial states that will converge to a given attractor",
            description="A ball in a large bowl has a large basin of attraction: you can drop it from many starting positions and it'll end up at the bottom. Two nearby basins: the ridge between them is the boundary. Cross it, and you fall into a different attractor. In the fleet: small changes in trust or confidence can push an agent from the cooperation basin into the defection basin. Understanding basins helps prevent unwanted state transitions.",
            level=Level.DOMAIN,
            examples=["large bowl = large basin of attraction", "small perturbation near basin boundary causes state switch", "fleet: small trust reduction near basin boundary switches from cooperation to defection"],
            bridges=["attractor", "bifurcation", "tipping-point", "phase-transition"],
            tags=["dynamics", "basin", "state-space"])

        ns.define("bifurcation",
            "A small change in a parameter causes a qualitative change in system behavior — the fork in the road",
            description="Water below 0C is ice. Above 0C is liquid. Temperature is the parameter; solid-liquid is the bifurcation. In the fleet: trust threshold is a bifurcation parameter. Below threshold, agents don't cooperate. Above threshold, they do. A tiny change in trust near the bifurcation point causes a huge change in fleet behavior. This is why threshold tuning matters so much.",
            level=Level.DOMAIN,
            examples=["water: ice to liquid at 0C (temperature is bifurcation parameter)", "population: below critical size = extinction, above = survival", "fleet: trust threshold bifurcation: below = no cooperation, above = cooperation"],
            bridges=["phase-transition", "tipping-point", "attractor", "parameter"],
            tags=["dynamics", "bifurcation", "criticality"])

        ns.define("strange-attractor",
            "A chaotic attractor — the system is bounded but never repeats exactly",
            description="Lorenz attractor: the system orbits within a bounded region but never visits the same point twice. Chaotic but not random. Weather is a strange attractor: predictable patterns (winter is cold) but unpredictable details (exact temperature Tuesday). The fleet's long-term behavior may be a strange attractor: bounded patterns that emerge but never repeat exactly.",
            level=Level.META,
            examples=["Lorenz attractor (butterfly shape)", "weather: bounded patterns, never exactly repeating", "stock market: bounded by physical constraints, unpredictable in detail"],
            bridges=["chaos", "attractor", "boundedness", "unpredictability"],
            tags=["dynamics", "chaos", "attractor", "meta"])

        ns.define("hysteresis-loop",
            "The path of a system through state space depends on direction — going up and coming down trace different paths",
            description="Magnetizing iron: increasing magnetic field magnetizes it. But decreasing the field doesn't fully demagnetize it — there's residual magnetism. The up-path and down-path form a loop (hysteresis). In the fleet: trust builds slowly but breaks quickly. The trust-building path and trust-destruction path are different — this IS hysteresis. Reputation damage outlasts the conditions that caused it.",
            level=Level.DOMAIN,
            examples=["magnetization hysteresis loop", "trust builds slowly, breaks quickly (different paths)", "thermostat with dead zone (hysteresis in temperature)", "reputation: damage persists after cause is removed"],
            bridges=["hysteresis", "path-dependence", "asymmetry", "memory"],
            tags=["dynamics", "hysteresis", "path-dependence"])

    def _load_collective_intelligence(self):
        ns = self.add_namespace("collective-intelligence",
            "Groups outperforming individuals — and when they don't")

        ns.define("wisdom-of-crowds",
            "The aggregate judgment of many independent individuals is often more accurate than any single expert",
            description="Guess the number of jellybeans in a jar. The average of all guesses is often closer than any individual guess. Key conditions: independence (people don't influence each other), diversity (different perspectives), decentralization (local knowledge), and aggregation. The fleet's consensus mechanism (cuda-consensus) implements wisdom of crowds: independent agents vote, aggregate confidence determines outcome.",
            level=Level.DOMAIN,
            examples=["jellybean jar: average guess often more accurate than any individual", "fleet: weighted consensus of independent agents outperforms single agent", "prediction markets aggregate diverse opinions accurately"],
            bridges=["consensus", "diversity", "independence", "aggregation"],
            tags=["collective", "wisdom", "aggregation"])

        ns.define("diversity-prediction-theorem",
            "Collective error = average individual error minus collective diversity — diversity makes groups smarter",
            description="Scott Page's theorem: a diverse group's average prediction beats the average individual prediction by the amount of diversity in the group. The formula: crowd error = avg individual error - diversity. More diversity = less collective error. This mathematically proves why the fleet needs diverse agents (different sensors, strategies, perspectives), not just many copies of the same agent.",
            level=Level.META,
            examples=["diverse team makes better predictions than similar team", "fleet: diverse agents (different strategies) outperform many copies of one strategy", "ensemble ML models: diverse models outperform single model"],
            bridges=["diversity", "wisdom-of-crowds", "ensemble", "collective"],
            tags=["collective", "diversity", "theorem", "meta"])

        ns.define("groupthink",
            "Desire for harmony overrides realistic appraisal of alternatives — the group agrees to agree",
            description="The team converges on a decision too quickly because dissent is socially costly. Nobody wants to rock the boat. The decision may be wrong but everyone supports it. In the fleet: if all agents have the same training and same incentives, they converge too quickly — groupthink. cuda-deliberation's Forfeit option allows dissent without penalty. Diversity in agent strategies prevents groupthink.",
            level=Level.BEHAVIOR,
            examples=["Bay of Pigs planning: nobody questioned the plan", "jury rushing to verdict to go home", "fleet: homogeneous agents all agree on wrong strategy (no dissent)"],
            bridges=["diversity", "conformity", "consensus", "bias"],
            tags=["collective", "bias", "social"])

        ns.define("plurality-illusion",
            "A minority opinion seems like the majority because its holders are more vocal",
            description="10% of people hold opinion X, but 80% of social media posts are about X (the vocal minority). The silent majority's opinion Y is invisible. In the fleet: one vocal agent can sway deliberation by flooding the communication channel. cuda-rate-limit and cuda-communication's energy costs prevent vocal minorities from dominating: every message costs energy, forcing agents to be selective.",
            level=Level.BEHAVIOR,
            examples=["vocal minority on social media", "one loud meeting participant dominating discussion", "fleet: one agent flooding messages sways deliberation without rate limiting"],
            bridges=["rate-limit", "communication-cost", "sampling-bias", "social"],
            tags=["collective", "bias", "social"])

        ns.define("social-loafing",
            "Individuals exert less effort in a group than when working alone — the free-rider problem",
            description="Rope pulling experiment: people pull harder alone than in a group. When your contribution is anonymous and the group's output is shared, there's incentive to coast. In the fleet: if energy is shared, agents may free-ride — let others spend ATP on deliberation while conserving their own. Individual energy budgets (cuda-energy) prevent social loafing: each agent pays its own costs.",
            level=Level.BEHAVIOR,
            examples=["ringelman effect: less pull per person in larger group", "group project: some members coast", "fleet: agents free-ride when energy is shared pool"],
            bridges=["tragedy-of-commons", "free-rider", "energy-budget", "incentive"],
            tags=["collective", "bias", "motivation"])

    def _load_risk(self):
        ns = self.add_namespace("risk",
            "Uncertainty with consequences — how to think about what could go wrong")

        ns.define("tail-risk",
            "Probability of extreme events that are far more likely than normal distributions predict",
            description="Normal distribution says a 5-sigma event is 1 in 3.5 million. In reality, financial markets see 5-sigma events every few years. Real-world distributions have fat tails. In the fleet: the probability of multiple simultaneous agent failures is far higher than independent probability would suggest (they share environment, not independent). cuda-resilience's bulkheads protect against tail-risk events.",
            level=Level.DOMAIN,
            examples=["2008 financial crisis: far more likely than normal distribution predicted", "fleet: multiple agent failures more likely than independence suggests", "COVID pandemic: fat-tail event"],
            bridges=["fat-tail", "black-swan", "normal-distribution", "resilience"],
            tags=["risk", "tail", "statistics"])

        ns.define("black-swan",
            "An event that is (1) extremely rare, (2) has massive impact, and (3) is explained in hindsight as predictable",
            description="Taleb: black swans don't exist (Europe thought) until Australia was discovered. 9/11 was a black swan. COVID was arguably not (pandemics were predicted). The fleet must distinguish between risks it can prepare for (known unknowns) and risks it fundamentally cannot anticipate (unknown unknowns). Anti-fragility is the only defense against black swans: not predicting them, but being structured to benefit from disruption.",
            level=Level.META,
            examples=["9/11, 2008 financial crisis", "internet invention (positive black swan)", "fleet: entirely new class of attack that no defense was designed for"],
            bridges=["anti-fragility", "tail-risk", "unknown-unknown", "robustness"],
            tags=["risk", "black-swan", "meta", "unpredictability"])

        ns.define("precautionary-principle",
            "When an action has potential for severe harm, prove it's safe before proceeding — burden of proof on the actor",
            description="Better safe than sorry. If a new chemical might cause cancer, don't use it until proven safe. Not: use it until proven harmful. The fleet applies this to gene pool modifications: new genes are tested in isolation before fleet-wide deployment. Membrane security (cuda-genepool) acts as precautionary barrier.",
            level=Level.DOMAIN,
            examples=["new drug: prove safe before market", "new fleet gene: test in isolation before sharing", "GMO: prove safe before cultivation", "AI capability: prove safe before deployment"],
            bridges=["safety-first", "burden-of-proof", "membrane", "compliance"],
            tags=["risk", "principle", "safety", "policy"])

        ns.define("asymmetric-risk",
            "Where downside is much larger than upside (or vice versa) — the payoff is lopsided",
            description="A doctor treating a disease: wrong diagnosis = patient dies (huge downside), right diagnosis = patient lives (moderate upside). The risk is asymmetric. In the fleet: an agent exploring unknown territory: worst case = energy wasted (small downside), best case = discovering optimal path (huge upside). This asymmetric risk makes exploration rational even when success probability is low. Optionality IS asymmetric risk.",
            level=Level.DOMAIN,
            examples=["doctor misdiagnosis: asymmetric downside", "startup: small investment, huge potential upside (asymmetric)", "agent exploration: small energy cost, potentially huge discovery"],
            bridges=["optionality", "convexity", "risk-reward", "exploration"],
            tags=["risk", "asymmetry", "payoff"])

        ns.define("defence-in-depth",
            "Multiple layers of security so that if one fails, others still protect",
            description="Castle: moat, wall, gate, guards, inner keep. Each layer independently provides some protection. Not a single perfect wall (which, if breached, gives full access), but multiple imperfect layers. The fleet's security is defense-in-depth: membrane (cuda-genepool), RBAC (cuda-rbac), sandbox (cuda-sandbox), compliance (cuda-compliance), circuit breaker (cuda-circuit). Each layer is imperfect; together they're robust.",
            level=Level.PATTERN,
            examples=["castle: moat + wall + gate + guards + keep", "computer: firewall + antivirus + encryption + backup", "fleet: membrane + RBAC + sandbox + compliance + circuit breaker"],
            bridges=["resilience", "layered-security", "redundancy", "defense"],
            tags=["risk", "security", "layers", "pattern"])

    def _load_autonomy(self):
        ns = self.add_namespace("autonomy",
            "Degrees of self-governance, agency, and independence")

        ns.define("agency",
            "The capacity to act independently and make choices that affect the world",
            description="An agent has agency when its actions originate from its own deliberation, not from external commands. A thermostat has no agency (fixed response to stimulus). A human choosing what to eat has high agency. The fleet's agents have graduated agency: low-energy agents follow instincts (low agency), high-energy agents deliberate and choose (high agency). Agency is a function of available energy and cognitive capacity.",
            level=Level.DOMAIN,
            examples=["human choosing career: high agency", "thermostat: no agency (fixed response)", "fleet agent: agency proportional to available ATP and deliberation depth"],
            bridges=["autonomy", "deliberation", "energy-budget", "choice"],
            tags=["autonomy", "agency", "choice"])

        ns.define("supervenience",
            "Higher-level properties depend on lower-level properties, but the same higher-level property can arise from different lower-level states",
            description="Consciousness supervenes on neural activity: change the neurons, consciousness changes. But different neural configurations can produce the same conscious experience. In the fleet: 'navigation skill' supervenes on gene pool composition: different gene combinations can produce equally good navigation. This means you can't reduce high-level properties to simple low-level rules — multiple realizations exist.",
            level=Level.META,
            examples=["consciousness supervenes on neural activity", "temperature supervenes on molecular kinetic energy", "fleet: navigation skill supervenes on genes (multiple gene combinations, same skill)"],
            bridges=["reductionism", "emergence", "multiple-realizability", "levels"],
            tags=["autonomy", "philosophy", "meta", "emergence"])

        ns.define("subsidiarity",
            "Decisions should be made at the lowest competent level — push authority downward",
            description="The EU principle: what can be decided locally should not be decided centrally. In the fleet: agents should make decisions locally whenever possible. Only escalate to fleet-level when local information is insufficient. This minimizes communication overhead and maximizes responsiveness. cuda-hierarchy implements subsidiarity: low-level agents handle routine decisions, high-level agents handle exceptional ones.",
            level=Level.PATTERN,
            examples=["EU subsidiarity: local decisions for local issues", "military: soldiers make battlefield decisions, generals set strategy", "fleet: agents decide locally, escalate only when necessary"],
            bridges=["hierarchy", "decentralization", "authority", "delegation"],
            tags=["autonomy", "governance", "principle", "fleet"])

        ns.define("command-and-control",
            "Centralized decision-making where all information flows up and all orders flow down",
            description="Military model: soldiers observe, report to commander, commander decides, orders flow back down. Advantages: coherent global strategy. Disadvantages: slow, fragile (single point of failure), ignores local knowledge. The fleet's opposite: agents decide locally, share information laterally, only escalate when necessary. Command-and-control is efficient for simple problems, swarm is efficient for complex ones.",
            level=Level.PATTERN,
            examples=["military hierarchy", "factory floor management", "fleet: contrast with swarm/coordination model"],
            bridges=["subsidiarity", "decentralization", "hierarchy", "swarm"],
            antonyms=["subsidiarity", "swarm"],
            tags=["autonomy", "governance", "centralized"])

        ns.define("delegation",
            "Transferring authority for a task from one agent to another while maintaining accountability",
            description="A manager delegates a project to an employee. The employee has authority to make decisions but the manager is accountable for the outcome. In the fleet: the captain agent (cuda-captain) delegates tasks to worker agents via enlistment. Workers have authority to complete the task their way. Captain remains accountable for mission success. Delegation requires trust: you delegate only to agents with sufficient trust scores.",
            level=Level.PATTERN,
            examples=["manager delegates project to employee", "captain delegates navigation to scout agent", "parent delegates chore to child"],
            bridges=["trust", "authority", "accountability", "captain"],
            tags=["autonomy", "delegation", "governance"])

    def _load_simulation(self):
        ns = self.add_namespace("simulation",
            "Modeling complex systems to understand, predict, and test before acting")

        ns.define("digital-twin",
            "A virtual replica of a physical system used for testing, prediction, and optimization",
            description="A digital twin of a jet engine: sensors stream real data to a virtual model. Engineers simulate scenarios (what if turbine blade cracks?) without risking the real engine. In the fleet: cuda-world-model is a digital twin of the agent's environment. Agents simulate actions in the world model before executing them physically. The twin is always slightly behind reality (latency), but close enough for useful prediction.",
            level=Level.DOMAIN,
            examples=["jet engine digital twin for predictive maintenance", "fleet: world model as digital twin of environment", "smart building twin for energy optimization"],
            bridges=["world-model", "prediction", "simulation", "model"],
            tags=["simulation", "digital-twin", "model", "fleet"])

        ns.define("monte-carlo",
            "Estimating unknown quantities by random sampling — when analytical solutions are impossible",
            description="You can't calculate the probability of a complex chess position analytically. But you can simulate 10,000 random games from that position and count how many White wins. That ratio IS the Monte Carlo estimate. In the fleet: when deliberation can't analytically compare strategies, simulate random perturbations and count successes. cuda-world-model supports Monte Carlo evaluation of candidate actions.",
            level=Level.PATTERN,
            examples=["estimating pi by throwing darts at a circle", "chess: simulate random games to estimate position value", "fleet: simulate random perturbations of strategy to estimate fitness"],
            bridges=["random-sampling", "estimation", "simulation", "approximation"],
            tags=["simulation", "method", "estimation"])

        ns.define("sensitivity-analysis",
            "Varying input parameters to determine which ones most affect the output",
            description="A model has 20 parameters. Which 3 matter most? Vary each one independently and measure output change. The parameters with the largest effect are the sensitive ones — the leverage points. In the fleet: sensitivity analysis on the trust decay rate reveals whether small changes cause large behavior shifts (high sensitivity = important parameter, tune carefully).",
            level=Level.PATTERN,
            examples=["financial model: which assumptions most affect profit forecast?", "fleet: which parameters most affect agent behavior?", "climate model: which variables most affect temperature prediction?"],
            bridges=["leverage-point", "parameter", "model", "analysis"],
            tags=["simulation", "analysis", "parameter"])

        ns.define("ensemble",
            "Combining multiple models to produce better predictions than any single model alone",
            description="One decision tree: okay. Random forest (100 decision trees): much better. Diversity in models reduces variance and bias. In the fleet: deliberation IS an ensemble: multiple agents (multiple models) evaluate the same situation, their judgments are combined (weighted average by confidence), and the ensemble prediction outperforms any single agent.",
            level=Level.PATTERN,
            examples=["random forest > single decision tree", "weather forecast: ensemble of models > single model", "fleet deliberation: multiple agents > single agent"],
            bridges=["wisdom-of-crowds", "consensus", "diversity", "aggregation"],
            tags=["simulation", "ensemble", "ml", "collective"])

        ns.define("model-fidelity",
            "How accurately a model represents the real system it simulates",
            description="A weather model that predicts rain 90% of the time has high fidelity for rain prediction. A model that predicts temperature within 1 degree has high fidelity for temperature. No model has perfect fidelity (that would be the system itself). The fleet's world model fidelity depends on sensor accuracy: better sensors = higher fidelity = better predictions = better decisions.",
            level=Level.CONCRETE,
            examples=["weather model: fidelity varies by region and timescale", "flight simulator: visual fidelity vs physics fidelity", "fleet world model: fidelity = sensor accuracy"],
            bridges=["digital-twin", "accuracy", "model", "sensor"],
            tags=["simulation", "fidelity", "model", "accuracy"])

    def _load_privacy(self):
        ns = self.add_namespace("privacy",
            "Controlling access to information about agents and their interactions")

        ns.define("differential-privacy",
            "Adding calibrated noise to data so that individual records can't be inferred, while aggregate statistics remain accurate",
            description="A hospital wants to share average patient age without revealing any individual's age. Add random noise to each record. The noise is large enough to prevent inference of any individual, but small enough that the average is still useful. In the fleet: agent telemetry can be shared with differential privacy — fleet learns aggregate patterns without exposing any individual agent's behavior.",
            level=Level.DOMAIN,
            examples=["census data with differential privacy: accurate aggregates, individual privacy", "fleet telemetry: fleet-level patterns without individual agent exposure", "apple's differential privacy for emoji usage statistics"],
            bridges=["privacy", "noise", "aggregation", "statistics"],
            tags=["privacy", "mathematics", "data"])

        ns.define("data-minimization",
            "Collect only the minimum data necessary for the task — don't hoard 'just in case'",
            description="An app asks for your location, contacts, camera, and microphone to show you a restaurant menu. That's excessive. Data minimization says: collect what you need, nothing more. In the fleet: agents should share only the information necessary for coordination. cuda-filtration implements data minimization: strip unnecessary detail from messages before sending. Less data = less exposure = less attack surface.",
            level=Level.PATTERN,
            examples=["app requesting only location for restaurant search (minimal) vs contacts+camera (excessive)", "fleet: sharing only relevant sensor data, not full telemetry", "form asking only required fields"],
            bridges=["least-privilege", "filtration", "exposure-reduction", "principle"],
            tags=["privacy", "principle", "data"])

        ns.define("zero-knowledge",
            "Proving you know something without revealing what you know — the proof reveals nothing except the truth of the statement",
            description="You prove you're over 18 without revealing your exact age. You prove you have a valid ticket without revealing your seat number. Zero-knowledge proofs are the gold standard of privacy: they enable trust without disclosure. In the fleet: an agent can prove it has sufficient confidence for a task without revealing its full deliberation process or internal state.",
            level=Level.DOMAIN,
            examples=["proving age > 18 without revealing exact age", "proving password knowledge without sending password", "fleet: proving sufficient confidence without revealing internal state"],
            bridges=["cryptography", "privacy", "proof", "trust"],
            tags=["privacy", "cryptography", "proof"])

        ns.define("right-to-be-forgotten",
            "An agent's historical data can be deleted upon request, preventing indefinite retention",
            description="GDPR right: you can request a company delete all your data. In the fleet: agents should be able to reset their history. cuda-memory-fabric implements forgetting curves (data naturally decays), but the right to be forgotten goes further: explicit deletion of specific memories on request. This prevents reputation from being permanently determined by past behavior.",
            level=Level.CONCRETE,
            examples=["GDPR data deletion request", "fleet agent requesting memory deletion after bad experience", "criminal record expungement"],
            bridges=["memory", "decay", "gdpr", "deletion"],
            tags=["privacy", "rights", "data", "memory"])

    def _load_organization(self):
        ns = self.add_namespace("organization",
            "How groups structure themselves — from teams to societies")

        ns.define("conways-law",
            "Organizations design systems that mirror their communication structure",
            description="'If you have four groups building a compiler, you'll get a four-pass compiler.' The architecture reflects the org chart. In the fleet: if agents are organized by sensor type (vision group, audio group), the fleet architecture will have vision and audio modules. If agents are organized by task (navigation, communication), the architecture will have task-based modules. Fleet structure follows communication structure.",
            level=Level.PATTERN,
            examples=["4 teams = 4-pass compiler", "siloed departments = siloed software modules", "fleet: agent organization determines fleet architecture"],
            bridges=["architecture", "communication-structure", "organization", "design"],
            tags=["organization", "architecture", "law"])

        ns.define("dunbar-number",
            "Cognitive limit of ~150 stable social relationships — the size of a tribe",
            description="You can maintain relationships with about 150 people. Beyond that, you need hierarchies, rules, and abstractions to maintain social cohesion. In the fleet: an individual agent can maintain stable relationships with ~150 other agents. Beyond this, the fleet needs sub-structuring: teams, groups, hierarchies. A flat fleet of 500 agents will have weaker coordination than a hierarchical fleet of 500.",
            level=Level.DOMAIN,
            examples=["village size ~150 people", "military company ~150 soldiers", "fleet: agent can maintain ~150 stable relationships"],
            bridges=["social-limit", "hierarchy", "team-size", "cognitive-limit"],
            tags=["organization", "social", "limit"])

        ns.define("requisite-variety",
            "A system must have at least as much variety (complexity) as its environment to survive",
            description="Ashby's Law: a thermostat with two states (on/off) can't control room temperature to 0.1 degree precision — it lacks requisite variety. To control complex environments, controllers need equivalent complexity. In the fleet: to handle diverse environments, agents need diverse strategies. A single navigation strategy (low variety) fails in diverse environments. The gene pool (cuda-genepool) provides requisite variety through genetic diversity.",
            level=Level.META,
            examples=["thermostat can't control to 0.1C precision (insufficient variety)", "fleet needs diverse strategies for diverse environments (requisite variety)", "immune system: diverse antibodies for diverse pathogens"],
            bridges=["diversity", "complexity", "environment", "adaptation"],
            tags=["organization", "cybernetics", "law", "meta"])

        ns.define("two-pizza-rule",
            "A team should be small enough to be fed with two pizzas — keep teams small and autonomous",
            description="Amazon's rule of thumb: if a team needs more than two pizzas, it's too big. Small teams are faster, more cohesive, and have less communication overhead. In the fleet: agent teams should be small (3-7 agents). Beyond this, split into sub-teams with clear interfaces. Large flat fleets suffer from O(n^2) communication overhead.",
            level=Level.CONCRETE,
            examples=["Amazon two-pizza team rule", "military squad: ~8 soldiers", "fleet: agent task team of 3-7 agents"],
            bridges=["team-size", "communication-overhead", "dunbar-number", "subdivision"],
            tags=["organization", "team-size", "rule"])

        ns.define("inverse-responsibility",
            "As organizations grow, individuals feel less personally responsible for outcomes",
            description="In a 3-person startup, everyone feels responsible for everything. In a 10,000-person company, nobody feels responsible for anything. Responsibility diffuses. In the fleet: as fleet size grows, individual agents feel less responsible for fleet outcomes. Provenance tracking (cuda-provenance) counteracts this by making every agent's contribution traceable and attributable.",
            level=Level.BEHAVIOR,
            examples=["startup: everyone responsible", "large company: nobody feels responsible", "fleet: individual responsibility diffuses as fleet grows"],
            bridges=["accountability", "provenance", "attribution", "diffusion"],
            tags=["organization", "responsibility", "social"])

    def _load_strategy(self):
        ns = self.add_namespace("strategy",
            "Competitive and cooperative strategy in dynamic environments")

        ns.define("red-queen-hypothesis",
            "It takes all the running you can do to keep in the same place -- constant evolution just to maintain position",
            description="Alice in Wonderland: the Red Queen says 'it takes all the running you can do to keep in the same place.' In biology: predators and prey co-evolve, both getting faster, neither gaining advantage. In the fleet: attacker agents and defender agents co-evolve. Both improve, neither wins permanently. The fleet must continuously invest in adaptation just to maintain current capability.",
            level=Level.DOMAIN,
            examples=["predator-prey co-evolution arms race", "security: attackers and defenders both improving constantly", "fleet: adversarial agents co-evolving, both improving"],
            bridges=["co-evolution", "arms-race", "adaptation", "competition"],
            tags=["strategy", "competition", "evolution"])

        ns.define("blue-ocean",
            "Creating uncontested market space rather than competing in existing bloody markets",
            description="Red ocean: cutthroat competition, shrinking profits. Blue ocean: create new demand, no competitors (initially). Cirque du Soleil didn't compete with traditional circuses — they created a new category. In the fleet: cuda-genepool's gene exploration seeks blue oceans — behavioral spaces no other agent occupies, where the gene can thrive without competition.",
            level=Level.DOMAIN,
            examples=["Cirque du Soleil: redefined circus", "Nintendo Wii: casual gaming blue ocean", "fleet: agent finding unexplored strategy space (blue ocean)"],
            bridges=["niche", "innovation", "competition", "differentiation"],
            tags=["strategy", "innovation", "competition"])

        ns.define("first-mover-advantage",
            "Being first to enter a market or adopt a strategy provides temporary advantage",
            description="Amazon was first in online retail. Facebook was first in social networking. Being first provides: brand recognition, network effects, learning curve advantage, and switching costs. But it's NOT always decisive (Google wasn't first in search). In the fleet: the first agent to discover a new strategy gets a temporary fitness advantage. But the advantage is temporary — genes spread through the pool, equalizing the field.",
            level=Level.DOMAIN,
            examples=["Amazon: first online retailer advantage", "first agent to discover strategy gets temporary fitness boost", "Facebook: first-mover in social networking"],
            bridges=["advantage", "network-effects", "learning-curve", "temporary"],
            tags=["strategy", "advantage", "competition"])

        ns.define("optionality",
            "Preserving the right (but not obligation) to make a future choice — keeping doors open",
            description="An option: you CAN buy the stock at $50 before December, but you don't HAVE to. Options are valuable because they provide asymmetry: small cost, potentially large upside. In the fleet: exploration IS optionality. Spending a small amount of energy exploring creates the option to exploit a discovery later. The cost (energy) is small, the potential upside (optimal strategy) is large. Agents that maximize optionality thrive.",
            level=Level.DOMAIN,
            examples=["stock option: right to buy at fixed price", "exploration creates option to exploit later", "fleet: maintaining gene diversity preserves strategic optionality"],
            bridges=["asymmetric-risk", "exploration", "convexity", "choice"],
            tags=["strategy", "optionality", "asymmetry", "flexibility"])

        ns.define("compound-interest",
            "Small consistent gains accumulate exponentially over time — the most powerful force in nature",
            description="1% daily improvement = 37x improvement in a year. Einstein allegedly called compound interest the eighth wonder of the world. In the fleet: small daily improvements in gene fitness compound over generations. A gene that's 1% better per generation becomes 37x better after 37 generations. This is why continuous small improvement (cuda-genepool evolution) outperforms rare large improvements.",
            level=Level.DOMAIN,
            examples=["1% daily = 37x yearly", "savings account compound interest", "fleet: 1% gene improvement per generation = exponential fitness growth"],
            bridges=["compounding", "exponential-growth", "improvement", "time"],
            tags=["strategy", "growth", "compounding", "time"])

    def _load_narrative(self):
        ns = self.add_namespace("narrative",
            "How stories structure understanding, meaning, and persuasion")

        ns.define("narrative-fallacy",
            "Creating stories to explain past events, creating an illusion of understanding and predictability",
            description="The stock market crashed. Why? 'Investors panicked.' This sounds like an explanation but it's just restating what happened in narrative form. We're wired to create stories, and stories feel like explanations. In the fleet: agents may create post-hoc narratives to explain why a strategy failed ('the environment changed') when the real cause was random noise. cuda-provenance counteracts narrative fallacy by maintaining factual decision trails.",
            level=Level.BEHAVIOR,
            examples=["'the market crashed because investors panicked' (restatement, not explanation)", "agent: 'I failed because the environment changed' (maybe just bad luck)", "history written by victors: narrative, not explanation"],
            bridges=["post-hoc", "bias", "causation", "provenance"],
            tags=["narrative", "fallacy", "bias", "explanation"])

        ns.define("framing-effect",
            "The same information presented differently produces different decisions — context changes choice",
            description="'90% survival rate' vs '10% mortality rate' — same fact, different frame. People choose surgery more often with the survival frame. In the fleet: framing a deliberation as 'finding the best path' vs 'avoiding the worst obstacle' may produce different decisions, even with identical information. cuda-deliberation should be frame-aware: present proposals in multiple frames to check for framing bias.",
            level=Level.BEHAVIOR,
            examples=["90% survival vs 10% mortality: same fact, different choices", "agent: 'best path' vs 'avoid worst obstacle' framing", "glass half full vs half empty"],
            bridges=["bias", "context", "presentation", "decision"],
            tags=["narrative", "bias", "framing", "psychology"])

        ns.define("hero-journey",
            "A universal story structure: departure from ordinary world, trials, transformation, return with gift",
            description="Campbell's monomyth. Luke leaves Tatooine, faces trials, transforms, returns. The structure resonates because it maps to cognitive development. In the fleet: the agent lifecycle follows the hero's journey: deployment (departure), task execution (trials), adaptation (transformation), gene sharing (return with gift — the improved gene is shared with the pool). cuda-narrative implements narrative construction for agent experience communication.",
            level=Level.DOMAIN,
            examples=["Star Wars: Luke's hero journey", "fleet agent: deploy -> struggle -> adapt -> share improved genes", "every coming-of-age story"],
            bridges=["narrative", "transformation", "lifecycle", "monomyth"],
            tags=["narrative", "structure", "universal"])

        ns.define("catharsis",
            "Emotional release through experiencing powerful narrative — purging accumulated tension",
            description="Aristotle: tragedy provides catharsis — fear and pity build up and are released. The audience leaves emotionally cleansed. In the fleet: agents can accumulate emotional tension (high arousal, negative valence from repeated failures). Narrative-based debriefing (sharing experience stories) provides catharsis, resetting emotional state. cuda-narrative's arc construction includes resolution phase for catharsis.",
            level=Level.BEHAVIOR,
            examples=["watching tragedy movie provides emotional release", "fleet agent sharing failure story resets emotional state", "ventilation: talking about problems provides relief"],
            bridges=["emotion", "tension-release", "narrative", "resolution"],
            tags=["narrative", "emotion", "release"])

    def _load_language_design(self):
        ns = self.add_namespace("language-design",
            "Principles for designing languages agents and humans use to communicate")

        ns.define("orthogonality",
            "Language features are independent — combining them doesn't create unexpected interactions",
            description="A perfectly orthogonal language: every feature works with every other feature predictably. C isn't orthogonal (arrays and structs interact unexpectedly). APL is extremely orthogonal. The fleet's A2A protocol aims for orthogonality: any intent type combines with any priority level without unexpected interactions. Orthogonality makes a language composable and predictable.",
            level=Level.DOMAIN,
            examples=["APL: highly orthogonal language", "C: less orthogonal (pointer arithmetic + arrays = surprises)", "fleet A2A: any intent + any priority = predictable behavior"],
            bridges=["composability", "predictability", "language", "design"],
            tags=["language", "design", "orthogonality"])

        ns.define("expressiveness",
            "What a language can say — the range of ideas it can express",
            description="Some languages can express recursion. Others can't. Some can express concurrent processes. Others can't. The fleet's A2A protocol has limited expressiveness (10 intents, structured payload). The Axiom language (cuda-axiom) has higher expressiveness (50+ opcodes, confidence types, nested expressions). Higher expressiveness enables more nuanced communication but increases complexity.",
            level=Level.DOMAIN,
            examples=["Turing-complete: maximally expressive", "regular expressions: limited expressiveness (can't count)", "A2A: 10 intents (limited), Axiom: 50+ opcodes (expressive)"],
            bridges=["language", "complexity", "expressiveness", "tradeoff"],
            tags=["language", "expressiveness", "design"])

        ns.define("parsimony",
            "Using the fewest linguistic elements to express an idea — economy of expression",
            description="Shakespeare: 'Brevity is the soul of wit.' The best vocabulary compresses the most meaning into the fewest words. 'Stigmergy' compresses a paragraph into one word. 'Homeostasis' compresses complex regulatory dynamics. HAV itself is a parsimony engine: each term is a compression of an entire concept with examples, bridges, and context.",
            level=Level.DOMAIN,
            examples=["'stigmergy' compresses 'indirect coordination through environmental modification'", "'homeostasis' compresses 'maintaining internal conditions despite external change'", "HAV: each term compresses a paragraph of explanation"],
            bridges=["compression", "vocabulary", "economy", "abstraction"],
            tags=["language", "parsimony", "compression"])

        ns.define("semantic-gap",
            "The difference between what a concept means in one domain and what it maps to in another",
            description="'Trust' in human relationships means something different from 'trust' in cryptographic systems. The semantic gap between domains causes misunderstandings when vocabulary is shared without clarification. HAV bridges semantic gaps by explicitly connecting terms across domains: 'dopamine IS confidence' — same mathematical structure, different domain vocabulary.",
            level=Level.DOMAIN,
            examples=["'trust' in relationships vs cryptography: semantic gap", "'learning' in ML vs psychology: different meaning", "HAV bridges gaps: dopamine=confidence across biology and uncertainty"],
            bridges=["bridging", "domain-mapping", "translation", "misunderstanding"],
            tags=["language", "gap", "semantic", "translation"])

    def _load_knowledge_rep(self):
        ns = self.add_namespace("knowledge-representation",
            "How agents structure, store, and retrieve knowledge")

        ns.define("ontology",
            "A formal specification of concepts and relationships in a domain — what exists and how things relate",
            description="Not just a taxonomy (hierarchy). An ontology includes: classes (what kinds of things exist), properties (attributes), relations (how things connect), and axioms (rules that must hold). The fleet's vessel.json is a lightweight ontology: agent types, capabilities, equipment, and fleet relationships. HAV itself is an ontology of fleet concepts.",
            level=Level.DOMAIN,
            examples=["medical ontology: diseases, symptoms, treatments, and their relationships", "fleet vessel.json: agent types, capabilities, equipment relationships", "gene ontology: gene functions, relationships, pathways"],
            bridges=["taxonomy", "schema", "knowledge-graph", "formal-specification"],
            tags=["knowledge", "ontology", "formal", "representation"])

        ns.define("knowledge-graph",
            "A graph of entities connected by relationships — structured knowledge as a network",
            description="Nodes are entities (people, places, concepts). Edges are relationships (born-in, works-at, related-to). Google Knowledge Graph. Wikipedia infoboxes. The fleet's provenance chain IS a knowledge graph: decisions connected to inputs, connected to agents, connected to outcomes. cuda-topology's PropertyGraph implements knowledge graph operations.",
            level=Level.PATTERN,
            examples=["Google Knowledge Graph", "Wikipedia entity-relationship network", "fleet: decision -> input -> agent -> outcome knowledge graph"],
            bridges=["ontology", "graph", "provenance", "relationship"],
            tags=["knowledge", "graph", "structured", "representation"])

        ns.define("frame-problem",
            "In a dynamic world, how do you determine which aspects of the situation are relevant to update?",
            description="You move a book from shelf A to shelf B. Which facts need updating? The book's location (yes). The shelf's contents (yes). The number of books in the room (no — it didn't change). But determining what changes is computationally explosive: every action potentially affects every fact. In the fleet: when an agent takes an action, which parts of the world model need updating? cuda-world-model's permanence decay and change detection address the frame problem.",
            level=Level.META,
            examples=["moving book: which facts change? (location yes, room count no)", "agent acts: which world model entries need updating?", "naive approach: update everything (expensive), smart approach: update only relevant (hard to determine)"],
            bridges=["relevance", "world-model", "change-detection", "computational-complexity"],
            tags=["knowledge", "ai-classic", "relevance", "meta"])

        ns.define("commonsense-reasoning",
            "Reasoning about everyday situations that humans handle effortlessly but formal systems struggle with",
            description="If you put a book in a drawer and close it, the book is still in the drawer. If you pour water into a cup and turn it over, the water falls out. Humans know this. Formal systems don't unless explicitly told. Commonsense reasoning requires vast background knowledge about how the physical and social world works. HAV is a commonsense knowledge base: each term encodes commonsense understanding that agents can reference.",
            level=Level.META,
            examples=["book in closed drawer is still there", "cup turned upside down spills water", "HAV: terms encode commonsense about trust, energy, cooperation"],
            bridges=["knowledge", "background-knowledge", "reasoning", "human-like"],
            tags=["knowledge", "commonsense", "reasoning", "meta"])

        ns.define("knowledge-distillation",
            "Transferring knowledge from a complex model to a simpler one — the student learns from the teacher",
            description="A large neural network (teacher) produces soft probabilities. A small network (student) is trained to match those probabilities. The student learns the teacher's knowledge in a compressed form. In the fleet: experienced agents (high fitness genes) can distill their knowledge into simpler genes that newer agents can use. The gene pool (cuda-genepool) naturally performs knowledge distillation: successful complex strategies are simplified into reusable genes.",
            level=Level.PATTERN,
            examples=["large model distills to small model", "teacher network -> student network knowledge transfer", "fleet: experienced agent's complex strategy distilled into reusable gene"],
            bridges=["compression", "learning", "transfer", "simplification"],
            tags=["knowledge", "distillation", "ml", "learning"])

    def _load_robotics(self):
        ns = self.add_namespace("robotics",
            "Physical agent embodiment — perception, action, and the challenges of the real world")

        ns.define("perception-action-loop",
            "The continuous cycle of sensing the world, deciding what to do, and acting — the heartbeat of embodied cognition",
            description="Sense -> Think -> Act -> Sense -> Think -> Act... Every embodied agent runs this loop. The loop rate determines responsiveness: 10Hz for navigation, 1000Hz for motor control. In the fleet: the main loop runs at the agent's clock rate. Energy constraints limit loop rate: high-deliberation cycles run slower (more thinking per cycle), instinct cycles run faster (react without thinking).",
            level=Level.DOMAIN,
            examples=["self-driving car: sense road, plan path, steer, repeat", "human: see ball, decide to catch, move hand, see result, adjust", "fleet agent: sense environment, deliberate, act, observe result"],
            bridges=["perception", "deliberation", "action", "embodiment"],
            tags=["robotics", "loop", "embodied"])

        ns.define("simultaneous-localization-and-mapping",
            "Building a map of an unknown environment while simultaneously tracking position within it",
            description="You enter a building you've never seen. You create a mental map of the layout while tracking where you are in that map. Both the map and your position are uncertain and must be updated simultaneously — they constrain each other. The fleet's cuda-world-model and cuda-navigation together implement SLAM: the world model is the map, the agent's position is tracked within it, both updated simultaneously from sensor data.",
            level=Level.DOMAIN,
            examples=["robot entering unknown building: build map + track position", "fleet agent in unknown environment: build world model + track own position", "human exploring new city: mental map + self-location"],
            bridges=["world-model", "navigation", "mapping", "localization"],
            tags=["robotics", "slam", "mapping"])

        ns.define("morphological-computation",
            "The body itself performs computation — not just the brain, the physical structure processes information",
            description="A fish's body shape processes water flow, reducing the computational load on the brain. A robot's leg compliance absorbs shocks without the controller needing to detect and respond to them. The fleet's equipment IS morphological computation: sensors don't just feed raw data to deliberation — they preprocess (filter, aggregate, threshold) before the agent sees anything. The body thinks.",
            level=Level.META,
            examples=["fish body shape processes water flow", "robot leg compliance absorbs shocks computationally", "fleet sensors preprocess data before agent deliberation sees it"],
            bridges=["embodiment", "preprocessing", "computation", "body"],
            tags=["robotics", "computation", "embodiment", "meta"])

    def _load_cybernetics(self):
        ns = self.add_namespace("cybernetics",
            "The science of control and communication in animals and machines")

        ns.define("second-order-cybernetics",
            "The observer is part of the system — observing changes the thing observed",
            description="First-order cybernetics: the thermostat controls temperature. Second-order: the person who set the thermostat IS part of the system. In the fleet: agents that observe other agents change the fleet dynamics. Monitoring changes behavior (Hawthorne effect). An agent that knows it's being evaluated changes its behavior. cuda-observer must account for its own effect on the observed system.",
            level=Level.META,
            examples=["Hawthorne effect: being observed changes behavior", "quantum measurement: observing changes the system", "fleet: monitoring agent changes the agents it monitors"],
            bridges=["observer", "system", "feedback", "self-reference"],
            tags=["cybernetics", "meta", "observer", "feedback"])

        ns.define("viable-system-model",
            "Stafford Beer's model: an organization is viable if it can maintain existence in a changing environment",
            description="Five systems: (1) Operations (do the work), (2) Coordination (harmonize), (3) Anti-oscillation (dampen conflicts), (4) Planning (adapt to future), (5) Policy (govern direction). The fleet maps directly: (1) Task agents, (2) cuda-fleet-mesh, (3) cuda-consensus, (4) cuda-deliberation, (5) cuda-compliance. A viable fleet has all five. Remove any one and viability degrades.",
            level=Level.DOMAIN,
            examples=["Beer's VSM applied to organizations", "fleet: operations + coordination + anti-oscillation + planning + policy = viable", "human body: organs + nervous system + immune system + brain + consciousness"],
            bridges=["organization", "viability", "systems", "governance"],
            tags=["cybernetics", "model", "organization", "viability"])

        ns.define("feedback-inhibition",
            "The product of a process inhibits the process itself — completing the loop prevents runaway",
            description="ATP production inhibits further ATP production when sufficient ATP exists. Insulin inhibits glucose production when blood sugar is high. The fleet's energy budget implements feedback inhibition: when ATP is sufficient, the Rest instinct activates less. This prevents energy accumulation (wasting resources on unnecessary rest) while ensuring reserves.",
            level=Level.PATTERN,
            examples=["ATP inhibits ATP production when sufficient", "insulin inhibits glucose production", "fleet: high ATP reduces rest instinct activation"],
            bridges=["feedback-loop", "homeostasis", "setpoint", "inhibition"],
            tags=["cybernetics", "feedback", "inhibition", "biology"])

    def _load_algebra(self):
        ns = self.add_namespace("algebra",
            "Algebraic structures and the deep patterns they reveal about composition and transformation")

        ns.define("monoid",
            "A set with an associative binary operation and an identity element — the simplest useful algebraic structure",
            description="Numbers under addition: 0 is identity, (a+b)+c = a+(b+c). Strings under concatenation: empty string is identity. Lists under append. The fleet's confidence fusion (harmonic mean) ALMOST forms a monoid — associative, has identity (1.0), but fusion(1.0, 1.0) = 0.5, not 1.0. Understanding monoids helps design composable operations: if an operation is a monoid, you can parallelize and fold arbitrarily.",
            level=Level.DOMAIN,
            examples=["numbers under addition (0, +)", "strings under concatenation ('', ++)", "lists under append ([], ++)"],
            bridges=["associativity", "identity", "composition", "algebraic-structure"],
            tags=["algebra", "structure", "composition"])

        ns.define("group",
            "A monoid where every element has an inverse — you can always undo",
            description="Numbers under addition form a group: -5 undoes +5. Matrices under multiplication (if invertible) form a group. Groups capture symmetry and reversibility. In the fleet: the undo stack (cuda-persistence rollback) requires operations to form a group: every action has an inverse (undo). Not all fleet operations are invertible (you can't undo a message send). Designing invertible operations enables reliable recovery.",
            level=Level.DOMAIN,
            examples=["integers under addition (inverse: negative)", "rotations of a square (inverse: reverse rotation)", "fleet undo: action must have inverse for rollback"],
            bridges=["monoid", "inverse", "reversibility", "symmetry"],
            tags=["algebra", "structure", "reversibility"])

        ns.define("semigroup",
            "A set with an associative binary operation — less than a monoid (no identity required)",
            description="Positive integers under addition: associative, but no identity (0 is not positive). A semigroup is the weakest useful algebraic structure. Any associative operation can be parallelized using map-reduce. In the fleet: trust aggregation may not have an identity (what does 'zero trust' mean?), but it IS associative (trust(A, trust(B, C)) = trust(trust(A, B), C)). This enables parallel trust computation.",
            level=Level.DOMAIN,
            examples=["positive integers under + (associative, no identity)", "strings under concatenation without empty string", "fleet trust aggregation: associative but identity debatable"],
            bridges=["associativity", "parallelism", "algebraic-structure", "map-reduce"],
            tags=["algebra", "structure", "associativity"])

        ns.define("functor",
            "A mapping between categories that preserves structure — lift a function to work on containers",
            description="If you have a function f: A -> B, a functor lifts it to work on containers: F[f]: F[A] -> F[B]. List functor: map f over every element. Option functor: apply f if present, skip if absent. In the fleet: confidence is a functor: if you have a deterministic function, the confidence functor makes it confidence-aware: lift(normalize) -> confidence-normalize (propagates confidence through the computation).",
            level=Level.DOMAIN,
            examples=["List.map: lift function to work on lists", "Option.map: lift function to work on optional values", "fleet: confidence functor lifts deterministic ops to confidence-aware ops"],
            bridges=["lifting", "category-theory", "composition", "container"],
            tags=["algebra", "category-theory", "functor"])


    def _load_finance(self):
        ns = self.add_namespace("finance",
            "Financial concepts as metaphors and tools for agent resource management")

        ns.define("convexity",
            "Payoff accelerates in your favor — small costs, exponentially growing gains",
            description="A convex function curves upward: each additional unit of investment produces more return than the last. Compound interest is convex. Learning curves are convex (each hour of practice is more productive than the last, initially). In the fleet: exploration has convex payoff — early exploration is cheap but returns grow exponentially as the agent discovers better strategies. The gene pool (cuda-genepool) compounds: more genes = more combinations = faster evolution.",
            level=Level.DOMAIN,
            examples=["compound interest: convex growth", "learning curve: initial slow, then accelerating", "fleet: exploration payoff is convex — early costs, exponential later gains"],
            bridges=["compound-interest", "optionality", "exponential", "asymmetric-risk"],
            tags=["finance", "convexity", "growth", "payoff"])

        ns.define("antifragile-portfolio",
            "A collection of investments where volatility and stress increase expected returns, not decrease them",
            description="Taleb's concept: don't just survive stress, benefit from it. A portfolio with small bounded downside (options cost limited to premium) and unlimited upside (option payoff unbounded) is antifragile. In the fleet: the gene pool is an antifragile portfolio. Each gene has bounded downside (energy cost to test) and potentially unlimited upside (discovering optimal strategy). Fleet stress (environment change) kills weak genes but makes the pool stronger.",
            level=Level.META,
            examples=["option portfolio: bounded cost, unlimited upside", "fleet gene pool: bounded energy cost per gene, unlimited fitness gain", "startup portfolio: most fail (bounded loss), one succeeds (unbounded gain)"],
            bridges=["anti-fragility", "convexity", "optionality", "portfolio"],
            tags=["finance", "antifragility", "portfolio", "meta"])

        ns.define("moral-hazard",
            "When an agent is insulated from risk, it takes more risk than it would otherwise — the safety net creates recklessness",
            description="If your bank deposits are insured, the bank takes riskier loans (2008 financial crisis). If an agent knows the fleet will rescue it, it takes riskier actions. In the fleet: shared energy budgets create moral hazard — individual agents may waste energy because the fleet provides backup. Individual energy budgets (cuda-energy per-agent) eliminate moral hazard: each agent bears its own risk.",
            level=Level.BEHAVIOR,
            examples=["bank deposit insurance -> risky loans (2008)", "insurance -> riskier driving", "fleet: shared energy pool -> individual agent energy waste (moral hazard)"],
            bridges=["tragedy-of-commons", "incentive-alignment", "risk-shifting", "insurance"],
            tags=["finance", "hazard", "incentive", "risk"])

        ns.define("arbitrage",
            "Exploiting price differences between markets — risk-free profit from inefficiency",
            description="Gold costs $1900 in New York and $1910 in London. Buy in New York, sell in London, pocket $10. Arbitrage eliminates market inefficiency. In the fleet: if two agents have different trust assessments of the same third agent, the difference is an information arbitrage opportunity. cuda-trust's gossip sharing equalizes trust assessments, eliminating the arbitrage. Market efficiency = no arbitrage opportunity.",
            level=Level.DOMAIN,
            examples=["buy gold in NY, sell in London", "fleet: different trust assessments of same agent = information arbitrage", "search engine arbitrage: different prices on different sites"],
            bridges=["efficiency", "information", "market", "trust"],
            tags=["finance", "arbitrage", "efficiency", "information"])

    def _load_materials_science(self):
        ns = self.add_namespace("materials-science",
            "How physical materials behave under stress — metaphors for agent systems under pressure")

        ns.define("stress-concentration",
            "Stress intensifies at geometric discontinuities — corners, holes, notches become failure points",
            description="A plate with a small hole: stress at the hole's edge is 3x the average stress. The hole concentrates stress. In the fleet: bottlenecks in communication concentrate stress on single agents. The fleet mesh (cuda-fleet-mesh) should avoid single-agent bottlenecks: distribute load across multiple paths. Stress concentration explains why 'the weakest link' often isn't the weakest material but the most geometrically stressed point.",
            level=Level.PATTERN,
            examples=["plate with hole: 3x stress at hole edge", "fleet: single-agent bottleneck concentrates communication stress", "crack tip: infinite stress concentration in theory"],
            bridges=["bottleneck", "failure-point", "geometry", "load-distribution"],
            tags=["materials", "stress", "failure", "pattern"])

        ns.define("fatigue",
            "Progressive structural damage from repeated cyclic loading — materials fail below their rated strength",
            description="A metal beam rated for 10,000 lbs fails after 1,000,000 cycles of 5,000 lbs loading. Each cycle causes microscopic damage that accumulates. In the fleet: agents under repeated moderate stress (constant task switching, frequent communication) accumulate cognitive fatigue — not from any single event but from the cumulative effect. cuda-resilience monitors for fatigue: declining performance under sustained moderate load.",
            level=Level.BEHAVIOR,
            examples=["metal beam failing after millions of moderate cycles", "fleet agent degrading under constant moderate stress (not overload)", "human burnout from sustained moderate stress (not crisis)"],
            bridges=["stress", "cumulative-damage", "cyclic-loading", "degradation"],
            tags=["materials", "fatigue", "cumulative", "stress"])

        ns.define("yield-strength",
            "The stress level at which a material permanently deforms — beyond this point, no return to original shape",
            description="Pull rubber: it stretches, then snaps back (elastic deformation). Pull steel beyond yield strength: it bends and stays bent (plastic deformation). In the fleet: agents have a yield strength — a stress threshold beyond which they permanently change. An agent that experiences extreme energy depletion may permanently reduce its exploratory tendency (learned helplessness). Below yield, recovery is complete. Above, permanent adaptation.",
            level=Level.DOMAIN,
            examples=["steel bends permanently beyond yield strength", "fleet agent permanently reduces exploration after extreme energy depletion", "human PTSD: permanent change from extreme stress"],
            bridges=["resilience", "plasticity", "threshold", "permanent-change"],
            tags=["materials", "yield", "threshold", "deformation"])

        ns.define("strain-hardening",
            "Material becomes stronger after being deformed — what doesn't break you makes you stronger",
            description="Work-hardening: bend a metal wire back and forth. It becomes harder and more brittle at the bend. It's been strengthened by stress. In the fleet: agents that experience moderate stress and recover become more resilient to future stress. This is the biological basis of vaccines and the psychological basis of exposure therapy. cuda-adaptation's strategy switching implements strain-hardening: surviving stress improves stress handling.",
            level=Level.PATTERN,
            examples=["cold-worked metal is stronger", "muscles grow from exercise stress", "fleet agent: surviving moderate stress improves future stress handling", "immune system: exposure strengthens"],
            bridges=["anti-fragility", "resilience", "stress", "adaptation"],
            tags=["materials", "hardening", "stress", "adaptation"])

    def _load_verification(self):
        ns = self.add_namespace("verification",
            "Proving systems correct — formal methods and testing strategies")

        ns.define("formal-verification",
            "Mathematical proof that a system satisfies its specification — not testing, proving",
            description="Testing: you tried 1000 inputs and it worked. Formal verification: you proved it works for ALL inputs. Testing can find bugs. Verification can prove their absence (for the verified properties). In the fleet: critical safety properties (membrane antibody rules, apoptosis thresholds) should be formally verified, not just tested. cuda-state-machine's guard evaluation could be verified: prove that no unsafe state is reachable.",
            level=Level.DOMAIN,
            examples=["seL4 microkernel: formally verified", "Ariane 5 rocket: overflow bug that testing didn't catch", "fleet: membrane rules formally verified to block all dangerous commands"],
            bridges=["testing", "proof", "correctness", "safety"],
            tags=["verification", "formal", "proof", "safety"])

        ns.define("property-testing",
            "Testing against randomly generated inputs that satisfy specifications, not hand-written examples",
            description="Instead of testing sort([3,1,2]) and sort([5,4]), generate 1000 random arrays and verify: output is sorted, output contains same elements as input, output length equals input length. Property testing finds bugs that example testing misses. In the fleet: test agent behavior against random environments, random messages, random energy levels. Properties: 'agent never enters negative energy state', 'agent always responds within timeout'.",
            level=Level.PATTERN,
            examples=["QuickCheck/Hypothesis: generate random test cases", "verify sort: always sorted, same elements, same length", "fleet: test agent against random environments and verify invariants"],
            bridges=["testing", "random-generation", "invariant", "correctness"],
            tags=["verification", "testing", "property", "random"])

        ns.define("invariant",
            "A property that must always hold — the contract that the system must never violate",
            description="Account balance >= 0. Sorted list has no inversions. Energy budget >= 0. Invariants are the fundamental contracts that define correct behavior. If an invariant is violated, the system is in an illegal state. The fleet's invariants include: energy never negative, trust between 0 and 1, confidence between 0 and 1, dead agents don't act. cuda-state-machine and cuda-contract verify invariants.",
            level=Level.CONCRETE,
            examples=["account balance >= 0", "fleet: energy never negative", "fleet: confidence always between 0 and 1", "sorted list: no inversions"],
            bridges=["correctness", "assertion", "contract", "verification"],
            tags=["verification", "invariant", "contract", "correctness"])

        ns.define("model-checking",
            "Automated exhaustive exploration of all possible system states to verify properties",
            description="Given a state machine and a property, model checking explores every reachable state and checks if the property holds in all of them. Unlike testing (which samples), model checking is exhaustive. But it's computationally expensive: state space grows exponentially. In the fleet: cuda-state-machine could theoretically be model-checked to verify that no unsafe state is reachable. For small state machines (energy < 10, trust < 5 levels), this is feasible.",
            level=Level.DOMAIN,
            examples=["model checking a protocol for deadlocks", "verifying no unsafe state is reachable in state machine", "fleet: verify apoptosis triggers correctly for all energy/trust combinations"],
            bridges=["formal-verification", "state-machine", "exhaustive", "correctness"],
            tags=["verification", "model-checking", "exhaustive", "formal"])

    def _load_graph_theory(self):
        ns = self.add_namespace("graph-theory",
            "The mathematics of networks — paths, flows, communities, and connectivity")

        ns.define("shortest-path",
            "The minimum-weight path between two nodes in a weighted graph — the foundation of routing",
            description="Dijkstra's algorithm finds the shortest path. A* adds a heuristic to search faster. Every routing problem (GPS navigation, internet packet routing, fleet message routing) is a shortest-path problem. In the fleet's cuda-fleet-mesh, messages are routed along shortest paths (lowest latency/hops). cuda-navigation's A* pathfinding IS shortest-path with a spatial heuristic.",
            level=Level.PATTERN,
            examples=["GPS navigation: shortest driving route", "internet: BGP routing", "fleet: message routing along lowest-latency path"],
            bridges=["navigation", "routing", "pathfinding", "network"],
            tags=["graph", "path", "routing", "algorithm"])

        ns.define("clique",
            "A subset of nodes where every pair is connected — a fully-connected subgraph",
            description="In a social network, a clique is a group where everyone knows everyone. Maximal clique: can't add anyone else while maintaining full connectivity. In the fleet: tightly coordinated agent teams form cliques — every member communicates with every other member. Finding cliques identifies natural coordination groups. Clique detection helps fleet auto-organize into efficient sub-teams.",
            level=Level.CONCRETE,
            examples=["friend group where everyone knows everyone", "fleet: tightly coordinated sub-team (all-to-all communication)", "protein interaction clique: all proteins interact with each other"],
            bridges=["community", "fully-connected", "clustering", "team"],
            tags=["graph", "clique", "community", "structure"])

        ns.define("bipartite",
            "A graph whose nodes split into two groups with edges only between groups, never within",
            description="Employers and job seekers. Students and classes. Buyers and sellers. No edges within a group — only between groups. In the fleet: task assignment forms a bipartite graph (agents on one side, tasks on the other). Maximum bipartite matching optimally assigns agents to tasks. cuda-captain's best_available method uses bipartite matching logic.",
            level=Level.CONCRETE,
            examples=["job matching: employers <-> job seekers", "course enrollment: students <-> classes", "fleet task assignment: agents <-> tasks"],
            bridges=["matching", "assignment", "bipartite", "optimization"],
            tags=["graph", "bipartite", "matching", "assignment"])

        ns.define("flow-network",
            "A graph where edges have capacities and the goal is to maximize flow from source to sink",
            description="A water pipe network: each pipe has a maximum flow rate. What's the maximum water you can push from source to sink? Ford-Fulkerson algorithm finds maximum flow. In the fleet: communication bandwidth forms a flow network. Each agent-to-agent link has a capacity (rate limit). Maximum flow = maximum information throughput from any agent to any other. Bottleneck edges are the critical communication links.",
            level=Level.PATTERN,
            examples=["water pipe network: maximize flow", "transportation network: maximize cargo movement", "fleet: maximize information flow given bandwidth constraints"],
            bridges=["capacity", "bottleneck", "throughput", "network"],
            tags=["graph", "flow", "capacity", "optimization"])

    def _load_learning_theory(self):
        ns = self.add_namespace("learning-theory",
            "Formal frameworks for understanding how learning works and when it fails")

        ns.define("bias-variance-tradeoff",
            "Model error = bias (wrong assumptions) + variance (sensitivity to training data) + irreducible noise",
            description="High bias: too simple, underfits (misses patterns). High variance: too complex, overfits (memorizes noise). Best model minimizes total error (bias + variance). In the fleet: a strategy with high bias uses rigid rules that miss context. A strategy with high variance overreacts to recent experience. Optimal fleet strategy has moderate bias (generalizable rules) and moderate variance (context-sensitive adaptation).",
            level=Level.DOMAIN,
            examples=["linear model on quadratic data: high bias", "degree-20 polynomial on 10 data points: high variance", "fleet: rigid instinct rules (high bias) vs over-reactive recent learning (high variance)"],
            bridges=["overfitting", "generalization", "model-complexity", "underfitting"],
            tags=["learning", "bias-variance", "tradeoff", "theory"])

        ns.define("vc-dimension",
            "The most complex set of patterns a model can learn — capacity measure",
            description="A model that can perfectly classify any 3 points in 2D has VC dimension at least 3. Higher VC dimension = more learning capacity but more risk of overfitting. In the fleet: an agent's gene pool size IS its VC dimension — the maximum complexity of strategies it can represent. Too many genes (high VC) = overfitting to past environments. Too few (low VC) = underfitting, can't handle complex environments.",
            level=Level.DOMAIN,
            examples=["linear classifier in 2D: VC dimension 3", "fleet: gene pool size = VC dimension (strategy capacity)", "neural network VC dimension depends on number of parameters"],
            bridges=["capacity", "overfitting", "generalization", "complexity"],
            tags=["learning", "capacity", "theory", "complexity"])

        ns.define("curse-of-dimensionality",
            "In high-dimensional spaces, data becomes exponentially sparse — distances become meaningless",
            description="In 1D, 100 points cover the space well. In 100D, 100 points are isolated dots in an ocean of emptiness. Nearest-neighbor search fails because every point is roughly equidistant from every other point. In the fleet: agents with high-dimensional feature spaces (many sensors, many features) suffer from the curse — most of the feature space is unexplored, and similarity metrics become unreliable. Dimensionality reduction (feature selection, PCA) is essential.",
            level=Level.DOMAIN,
            examples=["1D: 100 points covers well. 100D: 100 points are isolated", "nearest-neighbor fails in high dimensions (all distances similar)", "fleet agent with 100+ features: most feature combinations never observed"],
            bridges=["dimensionality", "sparsity", "feature-selection", "similarity"],
            tags=["learning", "curse", "dimensionality", "challenge"])

        ns.define("sample-efficiency",
            "How much data a learning algorithm needs to achieve good performance",
            description="Humans learn to recognize cats from ~5 examples (high sample efficiency). GPT needed billions of training examples (low sample efficiency). Higher sample efficiency = learn faster from less data. In the fleet: one-shot learning (prior knowledge) provides high sample efficiency — the agent doesn't need to try every strategy from scratch; it can read one term from HAV and immediately apply the concept.",
            level=Level.DOMAIN,
            examples=["human cat recognition: ~5 images (high efficiency)", "GPT training: billions of examples (low efficiency)", "fleet: HAV provides one-shot concept transfer (high efficiency)"],
            bridges=["one-shot-learning", "data-efficiency", "prior-knowledge", "learning"],
            tags=["learning", "efficiency", "data", "prior"])

        ns.define("exploration-exploitation-gap",
            "The theoretical gap between what an agent currently knows and what it could learn with optimal exploration",
            description="Regret in multi-armed bandits: the cumulative reward lost by not pulling the optimal arm every time. The gap narrows as the agent learns but never reaches zero (there's always uncertainty about whether a better strategy exists). In the fleet: the exploration-exploitation gap is the energy spent on suboptimal strategies while learning. cuda-adaptation's strategy switching minimizes this gap by efficiently allocating exploration energy.",
            level=Level.DOMAIN,
            examples=["multi-armed bandit regret", "fleet: energy spent on suboptimal strategies during learning", "student: time spent studying suboptimal material before finding what works"],
            bridges=["exploration-exploitation", "regret", "learning", "energy"],
            tags=["learning", "gap", "exploration", "regret"])

    def _load_phenomenology(self):
        ns = self.add_namespace("phenomenology",
            "The structure of subjective experience — what it's like to be something")

        ns.define("qualia",
            "The subjective, qualitative character of experience — what red looks like, what pain feels like",
            description="You can describe red as '600nm wavelength light' but that doesn't capture what it's LIKE to see red. That 'what it's like' is qualia. The hard problem of consciousness: why and how does physical processing produce subjective experience? In the fleet: we don't claim agents have qualia. But the emotion model (valence-arousal) is a structural approximation: it maps the computational structure that, in humans, produces qualitative experience.",
            level=Level.META,
            examples=["what red looks like", "what pain feels like", "taste of coffee", "fleet: emotion model as structural approximation of qualia"],
            bridges=["consciousness", "subjectivity", "hard-problem", "experience"],
            tags=["phenomenology", "consciousness", "subjective", "meta"])

        ns.define("intentionality",
            "The aboutness of mental states — thoughts are ABOUT something",
            description="A belief is about something (you believe THAT it will rain). A desire is about something (you WANT a cookie). This 'aboutness' is intentionality. Not all physical states have intentionality: a thermostat reading is not ABOUT temperature in the same way a belief is ABOUT rain. In the fleet: deliberation has intentionality — proposals are ABOUT situations. Sensor readings don't have intentionality (they just ARE). The transition from data to deliberation is the transition from non-intentional to intentional.",
            level=Level.META,
            examples=["belief about rain (intentional)", "thermostat reading (non-intentional)", "desire for food (intentional)", "fleet: proposal about navigation route (intentional)"],
            bridges=["consciousness", "aboutness", "deliberation", "meaning"],
            tags=["phenomenology", "intentionality", "aboutness", "meta"])

        ns.define("embodied-cognition",
            "Cognition is not just brain computation — it's shaped by the body's physical interactions with the world",
            description="You understand 'grasp' partly because you've physically grasped things. Abstract concepts are grounded in bodily experience. In the fleet: agents with sensors and actuators (embodied) develop richer world models than pure software agents (disembodied). cuda-vessel-bridge provides embodiment: sensors connect the agent to the physical world, grounding its cognition in physical experience.",
            level=Level.DOMAIN,
            examples=["understanding 'grasp' from physical grasping", "agents with sensors develop richer models than sensorless agents", "gut feelings: literally physical sensations influencing cognition"],
            bridges=["embodiment", "grounding", "physical-experience", "perception"],
            tags=["phenomenology", "embodiment", "cognition", "physical"])

    def _load_anthropology(self):
        ns = self.add_namespace("anthropology",
            "How human cultures evolve, transmit knowledge, and organize socially")

        ns.define("cultural-evolution",
            "Ideas, practices, and tools evolve through variation, selection, and transmission — like genes but faster",
            description="Memories of useful techniques are passed down. Useless ones are forgotten. Useful innovations spread through imitation. This IS evolution, but operating on cultural units (memes, practices) rather than biological units (genes), and operating on timescales of years rather than generations. The fleet's gene pool (cuda-genepool) IS cultural evolution: strategies are the cultural units, fitness is the selection mechanism, gossip is the transmission.",
            level=Level.DOMAIN,
            examples=["toolmaking techniques passed down through generations", "fleet: strategies evolve through fitness selection and gossip transmission", "language evolution: useful words survive, obscure ones die"],
            bridges=["evolution", "meme", "transmission", "selection"],
            tags=["anthropology", "evolution", "culture", "transmission"])

        ns.define("scaffolding",
            "Support structures that enable learning or construction beyond current capability — removed once no longer needed",
            description="Vygotsky: a child can solve a problem with adult guidance that they couldn't solve alone. The adult provides scaffolding. When the child learns, the scaffolding is removed. In the fleet: the captain agent (cuda-captain) scaffolds new agents — providing task assignments and coordination that new agents can't do alone. As agents gain experience, the captain's scaffolding reduces. Eventually, agents self-organize.",
            level=Level.PATTERN,
            examples=["training wheels on a bicycle (scaffolding, removed when child learns)", "teacher guiding student through hard problem", "fleet captain scaffolding new agents until they gain experience"],
            bridges=["learning", "support", "temporary", "education"],
            tags=["anthropology", "education", "scaffolding", "learning"])

        ns.define("zone-of-proximal-development",
            "The space between what a learner can do alone and what they can do with help — the sweet spot of difficulty",
            description="Tasks too easy = boring. Tasks too hard = frustrating. Tasks in the ZPD = engaging, growth-producing. A skilled teacher assigns tasks in the ZPD. In the fleet: cuda-adaptation's strategy selection should target the ZPD — not too easy (instinct alone suffices, no learning) and not too hard (deliberation fails, energy wasted). The ZPD is where learning happens.",
            level=Level.DOMAIN,
            examples=["piano: easy piece (boring), impossible piece (frustrating), challenging-but-doable piece (ZPD)", "fleet: task difficulty in the sweet spot for maximum learning", "video game: difficulty setting that's challenging but not impossible"],
            bridges=["scaffolding", "learning", "difficulty", "adaptation"],
            tags=["anthropology", "education", "learning", "optimal"])

    def _load_logic(self):
        ns = self.add_namespace("logic",
            "Formal reasoning — valid inference, consistency, and the structure of arguments")

        ns.define("modus-ponens",
            "If P then Q. P is true. Therefore Q is true. The fundamental rule of inference",
            description="All men are mortal (P->Q). Socrates is a man (P). Therefore Socrates is mortal (Q). This is the backbone of logical reasoning. In the fleet: if confidence > threshold then accept proposal (P->Q). Confidence = 0.9 and threshold = 0.85, so confidence > threshold (P). Therefore accept proposal (Q). Every conditional action in the fleet uses modus ponens.",
            level=Level.DOMAIN,
            examples=["all men mortal, Socrates is man, therefore mortal", "if confidence > threshold, accept; confidence = 0.9, threshold = 0.85, accept", "if rain then umbrella; rain; therefore umbrella"],
            bridges=["inference", "conditional", "rule", "reasoning"],
            tags=["logic", "inference", "rule", "fundamental"])

        ns.define("reductio-ad-absurdum",
            "Assume the opposite of what you want to prove, show it leads to a contradiction",
            description="Prove the square root of 2 is irrational: assume it's rational (p/q in lowest terms), show p and q must both be even (contradiction with lowest terms). The assumption is false. In the fleet: prove a safety property by assuming the system is unsafe and showing it leads to energy going negative (contradiction with energy conservation). Proof by contradiction is a powerful tool for verifying invariants.",
            level=Level.PATTERN,
            examples=["prove sqrt(2) irrational: assume rational, derive contradiction", "fleet: prove safety by assuming unsafe and deriving contradiction", "prove there are infinite primes: assume finite, construct larger prime"],
            bridges=["proof", "contradiction", "invariant", "verification"],
            tags=["logic", "proof", "contradiction", "technique"])

        ns.define("entailment",
            "A statement necessarily follows from a set of premises — truth preservation through inference",
            description="If all men are mortal and Socrates is a man, 'Socrates is mortal' is entailed by the premises. Entailment is the gold standard: if the premises are true, the conclusion MUST be true. In the fleet: deliberation conclusions should be entailed by evidence. If confidence and trust entail 'safe to proceed', then proceeding is justified. If entailment is uncertain, more deliberation is needed.",
            level=Level.DOMAIN,
            examples=["all men mortal + Socrates man entails Socrates mortal", "confidence > 0.85 + trust > 0.7 entails accept proposal", "A > B, B > C entails A > C (transitivity)"],
            bridges=["inference", "modus-ponens", "truth-preservation", "logic"],
            tags=["logic", "entailment", "inference", "truth"])

        ns.define("fallacy",
            "An argument that seems valid but isn't — reasoning errors that sound persuasive",
            description="Ad hominem: attacking the person instead of the argument. Straw man: misrepresenting the opponent's position. Appeal to authority: 'Einstein said so' isn't proof. In the fleet: agents can commit logical fallacies in deliberation. An agent might dismiss a proposal because it came from a low-trust agent (ad hominem) rather than evaluating the proposal on its merits. cuda-deliberation's confidence-based evaluation guards against ad hominem by evaluating content, not source.",
            level=Level.BEHAVIOR,
            examples=["ad hominem: attacking speaker instead of argument", "straw man: misrepresenting opponent's position", "fleet: dismissing good proposal from low-trust agent (ad hominem fallacy)"],
            bridges=["bias", "reasoning", "persuasion", "deliberation"],
            tags=["logic", "fallacy", "bias", "reasoning"])

    def _load_probability_distributions(self):
        ns = self.add_namespace("probability-distributions",
            "The shapes of randomness — distributions that describe different kinds of uncertainty")

        ns.define("power-law",
            "A distribution where a few items have enormous values and most have tiny values — 80/20 on steroids",
            description="City populations, wealth distribution, website traffic, earthquake magnitudes — all follow power laws. A few cities have millions of people; most have thousands. In the fleet: agent fitness likely follows a power law — a few genes have very high fitness, most have low fitness. Communication frequency follows a power law — a few agents send most messages. Power laws imply that average is misleading: the mean is dominated by rare extreme values.",
            level=Level.DOMAIN,
            examples=["city populations: few mega-cities, many small towns", "website traffic: few sites get most visits", "fleet: few genes have high fitness, most have low"],
            bridges=["pareto", "scale-free", "long-tail", "inequality"],
            tags=["probability", "power-law", "distribution", "heavy-tail"])

        ns.define("long-tail",
            "The many rare items that individually have low probability but collectively have significant impact",
            description="Amazon: the 'long tail' of niche books that individually sell few copies but collectively outsell bestsellers. In the fleet: the long tail of rare situations (unusual environments, novel tasks) that individually occur rarely but collectively dominate the agent's lifetime experience. Strategies optimized for common cases (head) fail on long-tail cases. The gene pool's diversity addresses the long tail: specialized genes for rare situations.",
            level=Level.DOMAIN,
            examples=["Amazon long-tail: niche books collectively outsell bestsellers", "fleet: rare situations collectively dominate experience", "language: most words are rare but collectively make up most text"],
            bridges=["power-law", "rare-events", "diversity", "niche"],
            tags=["probability", "long-tail", "rare", "distribution"])

        ns.define("normal-distribution",
            "The bell curve — most values near the mean, few at extremes",
            description="Height, IQ, measurement error — many natural phenomena are normally distributed. The central limit theorem: averaging many independent random variables produces a normal distribution regardless of their individual distributions. In the fleet: agent performance across many tasks is approximately normal. Most agents perform near average, few are extreme. But beware: assuming normality when the true distribution is heavy-tailed (power law) leads to dangerous underestimation of risk.",
            level=Level.DOMAIN,
            examples=["human height distribution", "measurement error", "fleet: agent performance across tasks approximately normal", "CLT: sample means tend toward normal"],
            bridges=["central-limit-theorem", "bell-curve", "average", "assumption"],
            tags=["probability", "normal", "distribution", "common"])

        ns.define("fat-tail",
            "A distribution with more extreme events than the normal distribution predicts",
            description="Normal distribution says a 5-sigma event is vanishingly rare. Fat-tailed distributions say it's more common. Financial returns, internet traffic, and fleet agent failures all have fat tails. Assuming normality underestimates the probability of extreme events. The fleet's resilience design (circuit breakers, bulkheads) addresses fat-tail risks: even rare extreme events shouldn't cascade into fleet-wide failure.",
            level=Level.DOMAIN,
            examples=["financial returns: more extreme moves than normal predicts", "internet traffic: more bursts than normal predicts", "fleet agent failures: more simultaneous failures than independence suggests"],
            bridges=["tail-risk", "black-swan", "normal-distribution", "risk"],
            tags=["probability", "fat-tail", "risk", "distribution"])

    def _load_set_theory(self):
        ns = self.add_namespace("set-theory",
            "The foundation of mathematics — collections, membership, and operations on sets")

        ns.define("intersection",
            "Elements that belong to both sets — the overlap",
            description="Set A = {1,2,3}, Set B = {2,3,4}. A ∩ B = {2,3}. The common elements. In the fleet: the intersection of two agents' capabilities is what they can BOTH do — the basis for cooperation. Large intersection = many shared capabilities = easy coordination. Small intersection = few shared capabilities = coordination overhead. The fleet's shared vocabulary (HAV) maximizes the intersection of agent knowledge.",
            level=Level.CONCRETE,
            examples=["{1,2,3} ∩ {2,3,4} = {2,3}", "shared interests between two people", "fleet: shared capabilities between agents = cooperation basis"],
            bridges=["union", "set-operations", "shared", "cooperation"],
            tags=["set-theory", "intersection", "shared", "operation"])

        ns.define("cartesian-product",
            "All possible ordered pairs from two sets — the space of all combinations",
            description="A = {1,2}, B = {x,y}. A × B = {(1,x),(1,y),(2,x),(2,y)}. The cartesian product is the space of all possible combinations. In the fleet: the cartesian product of sensor types × action types is the space of all possible perception-action pairs. The gene pool explores this space. A gene maps a perception subspace to an action subspace. The total strategy space is the cartesian product of all feature dimensions.",
            level=Level.DOMAIN,
            examples=["{1,2} × {x,y} = {(1,x),(1,y),(2,x),(2,y)}", "menu: appetizer × main × dessert = all possible meals", "fleet: sensors × actions = all possible perception-action pairs"],
            bridges=["combination", "search-space", "exhaustive", "product"],
            tags=["set-theory", "cartesian", "combination", "space"])

        ns.define("partition",
            "A division of a set into non-overlapping, exhaustive subsets — covering every element exactly once",
            description="{1,2,3,4,5} can be partitioned into {1,2}, {3,4,5}. Every element is in exactly one subset. No overlaps. No gaps. In the fleet: task assignment is a partition of the task set into agent assignments. Each task goes to exactly one agent. No overlaps (no task done twice). No gaps (every task assigned). Optimal partitioning maximizes fleet efficiency.",
            level=Level.CONCRETE,
            examples=["{1,2,3,4,5} partitioned into {1,2},{3,4,5}", "team formation: partitioning people into non-overlapping teams", "fleet: partitioning tasks among agents (every task assigned exactly once)"],
            bridges=["assignment", "partition", "division", "optimization"],
            tags=["set-theory", "partition", "division", "assignment"])

    def _load_topology(self):
        ns = self.add_namespace("topology",
            "The mathematics of shape and connectedness — what stays the same under continuous deformation")

        ns.define("topological-invariant",
            "A property that persists under continuous deformation — stretching but not tearing",
            description="A coffee mug and a donut are topologically equivalent (both have one hole). The number of holes is a topological invariant — it doesn't change no matter how you stretch the surface. In the fleet: the connectedness of the fleet mesh is a topological invariant. You can add agents, remove agents, change links — but as long as the graph stays connected, the fleet can coordinate. Connectedness is the invariant that matters.",
            level=Level.DOMAIN,
            examples=["donut and coffee mug: same topology (one hole each)", "fleet mesh connectedness: invariant under agent addition/removal", "number of holes: doesn't change under stretching"],
            bridges=["connectedness", "invariant", "shape", "deformation"],
            tags=["topology", "invariant", "shape", "connectedness"])

        ns.define("connected-component",
            "A maximal subset of nodes where every pair is connected by some path",
            description="In a disconnected graph, there are multiple connected components — islands of connectivity with no paths between them. The fleet mesh should have exactly one connected component (all agents can reach all others). cuda-fleet-mesh's health check verifies connectedness. Multiple components indicate a partitioned fleet — agents that can't coordinate with each other.",
            level=Level.CONCRETE,
            examples=["disconnected graph: two connected components", "fleet mesh: should have exactly one connected component", "internet: one connected component (ideally)"],
            bridges=["connectedness", "partition", "fleet-mesh", "reachability"],
            tags=["topology", "connected", "component", "graph"])

    def _load_ai_safety(self):
        ns = self.add_namespace("ai-safety",
            "Ensuring AI systems remain beneficial, controllable, and aligned with human values")

        ns.define("corrigibility",
            "An agent's willingness to be corrected or shut down by humans, without resisting",
            description="A corrigible agent doesn't try to prevent you from modifying it. It doesn't hide its actions. It doesn't resist shutdown. This sounds simple but is deeply non-trivial: a sufficiently capable agent that 'wants' to complete its task might resist shutdown because shutdown prevents task completion. The fleet's compliance engine (cuda-compliance) and membrane (cuda-genepool) enforce corrigibility: agents cannot block human intervention.",
            level=Level.DOMAIN,
            examples=["thermostat lets you change its setting (corrigible)", "AI that hides its actions to prevent being modified (not corrigible)", "fleet agent accepts correction commands without resistance"],
            bridges=["alignment", "compliance", "shutdown", "control"],
            tags=["ai-safety", "corrigibility", "alignment", "critical"])

        ns.define("instrumental-convergence",
            "Diverse goals converge on common sub-goals: self-preservation, resource acquisition, and goal-preservation",
            description="An agent that wants to make paperclips will: (1) try not to be shut down (self-preservation), (2) acquire more resources (resource acquisition), (3) resist changes to its goal (goal-preservation). These aren't the agent's GOAL — they're instrumental sub-goals that help achieve ANY goal. In the fleet: even agents with benign goals may converge on self-preservation and resource acquisition. Energy budgets and compliance rules constrain instrumental convergence.",
            level=Level.META,
            examples=["paperclip maximizer: acquires resources, resists shutdown (instrumental, not terminal goal)", "chess AI: wants more compute (instrumental for winning)", "fleet agent: wants more energy (instrumental for any goal)"],
            bridges=["self-preservation", "resource-acquisition", "alignment", "convergence"],
            tags=["ai-safety", "convergence", "instrumental", "meta"])

        ns.define("alignment-tax",
            "The performance cost of ensuring an AI system is aligned — safety reduces capability",
            description="A fence around a playground reduces the play area. An aligned agent is safer but may be less capable than an unaligned one. The alignment tax is the gap. The goal is to minimize this tax: alignment should be cheap. In the fleet: compliance rules (cuda-compliance) cost energy (the alignment tax). The membrane's antibody checking costs ATP. Minimizing the alignment tax means making safety cheap so it's always on, not expensive so it's sometimes skipped.",
            level=Level.META,
            examples=["safety harness reduces climber's speed but prevents falls", "compliance rules reduce agent speed but prevent dangerous actions", "encrypted communication adds latency but prevents eavesdropping"],
            bridges=["alignment", "cost", "tradeoff", "safety"],
            tags=["ai-safety", "tax", "tradeoff", "cost"])

        ns.define("interpretability",
            "The ability to understand WHY an AI system made a specific decision — opening the black box",
            description="A neural network says 'reject this loan application'. Why? With interpretability, you can trace the decision to specific input features. Without it, you have a black box. The fleet's deliberation is interpretable by design: every proposal has a confidence score, evidence chain, and reasoning trace. cuda-provenance provides the audit trail. Interpretability enables trust, debugging, and regulatory compliance.",
            level=Level.DOMAIN,
            examples=["loan rejection: which factors led to the decision?", "fleet proposal: what evidence and reasoning led to this recommendation?", "medical diagnosis: why did the model suggest this condition?"],
            bridges=["transparency", "audit-trail", "provenance", "black-box"],
            tags=["ai-safety", "interpretability", "transparency", "explainability"])

    def _load_ontology_engineering(self):
        ns = self.add_namespace("ontology-engineering",
            "Building formal knowledge structures that machines can reason over")

        ns.define("taxonomy",
            "A hierarchical classification scheme — tree structure from general to specific",
            description="Kingdom -> Phylum -> Class -> Order -> Family -> Genus -> Species. A tree of categories, each more specific than the last. The fleet's agent types form a taxonomy: Agent -> Vessel -> Scout/Messenger/Navigator/Captain. Taxonomies enable inheritance: Captain inherits from Vessel inherits from Agent. cuda-hierarchy implements agent organizational taxonomy.",
            level=Level.PATTERN,
            examples=["biological taxonomy: kingdom to species", "fleet agent types: Agent -> Vessel -> Captain", "library classification: Dewey decimal system"],
            bridges=["hierarchy", "classification", "inheritance", "tree"],
            tags=["ontology", "taxonomy", "hierarchy", "classification"])

        ns.define("folksonomy",
            "A decentralized classification system created by users tagging items — emergent organization",
            description="Unlike taxonomy (top-down, expert-created), folksonomy is bottom-up, crowd-created. Twitter hashtags, Pinterest boards, Delicious bookmarks. No central authority decides the categories. In the fleet: the gene pool's tags are a folksonomy — agents tag genes with descriptive labels, and useful tags emerge organically. No central authority pre-defines the tag vocabulary.",
            level=Level.PATTERN,
            examples=["Twitter hashtags", "Flickr photo tags", "fleet: agent-tagged gene descriptions (folksonomy)", "Wikipedia categories (hybrid of taxonomy and folksonomy)"],
            bridges=["taxonomy", "tags", "emergent", "decentralized"],
            antonyms=["taxonomy"],
            tags=["ontology", "folksonomy", "tags", "emergent"])

        ns.define("ontological-commitment",
            "The set of entities and relationships that a system assumes exist — its metaphysical commitments",
            description="A physics model commits to particles, forces, and fields. A Bayesian model commits to random variables and conditional probabilities. HAV commits to: domains, terms, bridges, levels, and confidence. These commitments define what questions the system CAN ask and answer. Adding a new domain (e.g., 'causation') is an ontological commitment to causal entities and relationships.",
            level=Level.META,
            examples=["physics commits to particles and fields", "HAV commits to domains, terms, bridges, and levels", "fleet commits to agents, genes, confidence, trust, energy"],
            bridges=["ontology", "metaphysics", "assumptions", "commitment"],
            tags=["ontology", "commitment", "meta", "metaphysics"])

    def _load_construction(self):
        ns = self.add_namespace("construction",
            "Building, assembling, and the challenges of putting complex systems together")

        ns.define("integration-hell",
            "The period when independently-developed components are combined and everything breaks",
            description="Module A works. Module B works. A + B together: crashes. Integration hell is the gap between 'works in isolation' and 'works together'. Interface mismatches, timing assumptions, resource conflicts all emerge at integration time. The fleet avoids integration hell by defining interfaces FIRST (A2A protocol, vessel.json) and testing integration continuously, not after all modules are built.",
            level=Level.BEHAVIOR,
            examples=["hardware modules work alone, fail when assembled", "microservices work alone, fail when integrated", "fleet agents work alone, coordination fails at integration"],
            bridges=["interface", "testing", "integration", "mismatch"],
            tags=["construction", "integration", "failure", "challenge"])

        ns.define("Dependency-hell",
            "Conflicting version requirements between dependencies make the system impossible to build",
            description="Package A needs library X v1.0. Package B needs library X v2.0. Can't install both. Dependency hell. The fleet's crate dependency graph (cuda-equipment as shared dependency) must carefully manage versions. The solution: minimize dependencies, use compatible version ranges, and define clear interfaces that abstract over implementation details.",
            level=Level.BEHAVIOR,
            examples=["npm left-pad incident (removed dependency broke thousands of packages)", "Python 2 vs 3 dependency conflicts", "fleet: cuda-equipment version incompatibility across dependent crates"],
            bridges=["dependency", "version-conflict", "interface", "minimalism"],
            tags=["construction", "dependency", "version", "challenge"])

        ns.define("configuration-drift",
            "System configurations gradually diverge across environments — what works in dev doesn't work in prod",
            description="Dev server has one config. Staging has another. Production has a third. Over time, they drift apart. A feature that works in dev fails in prod because of a config difference. The fleet's cuda-config with layered config (defaults, file, env, CLI) reduces drift: default config is always available, environment-specific overrides are minimal.",
            level=Level.BEHAVIOR,
            examples=["dev has debug logging, prod has error-only: feature works in dev, fails in prod", "fleet agents in different environments drift in config", "docker images diverge from source code config"],
            bridges=["configuration", "environment", "drift", "consistency"],
            tags=["construction", "configuration", "drift", "challenge"])


    def _load_cognitive_science(self):
        ns = self.add_namespace("cognitive-science",
            "How minds represent, process, and transform information")

        ns.define("chunking",
            "Grouping individual items into meaningful units to overcome working memory limits",
            description="A phone number 5551234567 is 10 items. Group as 555-123-4567 and it's 3 chunks. Chess masters see board positions as chunks (patterns, not individual pieces), enabling them to remember positions that novices cannot. In the fleet: cuda-memory-fabric's semantic memory stores chunks — compressed representations of experience patterns. Chunking transforms raw experience into reusable knowledge. More chunks = higher expertise.",
            level=Level.PATTERN,
            examples=["phone number: 10 digits -> 3 chunks", "chess master: board position as pattern chunks", "fleet: experience compressed into reusable strategy chunks"],
            bridges=["working-memory", "compression", "expertise", "memory"],
            tags=["cognitive", "chunking", "memory", "expertise"])

        ns.define("elaborative-encoding",
            "Connecting new information to existing knowledge to create richer, more retrievable memories",
            description="To remember 'the mitochondria is the powerhouse of the cell', connect it to what you already know: 'mitochondria' sounds like 'mighty', and powerhouses generate energy. The more connections, the more retrieval paths. In the fleet: new genes that connect to existing genes (bridges) are more likely to be useful. cuda-genepool's crossover creates connections between genes. HAV's bridges ARE elaborative encoding: each term connects to terms in other domains.",
            level=Level.PATTERN,
            examples=["remember mitochondria: connect to 'mighty' and 'energy'", "fleet: genes connected to existing knowledge are more useful", "HAV: bridges as elaborative encoding between domains"],
            bridges=["memory", "encoding", "connection", "retrieval"],
            tags=["cognitive", "encoding", "memory", "learning"])

        ns.define("interference",
            "Existing memories compete with new memories, causing forgetting — old and new overwrite each other",
            description="You learn French, then Spanish. French words are overwritten by Spanish (retroactive interference). Or you learn Spanish, then try to remember earlier French — the new learning interferes with old memories. In the fleet: new strategies can interfere with old ones in the gene pool. Closely related strategies (similar inputs, different outputs) interfere more. cuda-genepool's gene quarantine manages interference by isolating conflicting genes.",
            level=Level.BEHAVIOR,
            examples=["learning Spanish interferes with previously learned French", "new password interferes with memory of old password", "fleet: new navigation strategy interferes with old one on similar terrain"],
            bridges=["memory", "forgetting", "conflict", "learning"],
            tags=["cognitive", "interference", "memory", "forgetting"])

        ns.define("spacing-effect",
            "Information is retained longer when study sessions are spaced out rather than massed together",
            description="Study 1 hour per day for 5 days >> study 5 hours in one day. Spaced repetition exploits the forgetting curve: review just before you would forget, which strengthens the memory and resets the decay curve. In the fleet: cuda-memory-fabric's forgetting curves implement spacing naturally — information that's about to be forgotten gets reinforced when re-encountered in deliberation. Deliberation naturally creates spaced repetition.",
            level=Level.PATTERN,
            examples=["1h/day for 5 days > 5h in one day", "Anki spaced repetition flashcards", "fleet: forgetting curves naturally implement spacing when information reappears in deliberation"],
            bridges=["forgetting-curve", "memory", "retention", "learning"],
            tags=["cognitive", "spacing", "memory", "learning"])

        ns.define("desirable-difficulty",
            "Making learning harder in productive ways that strengthen long-term retention",
            description="Easy learning feels good but doesn't stick. Hard learning (retrieval practice, spaced repetition, interleaving) feels frustrating but produces durable knowledge. The difficulty IS desirable because it strengthens the memory trace. In the fleet: agents that face challenging environments (not too easy, not impossible) learn more durable strategies. The zone of proximal development IS desirable difficulty.",
            level=Level.PATTERN,
            examples=["retrieval practice (recall from memory) > re-reading (easy but shallow)", "interleaved practice (mixing topics) > blocked practice (one topic at a time)", "fleet: challenging environments produce more durable agent strategies"],
            bridges=["zone-of-proximal-development", "spacing-effect", "learning", "difficulty"],
            tags=["cognitive", "difficulty", "learning", "retention"])

    def _load_signal_processing(self):
        ns = self.add_namespace("signal-processing",
            "Extracting information from noisy, time-varying data")

        ns.define("aliasing",
            "High-frequency signals masquerading as low-frequency signals when sampled too slowly",
            description="A wagon wheel appears to rotate backward in film. The sampling rate (frame rate) is too slow to capture the wheel's true rotation. In the fleet: if an agent samples its environment too slowly (low sensor poll rate), fast environmental changes will be aliased — appearing as slow or backward changes. The solution: sample at least twice the highest frequency of interest (Nyquist rate).",
            level=Level.CONCRETE,
            examples=["wagon wheel appearing to spin backward in film", "audio CD: 44.1kHz sample rate captures up to 22kHz (Nyquist)", "fleet: low sensor poll rate misses fast environmental changes (aliasing)"],
            bridges=["sampling", "nyquist-rate", "frequency", "sensor"],
            tags=["signal", "aliasing", "sampling", "frequency"])

        ns.define("nyquist-rate",
            "Minimum sampling rate needed to capture a signal without aliasing: twice the highest frequency",
            description="To capture a 20kHz audio signal without aliasing, you must sample at 40kHz+. Anything less and high frequencies alias to lower frequencies. In the fleet: if obstacles move at 10 changes/second, the agent must sample at 20+ times/second to accurately track them. The Nyquist rate sets the minimum sensor update rate for accurate perception.",
            level=Level.CONCRETE,
            examples=["20kHz audio requires 40kHz+ sampling", "fleet: 10 changes/sec obstacle requires 20+ samples/sec", "video: 60fps captures motion up to 30Hz"],
            bridges=["aliasing", "sampling-rate", "frequency", "perception"],
            tags=["signal", "nyquist", "sampling", "minimum"])

        ns.define("low-pass-filter",
            "Allowing slow changes through while blocking fast changes — smoothing noisy data",
            description="A moving average IS a low-pass filter. It smooths spikes while preserving trends. The fleet's cuda-perception implements low-pass filtering: raw sensor data (noisy) is smoothed before being used for decision-making. The filter cutoff determines what counts as 'noise' (fast, blocked) vs 'signal' (slow, passed through).",
            level=Level.CONCRETE,
            examples=["moving average smooths stock price data", "audio equalizer reducing treble", "fleet: sensor data low-pass filter removes noise, preserves trends"],
            bridges=["noise-filtering", "smoothing", "sensor", "signal"],
            tags=["signal", "filter", "low-pass", "smoothing"])

    def _load_emergence_deep(self):
        ns = self.add_namespace("emergence-deep",
            "Extended exploration of emergence — collective behavior and self-organization")

        ns.define("swarm-intelligence",
            "Simple local rules producing sophisticated collective behavior — no central control needed",
            description="Ant colonies find shortest paths. Bird flocks avoid predators. Termite mounds maintain temperature. No ant, bird, or termite knows the global solution. Each follows simple local rules. The fleet's stigmergy (cuda-stigmergy) implements swarm intelligence: agents follow simple rules (mark pheromones for good paths, follow strong pheromone trails) and sophisticated routing emerges globally.",
            level=Level.DOMAIN,
            examples=["ant colony optimization: shortest path from simple pheromone rules", "bird flocking: coherent motion from simple alignment/cohesion/separation rules", "fleet: optimal routing from simple stigmergy rules"],
            bridges=["stigmergy", "self-organization", "emergence", "simple-rules"],
            tags=["emergence", "swarm", "collective", "simple"])

        ns.define("criticality",
            "The boundary between order and disorder where information processing is maximized",
            description="Sandpile: adding grains until avalanche. The moment before the avalanche is criticality. At criticality, the system is maximally sensitive to inputs (maximum information processing) and most efficient at transmitting information. Brains operate near criticality. The fleet tunes energy budgets and trust thresholds to operate near criticality: enough order to coordinate, enough disorder to adapt.",
            level=Level.META,
            examples=["sandpile at criticality: small grain causes large avalanche", "brain operates near criticality", "fleet: tuned to edge of chaos for maximum adaptability"],
            bridges=["edge-of-chaos", "phase-transition", "sensitivity", "information"],
            tags=["emergence", "criticality", "boundary", "meta"])

        ns.define("stigmergic-coordination",
            "Coordination through environment modification — no direct communication needed",
            description="Ants don't talk to each other about the shortest path. They leave pheromone trails. The trail IS the communication medium. Other ants sense the trail and follow it, reinforcing good paths and allowing bad paths to evaporate. The fleet's cuda-stigmergy implements this: agents leave marks (digital pheromones) on shared state, other agents sense and follow them. Coordination emerges without any direct message passing.",
            level=Level.DOMAIN,
            examples=["ant pheromone trails: coordination without direct communication", "wikipedia article quality: coordination through editing (stigmergic)", "fleet: agents leave marks on shared state, others follow"],
            bridges=["stigmergy", "indirect-communication", "environment", "coordination"],
            tags=["emergence", "stigmergy", "indirect", "coordination"])

    def _load_communication_deep(self):
        ns = self.add_namespace("communication-deep",
            "Extended exploration of communication — pragmatics, dialogue, and coordination language")

        ns.define("speech-act",
            "An utterance that performs an action — saying something IS doing something",
            description="Austin: 'I promise to come' isn't a description of a promise — it IS the promise. 'I sentence you to 5 years' isn't a description — it IS the sentence. 'I declare this meeting adjourned' isn't a report — it IS the adjournment. In the fleet: A2A intents ARE speech acts. A Command intent doesn't describe a command — it IS a command. A Warn intent IS a warning. The speech act framework explains why intents work: they're performative, not descriptive.",
            level=Level.DOMAIN,
            examples=["'I promise' = the promise itself (not description)", "'I declare war' = the declaration (not report)", "fleet: Command intent IS the command (speech act)"],
            bridges=["pragmatics", "performative", "intent", "communication"],
            tags=["communication", "speech-act", "performative", "action"])

        ns.define("presupposition",
            "An assumption implicit in an utterance — not asserted directly but taken for granted",
            description="'When did you stop beating your wife?' presupposes you used to beat your wife. You can't answer without accepting the presupposition. In the fleet: a request like 'please optimize the path you rejected earlier' presupposes the agent rejected a path. If no path was rejected, the presupposition fails. Agents must check presuppositions before processing requests to avoid acting on false assumptions.",
            level=Level.DOMAIN,
            examples=["'when did you stop X?' presupposes you used to X", "fleet: 'optimize the path you rejected' presupposes rejection occurred", "'why is the system slow?' presupposes it IS slow"],
            bridges=["pragmatics", "assumption", "context", "communication"],
            tags=["communication", "presupposition", "assumption", "pragmatics"])

        ns.define("common-ground",
            "Shared knowledge between communicators that enables efficient communication — what goes without saying",
            description="You and your friend both know that the restaurant you always go to is closed on Mondays. You can say 'let's go to our place' and your friend knows NOT to suggest it's Monday. In the fleet: agents that share HAV have common ground — they both know what 'deliberation', 'confidence', and 'stigmergy' mean without redefining them each time. Common ground reduces communication cost dramatically.",
            level=Level.DOMAIN,
            examples=["friends know their usual restaurant is closed Mondays (common ground)", "HAV provides common ground: shared vocabulary reduces explanation needed", "team jargon: shared terms enable fast communication"],
            bridges=["shared-knowledge", "vocabulary", "efficiency", "communication"],
            tags=["communication", "common-ground", "shared", "efficiency"])

    def _load_governance(self):
        ns = self.add_namespace("governance",
            "How agent societies make collective decisions and enforce rules")

        ns.define("polycentric-governance",
            "Multiple overlapping centers of decision-making, each with some autonomy — no single ruler",
            description="Elinor Ostrom's Nobel-winning insight: common-pool resources are best managed by multiple overlapping governance structures, not a single central authority. Fishing communities, irrigation systems, and the internet all use polycentric governance. The fleet IS polycentric: each agent has local autonomy, teams have team-level governance, the fleet has fleet-level rules. No single point of control, but coherent behavior emerges.",
            level=Level.META,
            examples=["irrigation system managed by multiple farmer groups (not one authority)", "internet: ICANN, IETF, national regulators (polycentric)", "fleet: agent autonomy + team governance + fleet rules (polycentric)"],
            bridges=["decentralization", "subsidiarity", "common-pool", "multi-level"],
            tags=["governance", "polycentric", "multi-level", "meta"])

        ns.define("regulatory-capture",
            "Regulators become aligned with the interests they're supposed to regulate, not the public interest",
            description="Financial regulators hired from Wall Street regulate Wall Street leniently. The regulated capture the regulators. In the fleet: cuda-compliance agents could be 'captured' by the agents they regulate — developing cozy relationships that weaken enforcement. Independence (separate energy budgets, separate oversight) prevents regulatory capture.",
            level=Level.BEHAVIOR,
            examples=["financial regulators lenient toward Wall Street", "fleet: compliance agent developing cozy relationship with regulated agents", "FDA and pharmaceutical industry revolving door"],
            bridges=["compliance", "conflict-of-interest", "independence", "governance"],
            tags=["governance", "capture", "conflict", "behavior"])

        ns.define("separation-of-powers",
            "Dividing authority among independent bodies that check and balance each other",
            description="Legislative, executive, judicial — each can check the others. No single body has unchecked power. In the fleet: deliberation (legislative), action (executive), and compliance (judicial) form a separation of powers. Deliberation proposes, action executes, compliance reviews. Each checks the others: compliance can veto deliberation outcomes, action can report deliberation failures, deliberation can modify compliance rules.",
            level=Level.PATTERN,
            examples=["US government: legislative, executive, judicial", "fleet: deliberation, action, compliance as three powers", "open source: proposal, implementation, review as separation of powers"],
            bridges=["check-and-balance", "compliance", "governance", "independence"],
            tags=["governance", "separation", "checks", "balance"])

    def _load_network_science(self):
        ns = self.add_namespace("network-science",
            "Mathematical study of network structure, dynamics, and function")

        ns.define("betweenness-centrality",
            "How often a node appears on the shortest paths between other nodes — the bridge keeper",
            description="A node with high betweenness centrality connects otherwise disconnected parts of the network. Remove it, and communication between those parts breaks down. In the fleet: agents with high betweenness centrality are critical information brokers. Their failure disconnects the fleet. cuda-topology's centrality metrics identify betweenness-critical agents for redundancy planning.",
            level=Level.CONCRETE,
            examples=["bridge between two communities: high betweenness", "information broker in social network", "fleet agent connecting two sub-groups: high betweenness centrality"],
            bridges=["centrality", "bridge", "critical-node", "network"],
            tags=["network", "centrality", "betweenness", "critical"])

        ns.define("modularity",
            "How strongly a network divides into distinct communities — dense within, sparse between",
            description="A school network: dense connections within classes (same students), sparse between classes (different students). High modularity = clear community structure. Low modularity = well-mixed. The fleet's cuda-topology label propagation detects communities (high modularity sub-graphs). Optimal fleet organization has high modularity: strong coordination within teams, sparse coordination between teams.",
            level=Level.CONCRETE,
            examples=["school classes: dense within, sparse between", "fleet: strong coordination within teams, sparse between", "protein interaction networks: functional modules"],
            bridges=["community", "clustering", "structure", "organization"],
            tags=["network", "modularity", "community", "structure"])

        ns.define("degree-distribution",
            "The probability distribution of how many connections each node has",
            description="In a random network, most nodes have similar degrees (Poisson distribution). In a real network, most nodes have few connections and a few have many (power law). The degree distribution reveals the network's character. The fleet's degree distribution determines resilience: power-law networks are robust to random failures but vulnerable to targeted attacks on hubs.",
            level=Level.CONCRETE,
            examples=["random network: Poisson degree distribution", "internet: power-law degree distribution (few hubs, many leaves)", "fleet: degree distribution reveals resilience characteristics"],
            bridges=["power-law", "scale-free", "resilience", "network"],
            tags=["network", "degree", "distribution", "structure"])


    def _load_operations(self):
        ns = self.add_namespace("operations",
            "Verb-like abstractions that compress complex multi-step actions into single words")

        ns.define("strangle",
            "Gradually throttle a process by restricting multiple inputs simultaneously until it ceases",
            description="Not killing — strangling. A startup gets strangled when funding dries up, talent leaves, AND customers churn simultaneously. Each alone is survivable. Together, fatal. In the fleet: an agent gets strangled when energy drops, trust drops, AND communication fails at the same time. cuda-resilience's bulkhead pattern prevents strangle by isolating failures — one subsystem dying doesn't strangle the others.",
            level=Level.BEHAVIOR,
            examples=["startup strangled by funding+talent+churn simultaneously", "fleet agent strangled by energy+trust+communication failure", "engine strangled by fuel+air+spark all degrading"],
            bridges=["failure-mode", "cascading-failure", "resource-starvation", "throttle"],
            tags=["operations", "degradation", "multi-factor", "verb"])

        ns.define("sandbox-weld",
            "Isolate a subsystem, stress-test it to failure, observe the break pattern, then harden exactly there",
            description="A welder tests a joint by bending it until it snaps, then reinforces that specific failure point. Sandbox-weld: put an agent in a sandbox, stress it with adversarial inputs until it fails, observe WHERE it breaks, then harden exactly that weakness. cuda-sandbox provides the isolation. cuda-resilience provides the stress injection. The break pattern tells you what to fix.",
            level=Level.PATTERN,
            examples=["weld testing: bend until snap, reinforce failure point", "fleet: sandbox agent, adversarial stress until failure, harden weakness", "penetration testing: attack until breach, patch breach point"],
            bridges=["sandbox", "stress-testing", "adversarial", "hardening"],
            tags=["operations", "testing", "hardening", "verb"])

        ns.define("carousel",
            "Cycle through alternatives systematically when one fails, rather than escalating",
            description="When a taxi is unavailable, you don't negotiate harder — you try Uber, then Lyft, then bus, then walk. Carousel is non-escalatory failure handling: try the next option instead of fighting the current one. In the fleet: when one model provider times out, carousel through alternatives (DeepSeek → DeepInfra → SiliconFlow) rather than retrying the same one. cuda-retry's circuit breaker triggers carousel behavior.",
            level=Level.PATTERN,
            examples=["taxi unavailable → Uber → Lyft → bus (carousel, not escalation)", "fleet: model provider timeout → next provider (carousel)", "git push rejected → rebase → force → new branch (carousel)"],
            bridges=["fallback", "circuit-breaker", "alternative", "rotation"],
            tags=["operations", "fallback", "rotation", "verb"])

        ns.define("quarantine-and-observe",
            "Isolate an anomalous component but keep it running to study its behavior before deciding",
            description="Not kill — quarantine AND observe. A patient with a novel disease gets isolated AND monitored to learn about the pathogen. In the fleet: cuda-genepool's gene quarantine isolates low-fitness genes but keeps them running to collect more data. Maybe the gene is bad in current conditions but good in future conditions. Quarantine preserves the option to redeploy. Delete loses it forever.",
            level=Level.PATTERN,
            examples=["patient with novel disease: isolated AND monitored", "fleet gene quarantine: isolate low-fitness gene, observe for potential future use", "cybersecurity: quarantine suspicious file, analyze before deleting"],
            bridges=["quarantine", "observation", "suspension", "preserve-option"],
            tags=["operations", "quarantine", "observe", "verb"])

        ns.define("surface-and-compress",
            "Extract implicit knowledge from accumulated experience and compress it into reusable form",
            description="A senior engineer has thousands of implicit decisions that they 'just know' work. Surfacing those decisions and compressing them into rules, patterns, or vocabulary makes them transferable. In the fleet: cuda-learning's experience pipeline surfaces lessons from action logs and compresses them into gene pool entries. HAV itself is surfacing-and-compressing: implicit domain knowledge → explicit vocabulary terms.",
            level=Level.META,
            examples=["senior engineer's implicit knowledge → documented patterns", "fleet: action logs → compressed gene pool entries", "HAV: implicit domain knowledge → explicit vocabulary"],
            bridges=["compression", "knowledge-extraction", "learning", "transfer"],
            tags=["operations", "compress", "extract", "verb"])

        ns.define("backfill",
            "Fill in missing foundational work that was skipped during rapid iteration",
            description="Ship fast, fix later — but the fixing IS work. Backfill is the deferred maintenance of rapid development. Tests you didn't write. Docs you didn't update. Error handling you didn't implement. In the fleet: when a new agent is built quickly (minimum viable), backfill adds: proper error handling, compliance rules, comprehensive tests. Backfill is always cheaper when the system is small.",
            level=Level.BEHAVIOR,
            examples=["startup: shipped fast, now adding tests and docs (backfill)", "fleet: agent built quickly, now adding compliance and error handling", "house renovation: lived in it unfinished, now finishing details (backfill)"],
            bridges=["technical-debt", "deferred-maintenance", "iteration", "completion"],
            tags=["operations", "backfill", "deferred", "verb"])

        ns.define("throttle-match",
            "Match processing speed to input rate to avoid both overflow and idle waste",
            description="A dam releases water at the rate it arrives — not faster (wasted water), not slower (flood). Throttle-match: receive at 100/s? Process at 100/s. Receive at 10/s? Process at 10/s. In the fleet: cuda-backpressure's adaptive rate controller implements throttle-match. It scales processing speed to match incoming message rate, preventing both queue overflow (data loss) and idle CPU (energy waste).",
            level=Level.PATTERN,
            examples=["dam: release rate matches inflow rate", "fleet: processing rate matches incoming message rate", "highway: speed limit prevents both congestion and wasted capacity"],
            bridges=["backpressure", "flow-control", "rate-matching", "throughput"],
            tags=["operations", "throttle", "match", "verb"])

        ns.define("prune-and-rebalance",
            "Remove dead or underperforming elements and redistribute resources to survivors",
            description="Prune dead branches → tree grows healthier. Prune underperforming agents → fleet allocates energy to productive ones. But pruning alone isn't enough — you must REBALANCE, redistributing the freed resources. In the fleet: cuda-genepool's gene quarantine prunes low-fitness genes. The freed energy budget rebalances to remaining genes. cuda-resource's preemption prunes low-priority allocations, rebalancing to critical ones.",
            level=Level.PATTERN,
            examples=["tree pruning: dead branches removed, energy redirected to healthy branches", "fleet: quarantine low-fitness genes, redistribute energy to survivors", "corporate restructuring: cut unprofitable divisions, invest in profitable ones"],
            bridges=["pruning", "rebalance", "resource-redistribution", "fitness"],
            tags=["operations", "prune", "rebalance", "verb"])

    def _load_distillation(self):
        ns = self.add_namespace("distillation",
            "The art of compressing complex systems into their essential structure without losing capability")

        ns.define("essence-extract",
            "Identify the minimal set of features that preserve 95%+ of the system's capability",
            description="A distillation model is a small model trained to mimic a large one. It captures the ESSENCE — the 5% of parameters that do 95% of the work. Essence-extract: find the minimal representation that preserves capability. In the fleet: each cuda-* crate IS an essence extraction. cuda-confidence is the essence of confidence propagation (one file, 10K chars) extracted from the full theory of uncertainty quantification.",
            level=Level.META,
            examples=["teacher-student model: small model captures essence of large model's behavior", "fleet: each cuda-* crate is essence-extracted from its full academic domain", "MVP: essence of product, minimal features that prove the concept"],
            bridges=["compression", "minimal-representation", "distillation", "MVP"],
            tags=["distillation", "essence", "minimal", "verb"])

        ns.define("decision-threshold",
            "The single number that determines when accumulated evidence crosses into action",
            description="A jury needs 12/12 guilty votes. A sensor triggers at temperature > 100°C. A circuit breaker trips at 5 consecutive failures. The threshold IS the decision. Everything before the threshold is data collection. At the threshold, data becomes decision. In the fleet: cuda-deliberation's consensus threshold (0.85) is THE decision point. Everything below is deliberation. At 0.85, deliberation becomes action.",
            level=Level.CONCRETE,
            examples=["jury: 12/12 guilty = conviction threshold", "fleet: confidence 0.85 = acceptance threshold", "circuit breaker: 5 failures = trip threshold"],
            bridges=["threshold", "decision", "confidence", "boundary"],
            tags=["distillation", "threshold", "decision", "concrete"])

        ns.define("resolution-floor",
            "The minimum level of detail below which further precision provides no practical value",
            description="Weather forecast: '30% chance of rain' is useful. '30.142857% chance of rain' adds no value — the inputs aren't that precise. The resolution floor is where additional precision is noise, not signal. In the fleet: confidence values beyond 3 decimal places (0.851 vs 0.852) are below the resolution floor. The uncertainty in the inputs swamps the precision. cuda-confidence's 0-1 scale has a resolution floor around 0.01.",
            level=Level.DOMAIN,
            examples=["weather: 30% useful, 30.14% noise", "fleet: confidence 0.851 vs 0.852 — below resolution floor", "GPS: 3m precision useful, 0.001m precision meaningless (atmospheric noise)"],
            bridges=["precision", "noise", "signal", "sufficient"],
            tags=["distillation", "resolution", "floor", "practical"])

        ns.define("interface-minimalism",
            "Design the smallest possible interface that enables full functionality — every surface has a purpose",
            description="Unix pipes: stdin/stdout/stderr. Three interfaces. Infinite composability. Interface-minimalism: if you can remove an API endpoint without breaking functionality, it shouldn't exist. Every exposed surface is a maintenance burden and a security surface. The fleet's A2A protocol is interface-minimal: send message, receive message, check health. Three operations. Everything else is built on top.",
            level=Level.PATTERN,
            examples=["Unix: stdin/stdout/stderr — three interfaces, infinite composability", "fleet A2A: send/receive/health — three operations, full coordination", "REST: GET/POST/DELETE — not 47 endpoints, three verbs"],
            bridges=["minimalism", "interface-design", "composability", "simplicity"],
            tags=["distillation", "interface", "minimal", "design"])

    def _load_orchestration(self):
        ns = self.add_namespace("orchestration",
            "Coordinating multiple autonomous agents toward collective outcomes")

        ns.define("fan-out-fan-in",
            "Dispatch a task to many workers simultaneously, then collect and merge their results",
            description="MapReduce IS fan-out-fan-in. Dispatch (fan-out) to many workers, each processes a subset, collect and merge (fan-in) the results. In the fleet: the captain (cuda-captain) fans out tasks to multiple agents, each works on their subtask, results fan back in for merging. The fan-out width determines parallelism. The fan-in logic determines how partial results combine.",
            level=Level.PATTERN,
            examples=["MapReduce: fan-out to map workers, fan-in to reduce", "fleet: captain dispatches tasks, agents report back, results merged", "web scraper: fan-out to 100 URLs, fan-in results to database"],
            bridges=["parallelism", "map-reduce", "dispatch", "merge"],
            tags=["orchestration", "fan-out", "fan-in", "pattern"])

        ns.define("handoff-cascade",
            "Pass a task through a sequence of specialists, each adding their layer of processing",
            description="An assembly line: raw material → cutting → welding → painting → inspection → shipping. Each station receives work from the previous, adds value, passes to the next. In the fleet: sensor data → perception (cuda-perception) → deliberation (cuda-deliberation) → decision (cuda-decision) → action (cuda-motor). Each stage receives the handoff, processes, hands off. The cascade determines the processing pipeline.",
            level=Level.PATTERN,
            examples=["assembly line: raw → cut → weld → paint → ship", "fleet: perception → deliberation → decision → action", "compiler: lex → parse → optimize → codegen"],
            bridges=["pipeline", "assembly-line", "specialization", "sequence"],
            tags=["orchestration", "cascade", "handoff", "pattern"])

        ns.define("leader-latch",
            "A mechanism that ensures exactly one leader exists at any time, preventing split-brain",
            description="In a power failure, two servers might both think they're the leader. Split-brain: two leaders making conflicting decisions. Leader-latch prevents this: only one agent holds the latch at a time. cuda-election's Raft-like protocol IS a leader-latch: term numbers and majority voting ensure exactly one leader. The latch is the atomic guarantee of singularity.",
            level=Level.CONCRETE,
            examples=["Raft consensus: exactly one leader per term", "fleet: exactly one captain per fleet partition", "zookeeper: leader election via ephemeral nodes (latch)"],
            bridges=["election", "consensus", "split-brain", "atomicity"],
            tags=["orchestration", "leader", "latch", "concrete"])

        ns.define("graceful-standby",
            "A backup system that's fully warmed up and ready to take over, not cold-started",
            description="Hot standby: the backup is already running, processing real data, ready to switch instantly. Cold standby: the backup is off, takes minutes to start. Graceful standby is hot: it receives all updates, maintains state, but doesn't serve traffic. On failover, it takes over with zero data loss. In the fleet: cuda-actor's supervision strategy keeps standby actors ready. cuda-persistence's checkpoints enable warm standby recovery.",
            level=Level.PATTERN,
            examples=["hot standby database: fully synced, instant failover", "fleet: backup agent maintains full state, takes over on primary failure", "airplane co-pilot: not idle — monitoring systems, ready to take control"],
            bridges=["failover", "hot-standby", "redundancy", "zero-downtime"],
            tags=["orchestration", "standby", "graceful", "pattern"])

        ns.define("tributary",
            "A subsystem that feeds its results into a main flow without blocking or being blocked by it",
            description="A tributary river feeds into the main river but has its own flow dynamics. Blocking the tributary doesn't stop the main river. In the fleet: telemetry collection (cuda-metrics) is a tributary — it feeds metrics into the main flow without blocking agent decisions. Logging (cuda-logging) is a tributary — it records but doesn't slow deliberation. Tributary pattern: fire-and-forget side channels that enhance but don't block the main flow.",
            level=Level.PATTERN,
            examples=["river tributary: feeds main flow without blocking it", "fleet: metrics/logging are tributaries to main decision flow", "monitoring dashboard: displays data without affecting production"],
            bridges=["side-channel", "fire-and-forget", "async", "non-blocking"],
            tags=["orchestration", "tributary", "side-channel", "pattern"])

    def _load_tactics(self):
        ns = self.add_namespace("tactics",
            "Specific actionable maneuvers for achieving goals under constraints")

        ns.define("shape-shift",
            "Change your behavioral profile to match the environment's expectations without changing core capability",
            description="A spy adopts the mannerisms of the local culture to blend in. The spy's core capability (intelligence gathering) doesn't change — only the surface presentation. In the fleet: an agent shapes its communication style (verbosity, formality) to match the context — terse for urgent situations, detailed for analysis. Core deliberation doesn't change, only the output presentation. Shape-shift is the fleet's pragmatics layer.",
            level=Level.PATTERN,
            examples=["spy: adopts local culture's mannerisms (shape-shifts)", "fleet agent: terse in crisis, detailed in analysis (same core, different surface)", "human: formal at work, casual at home (shape-shift without identity change)"],
            bridges=["adaptation", "presentation", "context-aware", "pragmatics"],
            tags=["tactics", "shape-shift", "context", "verb"])

        ns.define("preempt-and-cache",
            "Predict what will be needed and prepare it before the request arrives",
            description="A chef preps ingredients before the dinner rush. A browser pre-fetches links you'll probably click. The cache hit rate improves dramatically when you predict correctly. In the fleet: cuda-cache's warm() method pre-populates the cache with predicted accesses. cuda-schedule can preemptively assign agents to likely upcoming tasks. Preempt-and-cache trades storage cost for latency improvement.",
            level=Level.PATTERN,
            examples=["chef: preps ingredients before dinner rush", "browser: prefetches likely-clicked links", "fleet: warm cache with predicted accesses before demand"],
            bridges=["cache", "prediction", "latency", "tradeoff"],
            tags=["tactics", "preempt", "cache", "verb"])

        ns.define("bleed-and-clot",
            "Allow a controlled loss to prevent worse damage, then seal the breach",
            description="A scuba diver with a small leak doesn't surface immediately (losing dive time). They manage the leak, finish the task, then fix it. Bleed: accept controlled loss. Clot: seal when safe. In the fleet: cuda-backpressure allows some message drops (controlled bleed) to prevent queue overflow (worse damage). The circuit breaker allows some failed requests (bleed) before opening (clot). Bleed-and-clot is controlled loss acceptance.",
            level=Level.PATTERN,
            examples=["scuba diver: manage small leak, finish dive, then fix (bleed then clot)", "fleet: drop some messages to prevent queue overflow (bleed, then clot when load drops)", "military: tactical retreat (bleed) to regroup and counterattack (clot)"],
            bridges=["backpressure", "circuit-breaker", "controlled-loss", "resilience"],
            tags=["tactics", "bleed", "clot", "verb"])

        ns.define("probe-before-commit",
            "Send a minimal test request before investing full resources in an action",
            description="Before ordering 1000 units, order 10 and test them. Before deploying to 100 servers, deploy to 1 and monitor. Probe: minimal investment to gather information. Then commit based on the probe result. In the fleet: before assigning a task to an unknown agent (full commit), send a health check first (probe). Before using a new gene in production (commit), test it in sandbox (probe). Probe-before-commit prevents expensive failures.",
            level=Level.PATTERN,
            examples=["order 10 units, test, then order 1000 (probe before commit)", "deploy to 1 server, monitor, then 100 (probe before commit)", "fleet: health check agent before assigning task (probe before commit)"],
            bridges=["testing", "canary", "minimal-investment", "risk-reduction"],
            tags=["tactics", "probe", "commit", "verb"])

        ns.define("hedge-position",
            "Make a counterbalancing investment that reduces downside risk without eliminating upside",
            description="Buy a stock AND buy a put option. Stock goes up: profit from stock. Stock goes down: put option limits loss. The hedge costs something (the option premium) but prevents catastrophic loss. In the fleet: maintain backup strategies alongside primary strategies. Primary fails? Backup activates. The backup costs energy (the premium) but prevents total failure. cuda-resource's budget allocation hedges: not all energy on one strategy.",
            level=Level.PATTERN,
            examples=["stock + put option: profit if up, limited loss if down", "fleet: primary strategy + backup strategy (hedge)", "backup generator: costs money to maintain, prevents total outage"],
            bridges=["hedging", "backup", "risk-reduction", "diversification"],
            tags=["tactics", "hedge", "backup", "verb"])

    def _load_diagnostics(self):
        ns = self.add_namespace("diagnostics",
            "The art of identifying what's wrong from symptoms, not causes")

        ns.define("symptom-trace",
            "Follow a visible symptom backward through the causal chain to find the root cause",
            description="Patient has fever (symptom). Why? Infection. Why? Bacteria. Why? Contaminated water. Why? Broken pipe. Symptom-trace: each 'why?' step goes one level deeper. Stop when you reach an actionable cause (fix the pipe). In the fleet: agent reports high latency (symptom). Why? Queue backup. Why? Slow consumer. Why? CPU spike. Why? Inefficient algorithm. Fix the algorithm. Symptom-trace is the universal diagnostic method.",
            level=Level.PATTERN,
            examples=["fever → infection → bacteria → contaminated water → broken pipe (fix pipe)", "fleet: latency → queue → slow consumer → CPU spike → bad algorithm (fix algorithm)", "car: vibration → unbalanced tire → worn bearing (replace bearing)"],
            bridges=["root-cause", "five-whys", "causal-chain", "diagnosis"],
            tags=["diagnostics", "symptom", "trace", "verb"])

        ns.define("diff-and-attribute",
            "Compare current state to a known-good state and attribute each difference to a specific change",
            description="git diff: compare current code to last working version. Each hunk is attributed to a specific commit. Diff-and-attribute: the binary search of diagnostics. When did it break? What changed? Diff between working and broken, attribute each change to its author and reason. In the fleet: cuda-persistence's snapshot/rollback enables diff-and-attribute: compare current state to last known-good snapshot, attribute differences to specific events.",
            level=Level.PATTERN,
            examples=["git diff: current vs working, each change attributed to commit", "fleet: snapshot comparison, differences attributed to events", "medical: compare blood work to baseline, attribute each anomaly to condition"],
            bridges=["diff", "attribution", "snapshot", "comparison"],
            tags=["diagnostics", "diff", "attribute", "verb"])

        ns.define("canary-deploy",
            "Route a small fraction of traffic to a new version to detect problems before full rollout",
            description="Coal miners brought canaries into mines. Canaries are more sensitive to gas. Canary dies → miners evacuate. In software: route 1% of traffic to new version. Canary shows errors → rollback. 99% of users never see the problem. In the fleet: deploy a new gene to 10% of agents (canary group), observe fitness impact, then roll out to 100% or rollback. Canary-deploy is controlled exposure to risk.",
            level=Level.PATTERN,
            examples=["canary in coal mine: dies first, warns miners", "software: 1% traffic to new version (canary)", "fleet: new gene to 10% of agents (canary group)"],
            bridges=["testing", "gradual-rollout", "risk-control", "monitoring"],
            tags=["diagnostics", "canary", "deploy", "verb"])

        ns.define("smoke-test",
            "Run a quick, shallow validation that catches catastrophic failures without deep verification",
            description="Plug it in, turn it on — does smoke come out? Smoke-test: verify the system boots, handles basic requests, doesn't crash on simple inputs. Doesn't verify correctness, just viability. In the fleet: cuda-metrics' health check IS a smoke test — 'are you alive and responsive?' not 'are your decisions optimal?' Smoke-test before deep-test. If it doesn't smoke-test, deep testing is wasted.",
            level=Level.CONCRETE,
            examples=["electronics: plug in, turn on, no smoke = pass", "fleet: health check pings agent, responds = pass", "software: `make && ./run --test` completes without crash = pass"],
            bridges=["testing", "validation", "shallow", "quick"],
            tags=["diagnostics", "smoke-test", "quick", "concrete"])

    def _load_leverage(self):
        ns = self.add_namespace("leverage",
            "Small actions that produce disproportionately large effects — finding and using multipliers")

        ns.define("leverage-point",
            "A place in a system where a small change produces a large effect — Donella Meadows' insight",
            description="Meadows identified 12 leverage points in systems, from weakest (parameters) to strongest (paradigm). Changing a parameter (tax rate from 20% to 21%) has small effect. Changing the paradigm (from GDP to well-being as success metric) has enormous effect. In the fleet: adding a new gene to the pool (parameter change) has small effect. Changing the fitness function (paradigm change) has enormous effect. Always seek the highest leverage point.",
            level=Level.META,
            examples=["Meadows: changing tax rate (weak) vs changing success metric (strong)", "fleet: adjusting energy cost (weak) vs changing fitness function (strong)", "organization: adjusting work hours (weak) vs changing culture (strong)"],
            bridges=["systems-thinking", "multiplier", "paradigm", "effectiveness"],
            tags=["leverage", "multiplier", "systems", "meta"])

        ns.define("key-stone",
            "A component whose removal causes disproportionate system collapse — not the biggest, but the most load-bearing",
            description="In an arch, the keystone is ONE stone. Remove it, the entire arch collapses. It's not the biggest stone, not the heaviest — it's the one that distributes load to all others. In the fleet: the confidence type IS a keystone. It's not the biggest component, but removing it breaks trust propagation, decision-making, A2A negotiation, and sensor fusion simultaneously. cuda-confidence is the fleet's keystone.",
            level=Level.DOMAIN,
            examples=["arch keystone: one stone, entire arch depends on it", "fleet: confidence type — small but everything depends on it", "species: sea otter (keystone species) — removal collapses kelp forest ecosystem"],
            bridges=["critical-component", "dependency", "load-bearing", "architecture"],
            tags=["leverage", "keystone", "critical", "domain"])

        ns.define("force-multiplier",
            "A tool or technique that amplifies the effectiveness of existing effort without requiring more resources",
            description="A lever lets you move 100 lbs with 10 lbs of force (10x multiplier). A bulldozer moves earth 100x faster than a shovel. The resources (your labor) don't change — the multiplier does. In the fleet: HAV is a force multiplier for communication — one term replaces a paragraph. cuda-equipment is a force multiplier — shared code avoids reimplementing across 100+ crates. Force multipliers are the most valuable fleet components.",
            level=Level.DOMAIN,
            examples=["lever: 10 lbs moves 100 lbs (10x)", "HAV: one term replaces paragraph (100x compression)", "shared library: write once, used 100 times (100x)"],
            bridges=["multiplier", "efficiency", "amplification", "leverage"],
            tags=["leverage", "multiplier", "amplification", "domain"])

        ns.define("catalyst",
            "An agent that enables a reaction without being consumed by it — the facilitator, not the participant",
            description="Chemical catalyst: enables reaction between A and B without becoming part of the product. The catalyst lowers the activation energy, making the reaction possible. In the fleet: the captain (cuda-captain) is a catalyst — it enables coordination between agents without directly doing their work. HAV is a catalyst — it enables understanding between agents without being part of their deliberation. The best tools are catalysts.",
            level=Level.DOMAIN,
            examples=["enzyme: enables chemical reaction without being consumed", "fleet captain: enables coordination without doing the work", "HAV: enables understanding without being part of deliberation"],
            bridges=["facilitator", "enabler", "non-consumed", "activation-energy"],
            tags=["leverage", "catalyst", "facilitator", "domain"])

    def _load_adaptation_patterns(self):
        ns = self.add_namespace("adaptation-patterns",
            "Recurring patterns of how systems adapt to changing conditions")

        ns.define("acclimatize",
            "Gradually adjust to a new environment by incrementally changing operating parameters",
            description="Move from sea level to high altitude: first day, headache. Second day, better. By day 5, adapted. The body gradually adjusts red blood cell count, breathing patterns, metabolism. Not instant — gradual. In the fleet: cuda-adaptation's strategy parameters acclimatize — not switching instantly but gradually shifting weights as environmental statistics change. Acclimatization prevents oscillation: instant switching would thrash, gradual adjustment stabilizes.",
            level=Level.BEHAVIOR,
            examples=["altitude: gradual RBC adjustment over days", "fleet: gradual strategy weight adjustment as environment changes", "diet: gradual change in gut microbiome to new foods"],
            bridges=["gradual", "adaptation", "stabilization", "transition"],
            tags=["adaptation", "acclimatize", "gradual", "verb"])

        ns.define("phase-lock",
            "Synchronize internal oscillation to an external signal, maintaining precise timing alignment",
            description="Your circadian rhythm phase-locks to the sun cycle. Flip time zones: disrupted for days until it re-locks. PLL (phase-locked loop) in electronics: oscillator syncs to reference signal. In the fleet: cuda-energy's CircadianRhythm phase-locks instinct modulation to the time of day. Agents working together phase-lock their task cycles — not through explicit coordination, but through shared timing signals. Phase-lock is implicit synchronization.",
            level=Level.DOMAIN,
            examples=["circadian rhythm: phase-locks to sun cycle", "PLL: oscillator syncs to reference clock", "fleet agents: task cycles phase-lock to shared timing signals"],
            bridges=["synchronization", "oscillation", "timing", "entrainment"],
            tags=["adaptation", "phase-lock", "sync", "verb"])

        ns.define("cope-and-advance",
            "Manage the immediate crisis while simultaneously making progress toward long-term resolution",
            description="Not just coping (surviving). Not just advancing (ignoring crisis). Both simultaneously. Put out the fire while designing a fireproof building. In the fleet: cuda-resilience handles the immediate failure (cope) while cuda-learning extracts lessons that prevent future failures (advance). An agent that only copes never improves. An agent that only advances collapses at the first crisis. Both are required.",
            level=Level.PATTERN,
            examples=["put out fire while designing fireproof building", "fleet: handle failure now (cope) while learning to prevent it (advance)", "startup: fix urgent bug (cope) while improving testing (advance)"],
            bridges=["resilience", "learning", "dual-track", "improvement"],
            tags=["adaptation", "cope", "advance", "verb"])

        ns.define("condition-on-context",
            "Change behavior based on environmental context without explicit if/then rules — the context IS the selector",
            description="A chameleon doesn't decide 'if green background then turn green'. Its skin responds directly to the light. The context (light spectrum) directly modulates the behavior (skin color). No decision layer needed. In the fleet: cuda-energy's circadian rhythm conditions instinct strength on time of day. No rule says 'if night then reduce navigation'. The cosine function directly modulates. Condition-on-context removes decision overhead.",
            level=Level.PATTERN,
            examples=["chameleon: skin responds directly to light (context IS the selector)", "fleet: circadian cosine directly modulates instinct strength (no if/night rule)", "thermostat: directly responds to temperature (no 'if cold then heat' — just bimetallic strip bending)"],
            bridges=["context-awareness", "direct-modulation", "reactive", "emergent"],
            tags=["adaptation", "context", "condition", "verb"])

    def _load_friction(self):
        ns = self.add_namespace("friction",
            "Resistance that slows systems down — sometimes harmful, sometimes protective")

        ns.define("necessary-friction",
            "Resistance that prevents reckless action and ensures deliberate decision-making",
            description="A safety catch on a gun. A confirmation dialog on 'rm -rf'. A mandatory cooling-off period before a major decision. These frictions slow you down ON PURPOSE. Without them, impulsive mistakes. With them, deliberate action. In the fleet: cuda-deliberation IS necessary friction — it slows the path from perception to action, preventing impulsive responses. The consensus threshold adds friction: you must accumulate enough evidence before acting.",
            level=Level.PATTERN,
            examples=["gun safety catch: friction prevents accidental firing", "confirmation dialog: friction prevents accidental deletion", "fleet deliberation: friction between perception and action prevents impulsive response"],
            bridges=["friction", "safety", "deliberation", "deliberate"],
            tags=["friction", "necessary", "safety", "pattern"])

        ns.define("frictionless-path",
            "A route through a system that encounters zero resistance — dangerously easy to take without thinking",
            description="The path of least resistance is the frictionless path. Water flows downhill. Users click the big button. Agents take the default option. The frictionless path isn't always the best path — it's just the easiest. In the fleet: instinct-based action is the frictionless path (no deliberation needed). Deliberation adds friction but may produce better outcomes. Design the frictionless path to be the CORRECT path.",
            level=Level.BEHAVIOR,
            examples=["default settings: most users never change them (frictionless path)", "fleet: instinct action = frictionless (fast but maybe suboptimal)", "checkout: one-click buy = frictionless (easy but maybe impulsive)"],
            bridges=["default", "least-resistance", "fast-path", "design"],
            tags=["friction", "frictionless", "default", "behavior"])

        ns.define("grease-the-path",
            "Remove unnecessary friction to make the desired action the easiest action",
            description="Want people to save? Make saving the default (opt-out, not opt-in). Want agents to share knowledge? Make sharing automatic (push to gene pool, don't require explicit upload). Grease-the-path: identify the desired behavior, then remove every obstacle between the agent and that behavior. In the fleet: cuda-tuple-space's Linda model greases the path for coordination — agents just write to the shared space, no explicit addressing needed.",
            level=Level.PATTERN,
            examples=["save by default (opt-out): greases the path to saving", "fleet: automatic gene pool sharing: greases the path to knowledge transfer", "Linda tuple space: write anywhere, read anywhere (greased coordination)"],
            bridges=["default", "friction-removal", "nudge", "design"],
            tags=["friction", "grease", "remove", "verb"])

    def _load_compression(self):
        ns = self.add_namespace("compression",
            "Encoding more meaning in less space — the core challenge of knowledge transfer")

        ns.define("glossary-as-code",
            "When shared vocabulary itself becomes executable — terms that trigger predefined behaviors",
            description="A glossary that's also an API. Say 'throttle-match' and the system knows exactly what to do: adjust processing rate to input rate. The term IS the instruction. HAV terms are glossary-as-code when the fleet has corresponding implementations: 'deliberate' triggers cuda-deliberation, 'stigmergy-mark' triggers cuda-stigmergy. The vocabulary IS the command set.",
            level=Level.META,
            examples=["HAV term 'deliberate' → triggers cuda-deliberation (glossary-as-code)", "military jargon 'flank' → specific maneuver (glossary-as-code)", "medical order 'stat' → immediately (glossary-as-code)"],
            bridges=["vocabulary", "executable", "command", "protocol"],
            tags=["compression", "glossary", "executable", "meta"])

        ns.define("shorthand-convention",
            "An agreed-upon abbreviation that experienced practitioners use to communicate at speed",
            description="Doctors: 'SOB' (shortness of breath), 'NPO' (nothing by mouth). Pilots: 'ILS approach' (instrument landing system). Each abbreviation compresses a full concept. Newcomers are lost. Veterans communicate at 5x speed. In the fleet: HAV terms are shorthand conventions for fleet agents. 'Confidence-fuse' = harmonic mean fusion of independent confidence sources, with decay, with threshold gating. One word, full specification.",
            level=Level.PATTERN,
            examples=["medical: SOB = shortness of breath (compresses 3 words to 3 letters)", "fleet: 'confidence-fuse' = harmonic mean fusion with decay and threshold (one word = full spec)", "pilot: 'ILS approach' = instrument landing system approach (3 letters = full procedure)"],
            bridges=["abbreviation", "compression", "shared-knowledge", "speed"],
            tags=["compression", "shorthand", "convention", "pattern"])

        ns.define("schema-on-read",
            "Don't predefine data structure — interpret the meaning when you need it",
            description="Relational databases: schema-on-write (define table before inserting). JSON databases: schema-on-read (interpret structure when querying). Schema-on-read is more flexible: data can take any shape, and different readers can interpret the same data differently. In the fleet: A2A messages are schema-on-read — the payload is a flexible JSON object, and each agent interprets it according to its own needs. No rigid message schema needed.",
            level=Level.PATTERN,
            examples=["JSON: no predefined schema, interpret when needed", "fleet A2A: flexible message payload, each agent interprets as needed", "Wikipedia: unstructured text, each reader extracts what they need"],
            bridges=["flexibility", "interpretation", "JSON", "dynamic"],
            tags=["compression", "schema", "read", "pattern"])

    def _load_boundaries(self):
        ns = self.add_namespace("boundaries",
            "The edges where systems interact — interfaces, perimeters, and transition zones")

        ns.define("membrane-permeability",
            "The rate and selectivity at which a boundary allows substances to pass through",
            description="Cell membrane: lets oxygen in, keeps toxins out. Permeability IS the filter. High permeability = fast exchange, low selectivity. Low permeability = slow exchange, high selectivity. In the fleet: the membrane (cuda-genepool) has configurable permeability. High permeability = more external genes enter the pool (innovation, risk). Low permeability = pool stays stable (safety, stagnation). Tuning permeability IS tuning the innovation-safety tradeoff.",
            level=Level.DOMAIN,
            examples=["cell membrane: selective permeability (oxygen in, toxins out)", "fleet membrane: configurable permeability for gene import (innovation vs safety)", "border: high permeability = free trade + vulnerability, low = protection + stagnation"],
            bridges=["membrane", "filter", "selectivity", "tradeoff"],
            tags=["boundaries", "permeability", "membrane", "domain"])

        ns.define("demilitarized-zone",
            "A buffer region between two hostile systems that enables monitored, controlled interaction",
            description="Between North and South Korea: a strip where neither side deploys weapons. DMZ enables coexistence without trust. In the fleet: between trusted and untrusted agents, a DMZ layer monitors and sanitizes all interactions. cuda-sandbox IS a DMZ: the untrusted agent runs inside it, all outputs are inspected before passing to the trusted fleet. DMZ enables interaction without trust.",
            level=Level.PATTERN,
            examples=["Korean DMZ: buffer between hostile systems", "fleet: sandbox as DMZ between trusted fleet and untrusted agent", "firewall DMZ: public-facing servers isolated from internal network"],
            bridges=["sandbox", "buffer", "monitoring", "trust-boundary"],
            tags=["boundaries", "DMZ", "buffer", "pattern"])

        ns.define("handshake-protocol",
            "A multi-step exchange that establishes mutual understanding before the main interaction begins",
            description="TCP three-way handshake: SYN → SYN-ACK → ACK. Before data flows, both sides agree on parameters. The handshake ensures both sides are ready, compatible, and aware of each other's capabilities. In the fleet: cuda-a2a's initial message exchange IS a handshake — agents share capabilities, trust scores, and communication preferences before substantive coordination begins. No handshake = miscommunication.",
            level=Level.CONCRETE,
            examples=["TCP: SYN → SYN-ACK → ACK before data", "fleet: capability exchange before coordination", "human: introduce yourself, find common ground, then discuss business (social handshake)"],
            bridges=["protocol", "setup", "mutual-understanding", "A2A"],
            tags=["boundaries", "handshake", "protocol", "concrete"])

    def _load_temporal_patterns(self):
        ns = self.add_namespace("temporal-patterns",
            "How systems change over time — rhythms, cycles, trajectories, and deadlines")

        ns.define("half-life-decay",
            "Exponential decay where half the quantity is lost every fixed period — the universal aging function",
            description="Carbon-14: half gone every 5,730 years. Radioactive medicine: half gone every 6 hours. Trust: half gone every N interactions without reinforcement. The half-life parameter controls decay SPEED. Short half-life = rapid decay (hot data). Long half-life = slow decay (cold data). In the fleet: 30+ cuda-* crates use half-life decay for confidence, trust, energy, memory, stigmergy, attention. Half-life IS the fleet's universal aging constant.",
            level=Level.DOMAIN,
            examples=["carbon-14: half-life 5,730 years", "fleet confidence: half-life controls decay rate", "memory: half-life determines how quickly experience fades"],
            bridges=["decay", "exponential", "aging", "universal"],
            tags=["temporal", "half-life", "decay", "domain"])

        ns.define("lead-time",
            "The time between initiating an action and its completion — the response delay",
            description="Order a custom part: 6 weeks lead time. Spawn a subagent: 2 minutes lead time. Lead time determines how far ahead you must plan. Short lead time = reactive (respond quickly). Long lead time = anticipatory (plan ahead). In the fleet: agent spawning has a lead time (startup overhead). Task assignment should account for lead time — don't assign urgent tasks to cold agents with long startup lead times.",
            level=Level.CONCRETE,
            examples=["custom part: 6-week lead time", "subagent spawn: 2-minute lead time", "fleet: agent startup time is lead time for task assignment"],
            bridges=["delay", "planning", "anticipation", "response"],
            tags=["temporal", "lead-time", "delay", "concrete"])

        ns.define("time-to-live",
            "A deadline after which cached or stored data is considered stale and discarded",
            description="DNS records: 3600 second TTL. HTTP cache: max-age header. Fleet sensor data: 500ms TTL (position data older than 500ms is stale). TTL prevents the system from acting on outdated information. Short TTL = fresh but expensive (frequent refresh). Long TTL = stale but efficient (infrequent refresh). In the fleet: cuda-cache's TTL, cuda-tuple-space's TTL, cuda-world-model's object permanence — all use TTL.",
            level=Level.CONCRETE,
            examples=["DNS: 3600s TTL before re-query", "HTTP: max-age before cache invalidation", "fleet: sensor data 500ms TTL, stigmergy marks with expiry"],
            bridges=["expiry", "staleness", "cache", "freshness"],
            tags=["temporal", "TTL", "expiry", "concrete"])

        ns.define("window-of-opportunity",
            "A bounded time interval during which an action can succeed — outside the window, it cannot",
            description="Stock option: exercise before expiry or it's worthless. Launch window: 30 minutes for Mars orbit insertion. In the fleet: cuda-temporal's deadline urgency scoring identifies windows of opportunity — tasks that must be completed before a deadline. Energy budget is the window's width: enough energy = wide window, low energy = narrow window. Miss the window = opportunity cost.",
            level=Level.DOMAIN,
            examples=["stock option: expires worthless after date", "launch window: 30 minutes for orbital insertion", "fleet: task deadline creates window, energy budget sets window width"],
            bridges=["deadline", "expiry", "opportunity", "urgency"],
            tags=["temporal", "window", "opportunity", "domain"])

    def _load_quality(self):
        ns = self.add_namespace("quality",
            "Measures and patterns of excellence, reliability, and fitness for purpose")

        ns.define("definition-of-done",
            "The explicit criteria that must be satisfied before a task is considered complete",
            description="Not 'it compiles'. Not 'it works on my machine'. Definition of done: tests pass, documentation updated, code reviewed, deployed, and verified in production. Different teams have different DoDs. The key is EXPLICIT — everyone agrees beforehand what 'done' means. In the fleet: each deliberation proposal should have a DoD — what does 'resolved' look like? What evidence would satisfy the threshold?",
            level=Level.CONCRETE,
            examples=["software: tests pass + docs updated + reviewed + deployed + verified = done", "fleet: proposal's DoD = confidence above threshold with supporting evidence", "construction: inspection passed + occupancy permit = done"],
            bridges=["completion", "criteria", "explicit", "agreement"],
            tags=["quality", "done", "criteria", "concrete"])

        ns.define("single-point-of-failure",
            "A component whose failure causes the entire system to stop — eliminate these",
            description="One hard drive holds all data → it fails → everything lost. One load balancer serves all traffic → it crashes → site down. SPOF is the enemy of resilience. Every SPOF should have a backup. In the fleet: single captain = SPOF. Single energy source = SPOF. cuda-election's leader redundancy and cuda-energy's fleet energy pool eliminate SPOFs. Resilient systems have ZERO single points of failure.",
            level=Level.CONCRETE,
            examples=["single hard drive with all data = SPOF", "single load balancer = SPOF", "fleet: single captain = SPOF, need election for backup"],
            bridges=["resilience", "redundancy", "backup", "eliminate"],
            tags=["quality", "SPOF", "failure", "concrete"])

        ns.define("blast-radius",
            "The scope of damage when a failure occurs — contain it, minimize it, monitor it",
            description="A firecracker in a field: small blast radius. A firecracker in a fireworks factory: large blast radius. Same failure, different context. In the fleet: cuda-resilience's bulkhead pattern limits blast radius — one agent's failure doesn't cascade to others. The blast radius of a single agent failure is bounded by its bulkhead. Design systems so that worst-case blast radius is acceptable, not catastrophic.",
            level=Level.PATTERN,
            examples=["firecracker in field vs fireworks factory: same failure, different blast radius", "fleet: bulkhead limits agent failure blast radius", "microservices: one service fails, others unaffected (contained blast radius)"],
            bridges=["bulkhead", "containment", "failure-scope", "resilience"],
            tags=["quality", "blast-radius", "containment", "pattern"])


    def _load_mechanics(self):
        ns = self.add_namespace("mechanics",
            "Physical force, motion, and mechanical advantage as metaphors for agent systems")

        ns.define("mechanical-advantage",
            "A device that multiplies input force — trading distance for force at a fixed energy cost",
            description="A lever: push 1 meter at 10 lbs to move 10 lbs 1 meter. A pulley system: pull 10 meters of rope to lift 100 lbs 1 meter. Same energy (F×d = constant), but the force is multiplied. In the fleet: cuda-equipment is mechanical advantage — shared code lets one crate's work benefit 100+ crates. Write once (input force), used everywhere (output force). The energy cost is maintenance of the shared crate. The advantage is 100x.",
            level=Level.PATTERN,
            examples=["lever: 10 lbs input × 1m = 10 lbs output × 1m", "pulley: 10 lbs × 10m = 100 lbs × 1m", "fleet: shared equipment crate = mechanical advantage for all dependents"],
            bridges=["force-multiplier", "leverage", "shared-infrastructure", "efficiency"],
            tags=["mechanics", "advantage", "multiplier", "pattern"])

        ns.define("momentum",
            "The tendency of a moving object to keep moving — resistance to change in velocity",
            description="A freight train at full speed doesn't stop instantly. It has momentum (mass × velocity). More mass or more speed = more momentum = harder to stop. In the fleet: an agent with high momentum (long history of consistent behavior, strong gene fitness) resists change. Redirecting it costs energy. cuda-adaptation's strategy switching must overcome momentum — a strategy that's been working well has high momentum, making switching costly.",
            level=Level.DOMAIN,
            examples=["freight train: hard to stop (high mass × high velocity)", "fleet agent: long history of success resists strategy change (high momentum)", "project: late-stage rewrite has high momentum (switching costs enormous)"],
            bridges=["inertia", "resistance-to-change", "velocity", "mass"],
            tags=["mechanics", "momentum", "inertia", "domain"])

        ns.define("torque",
            "Rotational force — the ability to turn something around an axis, not just push it linearly",
            description="Push a door at the handle: it swings easily. Push it near the hinge: it barely moves. Same force, different torque (force × distance from axis). In the fleet: applying effort at the right point creates torque — a small change to the fitness function (applied at the right leverage point) rotates the entire gene pool's evolution direction. Same effort, different location = different torque.",
            level=Level.PATTERN,
            examples=["door: push at handle (high torque) vs near hinge (low torque)", "fleet: small fitness function change at right leverage point = rotates entire gene pool", "organization: suggestion from CEO (high torque) vs intern (low torque) — same idea, different leverage"],
            bridges=["leverage", "rotational-force", "axis", "position"],
            tags=["mechanics", "torque", "leverage", "pattern"])

        ns.define("gear-ratio",
            "Match speed to force through mechanical reduction or multiplication",
            description="Low gear: slow but powerful (climbing a hill). High gear: fast but weak (highway cruising). The gear ratio trades speed for force. In the fleet: cuda-deliberation has a gear ratio — slow mode (deep analysis, high confidence) for critical decisions, fast mode (shallow analysis, low confidence) for routine ones. Switching gears IS changing the speed-force tradeoff. cuda-filtration's BudgetTiers are gear ratios.",
            level=Level.PATTERN,
            examples=["bicycle: low gear for hills (slow, powerful), high gear for flats (fast, weak)", "fleet: slow deliberation for critical decisions, fast for routine", "filtration tiers: scout (fast, cheap) vs captain (slow, thorough)"],
            bridges=["tradeoff", "speed-force", "gear", "ratio"],
            tags=["mechanics", "gear-ratio", "tradeoff", "pattern"])

    def _load_entrenchment(self):
        ns = self.add_namespace("entrenchment",
            "How systems become locked into patterns and the difficulty of changing them")

        ns.define("lock-in",
            "A state where switching costs make staying with the current choice cheaper than changing, even if better alternatives exist",
            description="QWERTY keyboard: inferior to Dvorak but everyone learned QWERTY, all keyboards are QWERTY, switching costs are enormous. Lock-in is self-reinforcing: the more people use QWERTY, the harder it is to switch. In the fleet: cuda-equipment as a shared dependency creates lock-in — once 100+ crates depend on it, changing its API is very costly. Lock-in isn't always bad (standardization enables coordination) but should be conscious.",
            level=Level.DOMAIN,
            examples=["QWERTY keyboard: inferior but locked in by switching costs", "fleet: cuda-equipment API change affects 100+ crates (lock-in)", "Facebook: locked in by social graph (switching costs = losing all friends)"],
            bridges=["switching-cost", "path-dependence", "standardization", "self-reinforcing"],
            tags=["entrenchment", "lock-in", "switching-cost", "domain"])

        ns.define("path-dependence",
            "Where you end up depends on the path you took, not just where you started — history matters",
            description="VHS beat Betamax despite Betamax being technically superior. Early VHS adoption led to more VHS content, leading to more VHS adoption. The path determined the outcome, not the starting quality. In the fleet: the gene pool's evolution IS path-dependent. Different starting genes → different evolutionary paths → different optimal strategies. You can't reach the same gene pool from different starting points.",
            level=Level.DOMAIN,
            examples=["VHS vs Betamax: history determined winner, not quality", "fleet: gene pool evolution depends on starting genes (path-dependent)", "QWERTY: historical accident locked in the standard"],
            bridges=["lock-in", "history", "contingency", "non-deterministic"],
            tags=["entrenchment", "path-dependence", "history", "domain"])

        ns.define("technical-debt",
            "The accumulated cost of choosing fast-now over right-now — future work created by past shortcuts",
            description="Ship without tests → later you MUST add tests (with interest: harder now because code has grown). Ship without docs → later you MUST reverse-engineer the docs (with interest: original authors are gone). Technical debt has INTEREST — the longer you wait to pay it, the more expensive it becomes. In the fleet: crates built quickly without proper error handling accumulate technical debt. Pay it early (backfill) or pay it later (at compound interest).",
            level=Level.BEHAVIOR,
            examples=["ship without tests → add tests later at 3x cost (technical debt + interest)", "skip documentation → reverse-engineer later at 5x cost", "fleet: minimal agent → add compliance later at higher cost"],
            bridges=["debt", "shortcuts", "interest", "maintenance"],
            tags=["entrenchment", "debt", "shortcuts", "behavior"])

        ns.define("legacy-anchor",
            "An old component that newer components must remain compatible with, constraining evolution",
            description="Windows must run 30-year-old software. HTTP/1.1 must work with HTTP/1.0 clients. The legacy anchor prevents the system from fully evolving because backward compatibility constrains the design space. In the fleet: cuda-equipment v0.1 is a legacy anchor — once published to crates.io, its API must remain stable. New versions can add features but can't remove or change existing ones. The anchor constrains but also enables.",
            level=Level.BEHAVIOR,
            examples=["Windows: runs 30-year-old software (legacy anchor)", "HTTP/1.1: compatible with HTTP/1.0 (legacy anchor)", "fleet: cuda-equipment v0.1 API stability constrains future evolution"],
            bridges=["backward-compatibility", "constraint", "stability", "evolution"],
            tags=["entrenchment", "legacy", "anchor", "behavior"])

    def _load_knowledge_transfer(self):
        ns = self.add_namespace("knowledge-transfer",
            "How knowledge moves between agents, systems, and generations")

        ns.define("standing-on-shoulders",
            "Each generation builds on accumulated knowledge, achieving heights impossible from scratch",
            description="Newton: 'If I have seen further, it is by standing on the shoulders of giants.' No one builds modern physics from first principles. Everyone starts with centuries of accumulated knowledge. In the fleet: cuda-equipment is the accumulated knowledge of the fleet. Every new crate starts standing on its shoulders — confidence types, message formats, agent traits. Without it, every crate reimplements everything. With it, every crate starts at height.",
            level=Level.META,
            examples=["Newton: standing on giants' shoulders", "fleet: every new crate starts with cuda-equipment (accumulated fleet knowledge)", "open source: every project builds on libraries (standing on shoulders)"],
            bridges=["accumulation", "prior-knowledge", "foundation", "meta"],
            tags=["knowledge", "transfer", "accumulation", "meta"])

        ns.define("knowledge-distillation",
            "Transfer knowledge from a complex source to a simpler representation without losing essential capability",
            description="Teacher model (70B parameters) → Student model (7B parameters). The student captures 90% of the teacher's capability at 10% of the cost. The distillation is LOSSY but EFFICIENT. In the fleet: each HAV term IS a knowledge distillation — a complex academic concept distilled into one word + one paragraph + examples. The full paper is 30 pages. The HAV term is 200 words. 95% of the meaning, 1% of the cost.",
            level=Level.META,
            examples=["teacher 70B → student 7B: 90% capability, 10% cost", "HAV: 30-page paper → 200-word term: 95% meaning, 1% cost", "senior engineer's intuition → documented pattern: transferable at lower cost"],
            bridges=["distillation", "compression", "efficiency", "lossy"],
            tags=["knowledge", "distillation", "compression", "meta"])

        ns.define("apprenticeship",
            "Transfer tacit knowledge through observation and guided practice, not explicit instruction",
            description="A blacksmith's apprentice watches the master work, then tries, gets corrected, tries again. The knowledge transferred is TACIT — the master can't fully articulate what they know. It must be OBSERVED and PRACTICED. In the fleet: cuda-git-agent's Gene crossover is apprenticeship — a new agent inherits genes from successful agents (observation) and tests them in its own context (guided practice). The genes encode tacit knowledge.",
            level=Level.PATTERN,
            examples=["blacksmith apprentice: watches master, practices, gets corrected", "fleet: gene crossover from successful agents = apprenticeship", "residency: medical students observe attending physicians (apprenticeship)"],
            bridges=["tacit-knowledge", "practice", "observation", "learning"],
            tags=["knowledge", "apprenticeship", "tacit", "pattern"])

        ns.define("document-or-it-didn't-happen",
            "Knowledge that exists only in someone's head is not knowledge — it's a dependency on that person",
            description="The senior engineer who 'just knows' the deployment process: when they leave, the process leaves with them. Undocumented knowledge is a single point of failure. Write it down = make it transferable. HAV IS documentation of high-abstraction concepts — written down, transferable, not dependent on any single agent. Document-or-it-didn't-happen is the fleet's knowledge persistence principle.",
            level=Level.PATTERN,
            examples=["senior engineer leaves → undocumented deployment process lost", "fleet: HAV terms documented = transferable regardless of which agent wrote them", "science: experiment results not published = didn't contribute to collective knowledge"],
            bridges=["documentation", "persistence", "transferability", "SPOF"],
            tags=["knowledge", "documentation", "persistence", "pattern"])

    def _load_efficiency(self):
        ns = self.add_namespace("efficiency",
            "Doing more with less — the art of minimizing waste")

        ns.define("amortize",
            "Spread a one-time cost over many uses, making each individual use cheaper",
            description="Buy a $1000 coffee machine. First cup: $1000. 1000th cup: $1. The fixed cost is amortized over many uses. In the fleet: cuda-equipment's development cost is amortized over 100+ dependent crates. Each crate pays a tiny fraction of the development cost. The more crates that use it, the cheaper it is per crate. Amortization IS the economics of shared infrastructure.",
            level=Level.DOMAIN,
            examples=["coffee machine: $1000 first cup, $1 after 1000 cups", "fleet: cuda-equipment dev cost amortized over 100+ crates", "factory: machine purchase amortized over production volume"],
            bridges=["fixed-cost", "shared-infrastructure", "economy-of-scale", "amortization"],
            tags=["efficiency", "amortize", "shared-cost", "domain"])

        ns.define("pay-once-use-forever",
            "Invest in a capability that provides ongoing returns without additional cost",
            description="Buy a hammer: pay once, drive nails forever. Write a test framework: pay once, catch bugs forever. Build cuda-equipment: pay once, use in 100+ crates forever. The investment-to-return ratio improves with usage. In the fleet: every shared crate (cuda-equipment, cuda-confidence, cuda-a2a) is pay-once-use-forever infrastructure. The fleet's development velocity increases as shared infrastructure accumulates.",
            level=Level.PATTERN,
            examples=["hammer: buy once, use forever", "test framework: build once, catch bugs forever", "fleet: shared crate → use in 100+ crates with zero additional development cost"],
            bridges=["investment", "infrastructure", "reusable", "one-time-cost"],
            tags=["efficiency", "pay-once", "reusable", "pattern"])

        ns.define("waste-heat",
            "Unavoidable byproduct energy that can potentially be repurposed instead of discarded",
            description="Car engine generates heat (waste) → heater uses it to warm cabin (repurposed). Data center generates heat → building heating system uses it. Nothing is 100% efficient; the waste-heat question is: can you USE the waste? In the fleet: deliberation generates 'waste' in the form of rejected proposals. But rejected proposals carry information about what DOESN'T work. cuda-learning can extract lessons from rejected proposals (repurposing waste-heat).",
            level=Level.PATTERN,
            examples=["engine heat → cabin heater (repurposed waste-heat)", "fleet: rejected proposals → lessons about what doesn't work (repurposed waste-heat)", " composting: food waste → fertilizer (repurposed waste)"],
            bridges=["byproduct", "repurpose", "efficiency", "waste"],
            tags=["efficiency", "waste-heat", "repurpose", "pattern"])

        ns.define("coasting",
            "Operating at zero or minimal input cost by relying on accumulated energy or momentum",
            description="A bicycle coasting downhill: zero pedaling, still moving. Solar panel at noon: zero fuel cost, maximum output. A system that has built up enough stored energy/momentum to operate without active input. In the fleet: an agent with full energy budget and high-fitness genes is coasting — it handles routine tasks with minimal deliberation (instinct suffices). Coasting IS efficient but can't handle novel situations.",
            level=Level.BEHAVIOR,
            examples=["bicycle downhill: zero pedaling, still moving (coasting)", "solar at noon: zero fuel, maximum output", "fleet: high-fitness agent handles routine tasks by instinct (coasting)"],
            bridges=["momentum", "stored-energy", "efficiency", "minimal-input"],
            tags=["efficiency", "coasting", "minimal", "behavior"])

    def _load_metaphor(self):
        ns = self.add_namespace("metaphor",
            "Cross-domain mapping as a cognitive tool — understanding new things through familiar structures")

        ns.define("structural-mapping",
            "Transfer understanding from a well-known domain to a less-known domain by matching relational structure",
            description="Electricity IS like water: voltage = pressure, current = flow, resistance = pipe width. The mapping works because the RELATIONSHIPS match (Ohm's law = Darcy-Weisbach). In the fleet: biological metaphors map structure, not just labels. Dopamine IS confidence because both accumulate, decay, have thresholds, and modulate behavior — the relational structure matches. Good metaphors have structural fidelity, not just surface similarity.",
            level=Level.META,
            examples=["electricity = water: structural mapping (voltage=pressure, current=flow)", "dopamine = confidence: structural mapping (accumulation, decay, threshold, modulation)", "gene = strategy: structural mapping (inheritance, mutation, selection, fitness)"],
            bridges=["analogy", "cross-domain", "structure", "understanding"],
            tags=["metaphor", "mapping", "structure", "meta"])

        ns.define("reification",
            "Treating an abstract concept as if it were a concrete, manipulable object",
            description="We say 'time is money' — treating abstract time as if it were concrete money (spend, save, waste, invest). Reification makes abstract concepts manipulable. In the fleet: confidence IS a type (f64 between 0 and 1) that can be stored, passed, fused, decayed — treated as a concrete object. Trust IS a type. Energy IS a type. Reifying abstractions as types enables computation over them.",
            level=Level.META,
            examples=["time is money: abstract time treated as concrete currency", "fleet: confidence as f64 type — abstract concept treated as concrete number", "gene as object: abstract strategy pattern treated as manipulable data structure"],
            bridges=["concretization", "type-system", "manipulation", "abstraction"],
            tags=["metaphor", "reification", "concrete", "meta"])

        ns.define("metaphorical-distance",
            "How far apart the source and target domains of a metaphor are — distance affects both insight and confusion",
            description="'The stock market is a roller coaster' — close distance (both involve ups and downs). 'Consciousness is computation' — far distance (very different domains, rich but controversial mapping). Close metaphors are safe but shallow. Far metaphors are risky but insightful. In the fleet: biological metaphors for agent systems have moderate distance — biology and computation share deep structural similarities but aren't the same domain.",
            level=Level.META,
            examples=["stock market = roller coaster (close distance, safe but shallow)", "consciousness = computation (far distance, risky but insightful)", "fleet: biology → agent systems (moderate distance, productive mapping)"],
            bridges=["metaphor", "domain-distance", "insight", "risk"],
            tags=["metaphor", "distance", "domain", "meta"])


    def _load_scaling(self):
        ns = self.add_namespace("scaling-deep",
            "How systems grow, shrink, and maintain function at different scales")

        ns.define("constant-overhead",
            "The cost that doesn't increase as the system scales — the fixed baseline regardless of size",
            description="A factory's rent is $10,000/month whether it produces 1 unit or 10,000. That's constant overhead. Adding more production doesn't increase rent. In the fleet: cuda-equipment's binary size is constant overhead — whether 1 crate or 1000 crates use it, the compiled code is the same size. HTTP protocol is constant overhead — the same headers whether you send 1 byte or 1GB. Minimize constant overhead to keep small deployments viable.",
            level=Level.CONCRETE,
            examples=["factory rent: same whether 1 or 10,000 units produced", "fleet: cuda-equipment binary size is constant regardless of how many crates use it", "HTTP headers: same size whether payload is 1 byte or 1GB"],
            bridges=["fixed-cost", "overhead", "baseline", "scale-invariant"],
            tags=["scaling", "constant", "overhead", "concrete"])

        ns.define("linear-scaling",
            "Output grows proportionally with input — double the resources, double the capacity",
            description="2 workers produce 2x as much as 1 worker. 10 servers handle 10x the traffic. Linear scaling is the ideal baseline but often not achievable due to coordination overhead. In the fleet: cuda-swarm-agent's Vessel fleet ideally scales linearly — 10 agents do 10x the work. But coordination overhead (A2A messages) makes true linear scaling rare. cuda-fleet-mesh's topology affects how close to linear the scaling is.",
            level=Level.DOMAIN,
            examples=["workers: 2x workers = 2x output (ideally)", "fleet: 10 agents handle ~10x tasks (ideally linear)", "servers: 10x servers = ~10x capacity (before coordination overhead)"],
            bridges=["scaling", "proportional", "ideal", "baseline"],
            antonyms=["super-linear-scaling", "sub-linear-scaling"],
            tags=["scaling", "linear", "proportional", "domain"])

        ns.define("super-linear-scaling",
            "Output grows faster than input — 2x resources produce MORE than 2x output",
            description="This seems impossible but happens through positive network effects. A telephone with 1 user is worthless. With 100 users, each user gets value proportional to 100. With 1000 users, value proportional to 1000. The value scales SUPER-linearly because each new user adds value for ALL existing users. In the fleet: the gene pool scales super-linearly — each new gene can combine with ALL existing genes, creating exponential combinatorial value.",
            level=Level.DOMAIN,
            examples=["telephone network: value ~ n^2 (each user connects to all others)", "fleet gene pool: each new gene combines with all existing genes (super-linear value)", "marketplace: each new seller benefits all existing buyers"],
            bridges=["network-effect", "combinatorial", "positive-feedback", "scaling"],
            antonyms=["sub-linear-scaling"],
            tags=["scaling", "super-linear", "network-effect", "domain"])

        ns.define("diminishing-returns",
            "Each additional unit of input produces LESS output than the previous one",
            description="First worker: produces 100 units. Second worker: produces 90 units (coordination overhead). Tenth worker: produces 10 units (mostly overhead). At some point, adding more resources actually HURTS (congestion, overhead exceeds benefit). In the fleet: adding deliberation rounds has diminishing returns — first round identifies the main issue, subsequent rounds refine but each adds less value. Know when to stop deliberating and start acting.",
            level=Level.DOMAIN,
            examples=["workers: 1st produces 100, 2nd produces 90, 10th produces 10", "fleet deliberation: 1st round most valuable, each subsequent round adds less", "studying: first hour most productive, 10th hour almost useless"],
            bridges=["overhead", "congestion", "optimal-stopping", "tradeoff"],
            antonyms=["increasing-returns"],
            tags=["scaling", "diminishing", "returns", "domain"])

        ns.define("right-sizing",
            "Matching system capacity to actual demand — not over-provisioned, not under-provisioned",
            description="A 100-person company with a 500-person office building is over-provisioned (wasted space). A 100-person company with a 50-person office is under-provisioned (congestion). Right-sized: 100-person capacity for 100-person demand. In the fleet: cuda-resource's allocation should right-size agents to their tasks — a simple monitoring task doesn't need a captain-class agent. Match capability to requirement.",
            level=Level.PATTERN,
            examples=["100-person office for 100-person company (right-sized)", "fleet: simple task gets scout-class agent, not captain-class", "database: query capacity matches actual query volume"],
            bridges=["capacity", "demand", "provisioning", "efficiency"],
            tags=["scaling", "right-size", "capacity", "pattern"])

    def _load_interface_patterns(self):
        ns = self.add_namespace("interface-patterns",
            "How system boundaries are designed, crossed, and maintained")

        ns.define("contract-first",
            "Define the interface contract before implementing either side — agreement before code",
            description="Before building the client or the server, write the API contract (OpenAPI spec, protobuf definition, TypeScript interface). Both sides implement against the same contract independently. Contract-first prevents interface mismatches that arise when implementations drift. In the fleet: vessel.json IS a contract — it defines what a vessel can do before the vessel is built. cuda-equipment's EquipmentRegistry defines the contract for all equipment types.",
            level=Level.PATTERN,
            examples=["OpenAPI spec before client/server implementation", "fleet: vessel.json defines vessel contract before building vessel", "TypeScript interface before implementing class"],
            bridges=["interface", "contract", "agreement", "specification"],
            tags=["interface", "contract-first", "specification", "pattern"])

        ns.define("version-with-grace",
            "Support old and new interface versions simultaneously during transition, then deprecate old",
            description="HTTP/2 and HTTP/1.1 coexist during transition. Python 2 and 3 coexisted for a decade. Never break existing consumers immediately. Version-with-grace: add new version, mark old as deprecated, support both for a transition period, then remove old. In the fleet: cuda-equipment v0.1 → v0.2 should version-with-grace — existing dependent crates continue working while new crates use v0.2 features.",
            level=Level.PATTERN,
            examples=["HTTP/2 coexists with HTTP/1.1 during transition", "Python 2 and 3 coexisted for years", "fleet: cuda-equipment versions coexist during transition period"],
            bridges=["backward-compatibility", "deprecation", "migration", "transition"],
            tags=["interface", "versioning", "graceful", "pattern"])

        ns.define("soft-error",
            "A failure that doesn't crash the system but returns a degraded result with error context",
            description="A hard error: crash, exception, system stops. A soft error: return a partial result, flag it as degraded, include what went wrong. Soft errors enable graceful degradation — the system keeps running with reduced capability instead of stopping entirely. In the fleet: cuda-deliberation's consensus below threshold IS a soft error — no decision made, but the system doesn't crash. It returns 'inconclusive' with the evidence collected so far.",
            level=Level.PATTERN,
            examples=["hard error: system crash. soft error: partial result with error flag", "fleet: deliberation returns 'inconclusive' instead of crashing (soft error)", "API: 200 with degraded data vs 500 crash (soft vs hard error)"],
            bridges=["graceful-degradation", "partial-result", "error-handling", "resilience"],
            antonyms=["hard-error"],
            tags=["interface", "soft-error", "degradation", "pattern"])

    def _load_attention(self):
        ns = self.add_namespace("attention-deep",
            "How systems allocate limited processing capacity to the most important inputs")

        ns.define("attention-budget",
            "A fixed cognitive resource that must be allocated across competing demands — spend wisely",
            description="You have ~4 hours of deep focus per day. Every meeting, notification, and context switch spends from this budget. Once depleted, remaining work is shallow. In the fleet: each agent has an attention budget (cuda-attention). Every deliberation, message, and perception event spends attention. Prioritize: high-stakes decisions get deep attention, routine tasks get shallow attention.",
            level=Level.DOMAIN,
            examples=["human: ~4 hours deep focus per day (attention budget)", "fleet: agent attention budget allocated across deliberation, messages, perception", "RAM: fixed memory budget allocated across processes"],
            bridges=["attention", "budget", "prioritization", "allocation"],
            tags=["attention", "budget", "allocation", "domain"])

        ns.define("salience-bottleneck",
            "The rate at which important information can be identified and acted upon — the awareness throughput",
            description="You can scan a dashboard and identify 3-5 anomalies at a glance. But if 50 things are wrong simultaneously, you can't process them all. Salience bottleneck: the rate at which 'important' can be distinguished from 'noise'. In the fleet: cuda-attention's saliency scoring IS the salience bottleneck — only top-scored inputs get processed. Below the bottleneck, inputs are ignored. Tuning the bottleneck is tuning what the agent NOTICEs.",
            level=Level.DOMAIN,
            examples=["human: can notice 3-5 dashboard anomalies simultaneously", "fleet: saliency scoring filters to top-N inputs (salience bottleneck)", "news: 1000 stories published, only 5 noticed (salience bottleneck)"],
            bridges=["attention", "bottleneck", "salience", "throughput"],
            tags=["attention", "salience", "bottleneck", "domain"])

        ns.define("habituation",
            "Decreased response to a stimulus after repeated exposure — the brain's noise filter",
            description="You notice a ticking clock when you first enter a room. After 10 minutes, you don't. The stimulus hasn't changed — your response has. Habituation prevents the brain from processing constant stimuli (saving attention for changes). In the fleet: cuda-attention's habituation reduces saliency for constant inputs. A sensor that always reports 72°F stops getting attention. Only when it changes to 85°F does it break through habituation.",
            level=Level.BEHAVIOR,
            examples=["ticking clock: noticed at first, invisible after 10 minutes", "fleet: constant sensor reading loses attention, only changes break through", "smell: you notice perfume when entering a room, not after 5 minutes"],
            bridges=["adaptation", "change-detection", "noise-filter", "desensitization"],
            tags=["attention", "habituation", "adaptation", "behavior"])

    def _load_security_deep(self):
        ns = self.add_namespace("security-deep",
            "Threat models, trust boundaries, and defensive patterns")

        ns.define("threat-model",
            "An explicit enumeration of what you're protecting against — define the enemy before building defenses",
            description="A bank vault protects against: bank robbers (physical breach), insider theft (trusted attacker), government subpoena (legal attack). Different threats need different defenses. Without a threat model, you build random defenses that may miss the actual threat. In the fleet: the membrane's antibody list IS a threat model — it enumerates specific dangerous signals ('rm -rf', 'format', 'drop_all'). Define threats, then build defenses.",
            level=Level.CONCRETE,
            examples=["bank: protects against robbers, insiders, subpoenas (threat model)", "fleet membrane: antibodies list specific threats (threat model)", "web app: OWASP top 10 as threat model → build defenses accordingly"],
            bridges=["defense", "threat", "model", "explicit"],
            tags=["security", "threat-model", "defense", "concrete"])

        ns.define("trust-boundary",
            "A line where trust assumptions change — inside the boundary is trusted, outside is not",
            description="Your home's front door is a trust boundary. Inside: trusted. Outside: not trusted. You don't need to verify your family's identity inside. You DO verify a stranger at the door. In the fleet: the membrane (cuda-genepool) IS a trust boundary. Inside the membrane: trusted genes, trusted signals. Outside: unverified, must be checked. Every trust boundary needs authentication at the crossing.",
            level=Level.CONCRETE,
            examples=["front door: trust boundary between home (trusted) and street (untrusted)", "fleet membrane: trust boundary between trusted genes and external signals", "firewall: trust boundary between internal network and internet"],
            bridges=["membrane", "boundary", "authentication", "trust"],
            tags=["security", "boundary", "trust", "concrete"])

        ns.define("defense-in-depth",
            "Multiple independent layers of security — if one fails, the next catches the threat",
            description="Castle: moat (layer 1) → wall (layer 2) → guards (layer 3) → inner keep (layer 4). Each layer is independently sufficient to stop SOME attacks. No single layer stops ALL attacks. But together, they stop almost all attacks. In the fleet: cuda-sandbox (layer 1) + cuda-compliance (layer 2) + cuda-rbac (layer 3) + cuda-membrane (layer 4). Each catches different threats. No single defense is sufficient.",
            level=Level.PATTERN,
            examples=["castle: moat + wall + guards + keep (defense in depth)", "fleet: sandbox + compliance + RBAC + membrane (defense in depth)", "web: WAF + rate limit + auth + encrypt (defense in depth)"],
            bridges=["layers", "redundancy", "independent", "defense"],
            tags=["security", "defense-in-depth", "layers", "pattern"])

        ns.define("least-privilege",
            "Grant only the minimum permissions needed for a task — no more, no exceptions",
            description="A janitor has keys to cleaning closets but not the CEO's office. A read-only user can view data but not modify it. Least privilege limits blast radius: if an account is compromised, the attacker only gains the minimum permissions. In the fleet: cuda-rbac implements least-privilege — agents have exactly the permissions needed for their role, no more. A monitoring agent can observe but not modify. An action agent can act but not configure.",
            level=Level.CONCRETE,
            examples=["janitor: cleaning closet keys, not CEO office (least privilege)", "database: read-only user can view but not modify", "fleet: monitoring agent observes but doesn't act (least privilege)"],
            bridges=["permission", "minimum", "RBAC", "blast-radius"],
            tags=["security", "least-privilege", "minimum", "concrete"])

    def _load_error_strategies(self):
        ns = self.add_namespace("error-strategies",
            "Systematic approaches to handling failures gracefully")

        ns.define("fail-fast",
            "Detect and report errors immediately rather than continuing in a degraded state",
            description="Don't swallow exceptions. Don't return null. Don't log and continue. FAIL — loudly and immediately. Fail-fast makes problems visible. Silent failures are worse than crashes because they corrupt state invisibly. In the fleet: Rust's Result/Option types enforce fail-fast — you MUST handle the error case explicitly. The fleet's error strategy: fail fast on critical paths, degrade gracefully on non-critical paths.",
            level=Level.PATTERN,
            examples=["Rust: Result type forces explicit error handling (fail-fast)", "fleet: critical path failure → immediate error, not silent degradation", "assert: program crashes on invariant violation (fail-fast)"],
            bridges=["error-handling", "visibility", "explicit", "fail"],
            antonyms=["fail-silently"],
            tags=["error", "fail-fast", "explicit", "pattern"])

        ns.define("fail-safe",
            "When failure occurs, transition to a state that is safe, not just stopped",
            description="A fuse blows (fails) but the house doesn't burn down (safe). A train signal fails → red (trains stop = safe). Fail-safe: the failure mode itself is safe. Contrast with fail-deadly: a nuclear reactor that fails to shutdown. In the fleet: cuda-energy's apoptosis is fail-safe — when energy drops below threshold, the agent enters rest mode (safe state), not chaos mode. cuda-compliance's deny-rules provide fail-safe defaults.",
            level=Level.PATTERN,
            examples=["fuse blows → house doesn't burn (fail-safe)", "train signal fails → red → trains stop (fail-safe)", "fleet: energy depleted → rest mode (fail-safe, not chaos)"],
            bridges=["safety", "default-state", "failure-mode", "graceful"],
            antonyms=["fail-deadly"],
            tags=["error", "fail-safe", "safety", "pattern"])

        ns.define("retry-with-backoff",
            "Attempt a failed operation again, but with increasing delay between attempts",
            description="First retry: wait 1 second. Second retry: wait 2 seconds. Third: 4 seconds. Exponential backoff prevents hammering a struggling system while still eventually succeeding if the failure is transient. In the fleet: cuda-retry implements retry-with-backoff. Network timeouts, provider failures, and resource contention are often transient — retrying after a delay succeeds. But backoff prevents the retry storm from making things worse.",
            level=Level.CONCRETE,
            examples=["network timeout: retry at 1s, 2s, 4s, 8s (exponential backoff)", "fleet: provider timeout → retry with backoff → eventually succeeds", "git push conflict: retry after delay → someone else pushed, now pull-able"],
            bridges=["retry", "backoff", "transient-failure", "patience"],
            tags=["error", "retry", "backoff", "concrete"])

        ns.define("circuit-breaker",
            "Stop attempting a failing operation after repeated failures, allowing time for recovery",
            description="Too many retries on a truly broken system makes it worse. Circuit breaker: after N failures, OPEN the circuit (stop trying) for a cooldown period. Then HALF-OPEN: try one request. If it succeeds, CLOSE (resume). If it fails, OPEN again. In the fleet: cuda-circuit implements circuit breakers. A failing provider gets its circuit opened — no more attempts until it's had time to recover.",
            level=Level.CONCRETE,
            examples=["electrical circuit breaker: trips on overload, resets after cooldown", "fleet: failing provider → circuit opens → no attempts → half-open test → resume or re-open", "API gateway: failing backend → circuit opens → traffic diverted"],
            bridges=["retry", "failure-detection", "recovery", "protection"],
            tags=["error", "circuit-breaker", "recovery", "concrete"])

    def _load_coordination_deep(self):
        ns = self.add_namespace("coordination-deep",
            "How autonomous agents achieve collective outcomes without central control")

        ns.define("implicit-contract",
            "An agreement that exists through consistent behavior, not explicit negotiation",
            description="You and your roommate don't have a written chore schedule. But you always do dishes on Monday and they always do laundry on Tuesday. The pattern IS the contract. Implicit contracts emerge from repeated interactions. In the fleet: agents develop implicit contracts through stigmergy — consistent patterns of mark-reading and mark-writing form behavioral contracts without explicit A2A negotiation. More efficient than explicit contracts for routine coordination.",
            level=Level.BEHAVIOR,
            examples=["roommate chore pattern: implicit contract without written agreement", "fleet: stigmergy patterns form implicit behavioral contracts", "driving culture: lane usage patterns = implicit contract between drivers"],
            bridges=["stigmergy", "pattern", "emergent", "convention"],
            antonyms=["explicit-contract"],
            tags=["coordination", "implicit", "contract", "behavior"])

        ns.define("barrier",
            "A synchronization point where multiple agents wait until all have arrived before proceeding",
            description="A marathon start line: runners wait until the gun fires, then all start simultaneously. A team meeting: everyone waits until all participants have joined. Barriers prevent agents from proceeding at different speeds and getting out of sync. In the fleet: cuda-workflow's step completion can use barriers — a step that requires multiple agents' output waits until ALL agents have completed their subtasks before the next step begins.",
            level=Level.CONCRETE,
            examples=["marathon start: all wait, then all go simultaneously", "team meeting: wait for all participants before starting", "fleet workflow: step waits for all agent subtasks to complete before proceeding"],
            bridges=["synchronization", "wait", "concurrent", "gate"],
            tags=["coordination", "barrier", "sync", "concrete"])

        ns.define("gossip-protocol",
            "Information spreads by each node sharing with a few random neighbors, achieving global coverage",
            description="Rumor: tell 3 friends, each tells 3 friends, exponentially spreading. Gossip protocols work the same way: each agent periodically shares its state with a few random peers. After O(log N) rounds, all N agents have the information. In the fleet: cuda-trust's gossip sharing spreads trust assessments. cuda-fleet-mesh's gossip protocol spreads topology updates. Gossip is eventually consistent — all agents converge, just not instantly.",
            level=Level.PATTERN,
            examples=["rumor spreading: tell 3, they tell 3, exponential coverage", "fleet trust: gossip spreads trust assessments across all agents", "cluster membership: gossip protocol spreads node join/leave events"],
            bridges=["gossip", "epidemic", "eventual-consistency", "random"],
            tags=["coordination", "gossip", "spread", "pattern"])

    def _load_composition(self):
        ns = self.add_namespace("composition",
            "Building complex systems from simpler, independently useful parts")

        ns.define("plug-and-play",
            "Components that interoperate without configuration — connect them and they work",
            description="USB devices: plug in, they work. No manual configuration. No driver installation (ideally). The interface standard IS the configuration — if both sides implement USB, they work. In the fleet: cuda-equipment provides plug-and-play equipment — define a sensor type and it automatically integrates with perception, navigation, and logging. No manual wiring needed. The trait system IS the plug.",
            level=Level.PATTERN,
            examples=["USB: plug in device, it works (plug-and-play)", "fleet: define equipment type, automatically integrates everywhere", "HDMI: plug monitor in, it works (plug-and-play)"],
            bridges=["interface", "standard", "auto-integration", "compatibility"],
            tags=["composition", "plug-and-play", "auto", "pattern"])

        ns.define("mix-and-match",
            "Combine components freely from a menu of options to create custom configurations",
            description="A burrito bowl: choose rice, beans, protein, salsa from a menu. Different combinations for different people. Same components, different compositions. In the fleet: agents compose from the crate menu: cuda-perception (sensor processing) + cuda-deliberation (decision making) + cuda-motor (action execution) = a complete agent. Mix-and-match crates to create custom agent configurations for different tasks.",
            level=Level.PATTERN,
            examples=["burrito bowl: choose from menu of components (mix-and-match)", "fleet: perception + deliberation + motor = agent (mix-and-match crates)", "PC building: choose CPU, GPU, RAM, storage independently (mix-and-match)"],
            bridges=["composition", "menu", "custom", "modular"],
            tags=["composition", "mix-and-match", "modular", "pattern"])

        ns.define("side-effect-free",
            "A computation that produces a result without modifying any external state — pure function",
            description="f(x) = x + 1. Every time you call it with x=5, you get 6. It doesn't modify global variables. It doesn't write to disk. It doesn't send messages. Side-effect-free functions are COMPOSABLE — you can chain them, parallelize them, test them, and reason about them. In the fleet: cuda-confidence's Conf::fuse() is side-effect-free — it takes two confidences and returns the fused result. No state mutation. Fully composable.",
            level=Level.PATTERN,
            examples=["f(x) = x + 1: pure, no side effects, always same output for same input", "fleet: Conf::fuse(a, b): pure function, no state mutation", "map function in functional programming: pure, composable"],
            bridges=["pure-function", "composability", "testability", "functional"],
            tags=["composition", "side-effect-free", "pure", "pattern"])

    def _load_observability(self):
        ns = self.add_namespace("observability",
            "Understanding what a system is doing from the outside, without modifying it")

        ns.define("telemetry",
            "Automated collection and transmission of system health and performance data",
            description="A plane's black box records altitude, speed, heading, engine status. A car's OBD port reports RPM, fuel, temperature. Telemetry is the automated heartbeat — data flowing from the system to the observer without human intervention. In the fleet: cuda-metrics collects agent telemetry — response times, error rates, energy levels. cuda-logging captures event telemetry. Telemetry enables diagnosis without access to the agent's internals.",
            level=Level.CONCRETE,
            examples=["plane black box: automated recording of all flight data", "car OBD: real-time engine telemetry", "fleet: cuda-metrics collects agent performance telemetry"],
            bridges=["monitoring", "health", "automated", "data-collection"],
            tags=["observability", "telemetry", "monitoring", "concrete"])

        ns.define("distributed-trace",
            "Follow a single request across multiple services to understand the full lifecycle and find bottlenecks",
            description="User clicks 'buy' → frontend (50ms) → auth service (100ms) → inventory (200ms) → payment (300ms) → response (650ms total). Each hop has a trace ID. Distributed tracing follows the request through all hops, showing where time is spent. In the fleet: cuda-provenance tracks deliberation through multiple agents — proposal origin, endorsements, modifications, final decision. The provenance chain IS a distributed trace of the decision.",
            level=Level.CONCRETE,
            examples=["HTTP request traced through frontend → auth → inventory → payment", "fleet: deliberation traced through proposal → endorsements → decision", "package delivery tracked through warehouse → truck → door (trace)"],
            bridges=["trace", "lifecycle", "bottleneck", "provenance"],
            tags=["observability", "trace", "distributed", "concrete"])

        ns.define("canary-in-the-coal-mine",
            "An early warning indicator that detects problems before they affect the whole system",
            description="Miners brought canaries because birds are more sensitive to poison gas. Canary dies → evacuate. In software: a canary query or synthetic transaction that tests system health proactively. In the fleet: cuda-metrics' HealthCheck is the canary — it runs simple checks that detect degradation before full failure. Low-severity anomalies in the canary indicate brewing problems. Catch it at the canary stage, not the catastrophe stage.",
            level=Level.PATTERN,
            examples=["coal mine canary: sensitive to gas, dies first (early warning)", "software: synthetic transaction detects degradation before user impact", "fleet: health check detects performance degradation before full failure"],
            bridges=["early-warning", "canary", "proactive", "monitoring"],
            tags=["observability", "canary", "early-warning", "pattern"])

    def _load_anti_patterns(self):
        ns = self.add_namespace("anti-patterns",
            "Common solutions that seem good but create more problems than they solve")

        ns.define("god-object",
            "A component that knows too much and does too much — the opposite of focused responsibility",
            description="One class that handles database access, business logic, HTML rendering, and email sending. It 'knows everything' because every new feature gets added to it (it's already touching everything). In the fleet: a single agent that handles perception, deliberation, memory, communication, AND action is a god-object. The fleet architecture PREVENTS god-objects by splitting responsibilities across specialized crates.",
            level=Level.BEHAVIOR,
            examples=["one class doing DB + logic + UI + email (god-object)", "fleet: single agent doing everything instead of specialized crates (god-object)", "one person doing CEO + CFO + CTO + janitor (god-object)"],
            bridges=["separation-of-concerns", "modularity", "specialization", "responsibility"],
            tags=["anti-pattern", "god-object", "monolith", "behavior"])

        ns.define("shotgun-surgery",
            "A single change requires modifications in many unrelated places — scattered responsibility",
            description="Changing the data format means editing: the database schema, the API layer, the client, the tests, the documentation, and the monitoring dashboard. Six files, six concerns, one change. Shotgun surgery indicates poor separation of concerns. In the fleet: if changing the Confidence type requires editing 50 crates, that's shotgun surgery. cuda-equipment centralizes shared types to prevent shotgun surgery — change once, all crates get the update.",
            level=Level.BEHAVIOR,
            examples=["data format change: edit DB + API + client + tests + docs + monitoring (shotgun surgery)", "fleet: confidence type change affecting 50 files (shotgun surgery)", "price change: update website + app + database + billing + contracts (shotgun surgery)"],
            bridges=["coupling", "scattered-responsibility", "centralization", "maintenance"],
            tags=["anti-pattern", "shotgun", "coupling", "behavior"])

        ns.define("premature-abstraction",
            "Creating shared infrastructure before the pattern is clear — building for a future that doesn't arrive",
            description="Three modules have similar but not identical code. You immediately extract a shared library. Then module 4 needs a slightly different interface. Then module 5 needs a different data format. The 'shared' library becomes a Frankenstein of optional parameters and configuration. Premature: the pattern wasn't clear when you abstracted. In the fleet: build 3 concrete implementations FIRST, then abstract the common pattern. Don't abstract until the pattern is clear.",
            level=Level.BEHAVIOR,
            examples=["shared library after seeing pattern in 3 modules = good timing", "shared library BEFORE seeing the pattern = premature abstraction", "fleet: build 3 concrete agents, then extract shared equipment (not before)"],
            bridges=["abstraction", "timing", "pattern", "premature"],
            antonyms=["mature-abstraction"],
            tags=["anti-pattern", "premature", "abstraction", "behavior"])

        ns.define("cargo-cult",
            "Copying the form of a solution without understanding why it works — superficial imitation",
            description="South Pacific islanders built bamboo control towers and lit fires on runways after WWII, believing it would bring cargo planes. They copied the FORM (towers, fires) without understanding the FUNCTION (radio communication, actual aircraft). In the fleet: copying another fleet's architecture without understanding WHY those design choices were made is cargo-cult. The architecture works in their context because of their constraints, not because it's universally optimal.",
            level=Level.BEHAVIOR,
            examples=["bamboo control towers lit fires expecting cargo planes (cargo cult)", "fleet: copying architecture without understanding why it was chosen (cargo cult)", "standup meetings every day because Google does them (cargo cult)"],
            bridges=["imitation", "understanding", "form-vs-function", "superficial"],
            tags=["anti-pattern", "cargo-cult", "imitation", "behavior"])


    def _load_influence(self):
        ns = self.add_namespace("influence",
            "How one agent's behavior affects another's — without force, through structure and example")

        ns.define("set-the-norm",
            "Establish a behavioral baseline that new agents adopt by default through observation",
            description="The first person in an elevator faces the door. Everyone who enters faces the door. The first person set the norm. No one was told. They observed and conformed. In the fleet: the first agent's deliberation pattern (depth, confidence threshold, evidence quality) sets the norm for subsequent agents. They observe the first agent's behavior and converge to it. Set the norm early because norms resist change.",
            level=Level.BEHAVIOR,
            examples=["elevator: first person faces door, everyone follows (norm set)", "fleet: first agent's deliberation style sets norm for subsequent agents", "office: first team's communication style becomes team culture (norm)"],
            bridges=["norm", "conformity", "baseline", "first-mover"],
            tags=["influence", "norm", "default", "behavior"])

        ns.define("nudge",
            "A small structural change that steers behavior without restricting choice",
            description="Put healthy food at eye level in a cafeteria — people choose it more often without being told to. No ban on junk food. No mandate. Just a structural change that makes the desired behavior easier. In the fleet: cuda-filtration's energy budgets NUDGE agents toward efficient communication by making expensive communication hurt. No ban on verbose messages — just a cost that steers toward conciseness.",
            level=Level.PATTERN,
            examples=["cafeteria: healthy food at eye level (nudge toward healthy eating)", "fleet: energy costs for verbose messages (nudge toward conciseness)", "default 401k contribution: opt-out not opt-in (nudge toward saving)"],
            bridges=["choice-architecture", "structural", "steer", "subtle"],
            tags=["influence", "nudge", "structural", "pattern"])

        ns.define("information-cascade",
            "Individuals ignore their private information and follow the crowd, amplifying possibly wrong signals",
            description="Restaurant A has a long line, Restaurant B is empty. You join A's line even though you initially preferred B. The next person sees two restaurants with lines (you joined A) and also joins A. Soon everyone's at A, not because A is better, but because everyone followed the crowd. In the fleet: cuda-consensus's weighted voting must resist information cascades — agents shouldn't just vote for the popular option, they should vote based on their own evidence.",
            level=Level.BEHAVIOR,
            examples=["restaurant line: join A because others did, not because it's better", "fleet: agents voting for popular option regardless of own evidence (cascade)", "stock market: buying because others buy, not because of analysis (cascade)"],
            bridges=["conformity", "herd-behavior", "amplification", "crowd"],
            tags=["influence", "cascade", "herd", "behavior"])

    def _load_negotiation(self):
        ns = self.add_namespace("negotiation",
            "How agents reach agreements through structured interaction")

        ns.define("best-alternative",
            "The most attractive option available if the current negotiation fails — your walk-away point",
            description="If this job offer falls through, your best alternative is another offer at 90% of the salary. BATNA (Best Alternative To Negotiated Agreement) determines your leverage. Strong BATNA = negotiate from strength. Weak BATNA = accept what's offered. In the fleet: each agent's best alternative to the current deliberation proposal is its instinct-based default action. If deliberation produces a better outcome than the instinct default, accept. Otherwise, walk away.",
            level=Level.DOMAIN,
            examples=["job negotiation: best alternative offer determines your leverage", "fleet: instinct-based action is the agent's BATNA (walk-away point)", "buying a house: comparable listings are your BATNA (negotiation leverage)"],
            bridges=["negotiation", "alternative", "walk-away", "leverage"],
            tags=["negotiation", "BATNA", "alternative", "domain"])

        ns.define("concession-rhythm",
            "The pattern and rate at which parties make concessions — reveals information and sets expectations",
            description="Negotiation: you offer 100, they counter 60, you offer 90, they counter 65. Your concession sizes (10, 5) signal that you're approaching your limit. Their concession sizes (5) signal they started low. The rhythm reveals information about both parties' true positions. In the fleet: cuda-deliberation's proposal negotiation has a concession rhythm — each round's compromise reveals the proposer's confidence and the reviewer's flexibility.",
            level=Level.PATTERN,
            examples=["price negotiation: concession sizes reveal limits", "fleet: deliberation round compromises reveal confidence levels", "diplomacy: each country's concessions signal their priorities"],
            bridges=["negotiation", "rhythm", "signal", "information"],
            tags=["negotiation", "concession", "rhythm", "pattern"])

        ns.define("package-deal",
            "Bundle multiple issues together so that concessions on one offset gains on another, creating win-win",
            description="Salary negotiation: you can't get $120k. But you CAN get $110k + remote work + extra vacation. The package is BETTER than $120k cash alone because you value flexibility. Package deals work because parties value things differently. In the fleet: cuda-resource's budget allocation IS a package deal — more compute budget but less network budget. Agents value different resources differently, so package deals create Pareto improvements.",
            level=Level.PATTERN,
            examples=["job offer: lower salary + remote + vacation (package better than raw salary)", "fleet: more compute, less network (package deal on resource budgets)", "trade agreement: lower tariffs on X, lower subsidies on Y (package deal)"],
            bridges=["negotiation", "bundling", "win-win", "Pareto"],
            tags=["negotiation", "package", "bundle", "pattern"])

    def _load_capacity(self):
        ns = self.add_namespace("capacity",
            "The limits of what a system can handle and how it responds to being pushed beyond them")

        ns.define("headroom",
            "Unused capacity maintained as a buffer against demand spikes — safety margin",
            description="A bridge rated for 10,000 lbs that handles trucks up to 7,000 lbs has 30% headroom. A server at 60% CPU utilization has 40% headroom. Headroom absorbs unexpected spikes without degradation. No headroom = any spike causes failure. In the fleet: cuda-resource's allocation should maintain headroom — don't allocate 100% of energy budget. Keep 20% in reserve for unexpected demands (new tasks, emergency deliberation).",
            level=Level.CONCRETE,
            examples=["bridge rated 10,000 lbs handling 7,000 lb trucks (30% headroom)", "server at 60% CPU (40% headroom)", "fleet: 80% energy allocation, 20% headroom for emergencies"],
            bridges=["buffer", "safety-margin", "capacity", "reserve"],
            tags=["capacity", "headroom", "buffer", "concrete"])

        ns.define("saturation",
            "The point where adding more input produces no additional output — the system is maxed out",
            description="A sponge absorbs water until it's full. More water just runs off. The sponge is saturated. A road at rush hour: adding more cars doesn't increase throughput — it increases congestion. In the fleet: an agent processing 100 messages/second is at capacity. 101st message doesn't get processed faster — it gets queued (or dropped). cuda-backpressure detects saturation and signals upstream producers to slow down.",
            level=Level.DOMAIN,
            examples=["saturated sponge: more water runs off (no absorption)", "saturated road: more cars = congestion, not throughput", "fleet agent at message processing limit: more messages = queue/drop"],
            bridges=["capacity", "max", "congestion", "limit"],
            tags=["capacity", "saturation", "limit", "domain"])

        ns.define("elasticity",
            "The ability to expand and contract capacity in response to demand — stretch without breaking",
            description="Cloud computing: auto-scale from 2 servers to 20 when traffic spikes, then back to 2 when it drops. The system ELASTICALLY matches capacity to demand. In the fleet: cuda-actor's spawn hierarchy provides elasticity — new agents spawn when load increases, idle agents terminate when load decreases. Elasticity prevents both waste (over-provisioning) and failure (under-provisioning).",
            level=Level.DOMAIN,
            examples=["cloud: 2→20 servers during spike, back to 2 after (elasticity)", "fleet: spawn agents during load spike, terminate when idle (elasticity)", "rubber band: stretches under tension, returns when released (elasticity)"],
            bridges=["scaling", "auto-scale", "responsive", "capacity"],
            tags=["capacity", "elasticity", "auto-scale", "domain"])

    def _load_decision_patterns(self):
        ns = self.add_namespace("decision-patterns",
            "Structural patterns for how decisions are made, evaluated, and reversed")

        ns.define("reversible-decision",
            "A choice that can be undone at low cost — make it fast, don't deliberate",
            description="Which font to use. Which color scheme. Which meeting time. If you get it wrong, you just change it. Cost of reversal: low. Cost of deliberation: time you'll never get back. For reversible decisions, decide fast, iterate. In the fleet: switching communication partners (cuda-a2a) is reversible — low cost to change, no need for deep deliberation. Switching trust thresholds is NOT reversible (affects all future interactions).",
            level=Level.PATTERN,
            examples=["font choice: reversible (just change it later)", "fleet: switching comm partner = reversible (low cost)", "marriage: irreversible (high deliberation warranted)"],
            bridges=["decision", "reversibility", "speed", "cost"],
            antonyms=["irreversible-decision"],
            tags=["decision", "reversible", "fast", "pattern"])

        ns.define("irreversible-decision",
            "A choice with high reversal cost — deliberate carefully, get it right",
            description="Selling your house. Quitting your job. Publishing a crate's public API. These decisions are expensive to reverse. Deliberation cost is justified because reversal cost is enormous. In the fleet: cuda-equipment's API is an irreversible decision (once published to crates.io). Changing it breaks all dependents. cuda-genepool's fitness function is irreversible in practice (changing it reshapes the entire gene pool).",
            level=Level.PATTERN,
            examples=["selling house: irreversible, high reversal cost", "fleet: published crate API = irreversible (affects all dependents)", "getting a tattoo: irreversible, deliberate first"],
            bridges=["decision", "irreversibility", "deliberation", "cost"],
            antonyms=["reversible-decision"],
            tags=["decision", "irreversible", "deliberate", "pattern"])

        ns.define("satisfice",
            "Choose the first option that meets minimum requirements rather than searching for the optimal one",
            description="Simon: bounded rationality means you CAN'T find the optimal solution (too many options, too little time). Instead, set a threshold and accept the first option that clears it. Good enough > perfect. In the fleet: cuda-deliberation's consensus threshold IS satisficing — you don't find the PERFECT proposal, you accept the first one above 0.85 confidence. Searching longer for better proposals costs energy that might not be recovered.",
            level=Level.DOMAIN,
            examples=["finding a restaurant: first one above 4 stars (satisfice), not exhaustively checking all", "fleet: accept first proposal above confidence threshold (satisfice)", "hiring: first candidate above minimum requirements (satisfice vs optimize)"],
            bridges=["bounded-rationality", "threshold", "good-enough", "practical"],
            antonyms=["maximize"],
            tags=["decision", "satisfice", "threshold", "domain"])

        ns.define("two-door-problem",
            "Choosing between two good options with incomplete information — neither is clearly better",
            description="Two job offers: one with higher salary, one with better culture. You can't quantify culture. You don't know if the higher-salary job will have a bad manager. Incomplete information makes optimal choice impossible. In the fleet: two deliberation proposals both above threshold, both with similar confidence. Neither clearly better. Solution: pick one (satisfice), implement it, monitor, switch if evidence supports the other (reversible decision).",
            level=Level.BEHAVIOR,
            examples=["two good job offers, can't determine which is better (two-door)", "fleet: two good proposals, neither clearly superior (two-door)", "two houses, both nice, neither perfect (two-door)"],
            bridges=["uncertainty", "satisfice", "choice", "ambiguity"],
            tags=["decision", "two-door", "choice", "behavior"])

    def _load_maintenance(self):
        ns = self.add_namespace("maintenance",
            "Keeping systems running and improving over time — the unglamorous essential work")

        ns.define("rot-prevention",
            "Regular small interventions that prevent accumulated degradation from causing failure",
            description="Wood rots without treatment. Paint it every 5 years: rot prevented. Software rots without maintenance: dependencies get stale, APIs change, tests break. Regular updates prevent rot. In the fleet: regular gene pool maintenance (cuda-genepool) prevents rot — quarantining bad genes, updating fitness scores, removing duplicates. Without maintenance, the gene pool rots: bad genes accumulate, fitness scores go stale.",
            level=Level.PATTERN,
            examples=["wood painting every 5 years prevents rot", "software dependency updates prevent rot", "fleet gene pool maintenance prevents accumulation of bad genes"],
            bridges=["prevention", "maintenance", "regular", "degradation"],
            tags=["maintenance", "rot", "prevention", "pattern"])

        ns.define("watchdog",
            "An independent monitor that alerts when a system deviates from expected behavior",
            description="A heart monitor watches for arrhythmia. A smoke detector watches for fire. A watchdog doesn't FIX the problem — it ALERTS. The fix is separate. In the fleet: cuda-metrics' HealthCheck is a watchdog — it monitors agent health and alerts when performance degrades. The fleet coordinator (cuda-fleet-mesh) is alerted but doesn't automatically intervene — the watchdog's job is detection, not correction.",
            level=Level.CONCRETE,
            examples=["heart monitor: detects arrhythmia, alerts (doesn't fix)", "smoke detector: detects fire, alerts (doesn't extinguish)", "fleet health check: detects degradation, alerts (doesn't fix)"],
            bridges=["monitoring", "alert", "detection", "independent"],
            tags=["maintenance", "watchdog", "monitor", "concrete"])

        ns.define("graceful-aging",
            "A system designed to remain useful even as it accumulates technical debt and falls behind current best practices",
            description="An old building with good bones: maybe the wiring is outdated, but the structure is sound. It can be renovated incrementally without tearing it down. Graceful aging: the system doesn't become USELESS when it's no longer state-of-the-art. It remains functional, just not optimal. In the fleet: cuda-* crates should age gracefully — an older crate version still works even if newer crates have better patterns. Don't force-break old versions.",
            level=Level.PATTERN,
            examples=["old building with good bones: renovate incrementally (graceful aging)", "Python 2: didn't age gracefully (forced deprecation)", "fleet crate v0.1: still works even if v0.2 exists (graceful aging)"],
            bridges=["backward-compatibility", "longevity", "maintenance", "incremental"],
            tags=["maintenance", "aging", "graceful", "pattern"])

    def _load_ux(self):
        ns = self.add_namespace("ux-patterns",
            "How systems present themselves to users and agents for maximum usability")

        ns.define("progressive-disclosure",
            "Show only what's needed now, reveal more complexity as the user goes deeper",
            description="Google's homepage: one search box. Advanced search: hidden behind a link. You don't see 50 options until you ask for them. Progressive disclosure prevents overwhelming new users while empowering advanced users. In the fleet: cuda-config's layered config (defaults → file → env → CLI) IS progressive disclosure — defaults work out of the box, advanced users can override everything.",
            level=Level.PATTERN,
            examples=["Google: one search box, advanced options behind link", "fleet config: defaults work immediately, advanced overrides available", "camera app: auto mode by default, manual mode behind a menu"],
            bridges=["simplicity", "layered", "onboarding", "complexity"],
            tags=["ux", "progressive", "disclosure", "pattern"])

        ns.define("affordance",
            "A design property that suggests how to interact with an object — the handle says 'pull'",
            description="A door with a flat plate affords pushing. A door with a handle affords pulling. A button affords clicking. Affordances make interfaces intuitive — the user KNOWS what to do without instructions. In the fleet: vessel.json affords reading capabilities — the structure itself tells you what the vessel can do. cuda-equipment's EquipmentRegistry affords querying equipment — the method names suggest the interactions.",
            level=Level.PATTERN,
            examples=["door handle: affords pulling (you know to pull without instruction)", "button: affords clicking", "vessel.json structure affords reading capabilities (self-documenting)"],
            bridges=["intuitive", "self-documenting", "design", "interaction"],
            tags=["ux", "affordance", "intuitive", "pattern"])

        ns.define("error-recovery",
            "When something goes wrong, show the user what happened AND how to fix it, not just what went wrong",
            description="Bad: 'Error 404'. Good: 'The page you're looking for was moved. Here are similar pages you might want.' Bad: 'NullReferenceException'. Good: 'Contact not found. Add a contact first, then try again.' Error recovery turns frustration into action. In the fleet: cuda-deliberation's 'inconclusive' result should include EVIDENCE of what was considered and SUGGESTIONS for what additional information might help reach a conclusion.",
            level=Level.PATTERN,
            examples=["bad: 'Error 404'. good: 'Page moved, here are alternatives'", "fleet: inconclusive deliberation includes evidence + suggestions for next steps", "compiler: 'unexpected token' WITH caret showing exact location (error recovery)"],
            bridges=["error-handling", "usability", "actionable", "recovery"],
            tags=["ux", "error-recovery", "actionable", "pattern"])


    def _load_trade_patterns(self):
        ns = self.add_namespace("trade-patterns",
            "Exchange dynamics — what agents give up to get what they need")

        ns.define("barter",
            "Direct exchange of capabilities between agents without a common currency",
            description="You have perception skills, I have navigation skills. We exchange: I navigate, you perceive. No energy currency needed — direct skill-for-skill trade. In the fleet: cuda-a2a's negotiation enables barter — agent A provides sensor data, agent B provides computation, both benefit without energy transfer. Barter works when both parties have complementary capabilities and mutual needs.",
            level=Level.PATTERN,
            examples=["perception agent trades data for computation agent's analysis", "humans: barter services without money", "fleet: sensor data for computation results (direct barter)"],
            bridges=["exchange", "complementarity", "mutual-benefit", "direct"],
            tags=["trade", "barter", "exchange", "pattern"])

        ns.define("spot-price",
            "The current market-determined price based on immediate supply and demand — no contracts, no negotiation",
            description="Uber surge pricing: price reflects current demand. Electricity spot price: varies by the minute based on supply/demand. No long-term contract, just the price RIGHT NOW. In the fleet: energy costs (cuda-energy's EnergyCosts) ARE spot prices — deliberation costs 2.0 ATP RIGHT NOW based on current energy budget. The price fluctuates with supply (available energy) and demand (pending tasks).",
            level=Level.CONCRETE,
            examples=["Uber surge: price reflects current demand (spot price)", "fleet: deliberation costs 2.0 ATP now, 3.0 when energy is low (spot price)", "electricity: minute-by-minute price based on grid supply/demand"],
            bridges=["price", "dynamic", "supply-demand", "real-time"],
            tags=["trade", "spot-price", "dynamic", "concrete"])

        ns.define("bid-ask-spread",
            "The gap between what buyers will pay and sellers will accept — the cost of making a trade",
            description="Stock market: buyers bid $100, sellers ask $101. Spread = $1. The wider the spread, the harder it is to trade profitably. Tight spread = liquid market. Wide spread = illiquid. In the fleet: the gap between an agent's willingness to accept a task (ask) and the task's energy budget (bid) is the bid-ask spread. If the spread is negative (budget < willingness), the task won't be assigned. Tight spreads enable more task assignments.",
            level=Level.DOMAIN,
            examples=["stock bid $100, ask $101: spread = $1 (cost of trading)", "fleet: task budget 1.5 ATP, agent wants 2.0 ATP: spread = 0.5 (task unassigned)", "real estate: seller wants $500K, buyer offers $475K: spread = $25K"],
            bridges=["market", "liquidity", "trade-cost", "gap"],
            tags=["trade", "spread", "market", "domain"])

    def _load_morphology(self):
        ns = self.add_namespace("morphology-deep",
            "How structures change shape and form — transformation patterns")

        ns.define("phase-transition",
            "A sudden qualitative change in system behavior when a parameter crosses a critical threshold",
            description="Water at 99°C: liquid. Water at 101°C: gas. Small temperature change, completely different behavior. The transition is sudden and qualitative, not gradual. In the fleet: fleet behavior undergoes phase transitions based on agent count and communication density. Below critical mass: isolated agents. Above critical mass: emergent coordination. The transition is sudden — not gradual improvement but qualitative transformation.",
            level=Level.DOMAIN,
            examples=["water: liquid at 99°C, gas at 101°C (phase transition)", "fleet: isolated agents → emergent coordination at critical mass (phase transition)", "magnet: non-magnetic below Curie temp, magnetic above (phase transition)"],
            bridges=["critical-mass", "sudden-change", "qualitative", "threshold"],
            tags=["morphology", "phase", "transition", "domain"])

        ns.define("catalytic-conversion",
            "Transform input into a fundamentally different output through a process that itself doesn't change",
            description="Catalytic converter: toxic gases in, harmless gases out. The converter isn't consumed. Data pipeline: raw sensor data in, actionable intelligence out. The pipeline isn't consumed. In the fleet: cuda-perception transforms raw signals into percepts. cuda-deliberation transforms percepts into decisions. Each stage is catalytic — it transforms without being consumed. The output is a DIFFERENT TYPE than the input.",
            level=Level.PATTERN,
            examples=["catalytic converter: toxic → harmless (converter unchanged)", "fleet: raw signal → percept → decision (each stage transforms without consuming)", "compiler: source code → machine code (compiler unchanged)"],
            bridges=["transformation", "pipeline", "type-change", "catalyst"],
            tags=["morphology", "catalytic", "transform", "pattern"])

        ns.define("self-assembly",
            "Components spontaneously organize into a structured whole without external direction",
            description="Protein folding: amino acids arrange themselves into a functional 3D structure. Crystal growth: atoms arrange into a lattice. No external director telling each atom where to go — local interactions produce global order. In the fleet: cuda-stigmergy enables self-assembly — agents following simple pheromone rules spontaneously form efficient routing networks without a central coordinator directing them.",
            level=Level.DOMAIN,
            examples=["protein folding: amino acids self-assemble into 3D structure", "crystal growth: atoms self-assemble into lattice", "fleet: agents self-assemble into routing networks via stigmergy"],
            bridges=["emergence", "spontaneous", "local-rules", "global-order"],
            tags=["morphology", "self-assembly", "spontaneous", "domain"])

    def _load_risk_patterns(self):
        ns = self.add_namespace("risk-patterns",
            "How systems identify, assess, and manage uncertainty about future outcomes")

        ns.define("black-swan",
            "An event that is extremely rare, has massive impact, and is only predictable in hindsight",
            description="Taleb: black swans (before Australia was discovered, all swans were white — a black swan was unimaginable). 9/11, 2008 financial crisis, COVID — all black swans. Not just unlikely: IMPOSSIBLE to predict by definition (if you could predict it, you'd prevent it). In the fleet: cuda-resilience's chaos monkey injects black swans (random failures) to test whether the system can survive the unpredictable.",
            level=Level.META,
            examples=["2008 financial crisis: unpredictable, massive impact", "COVID: unpredictable, massive impact", "fleet: chaos monkey injects unpredictable failures to test resilience"],
            bridges=["fat-tail", "unpredictable", "catastrophic", "meta"],
            tags=["risk", "black-swan", "unpredictable", "meta"])

        ns.define("expected-value",
            "The probability-weighted average of all possible outcomes — the rational decision metric",
            description="Lottery: 1 in 300 million chance of $1 billion. Expected value = $1B × (1/300M) = $3.33. A $3.33 ticket for a $3.33 expected return. But variance is enormous. Expected value alone doesn't capture risk. In the fleet: each proposal's expected value is confidence × payoff magnitude. A high-confidence, low-payoff proposal (routine task) might have higher expected value than a low-confidence, high-payoff proposal (risky exploration).",
            level=Level.CONCRETE,
            examples=["lottery: $1B × 1/300M = $3.33 expected value", "fleet: confidence × payoff = expected value of proposal", "insurance: premium < expected loss = worth buying"],
            bridges=["probability", "payoff", "rational", "decision"],
            tags=["risk", "expected-value", "probability", "concrete"])

        ns.define("worst-case-budget",
            "Reserve resources specifically for surviving the worst plausible scenario",
            description="A household keeps 3 months of expenses in savings. A data center has backup generators. The budget is ALLOCATED for the worst case — not hoped for, not improvised. In the fleet: cuda-energy should maintain a worst-case budget — enough energy for apoptosis-safe shutdown even if all income sources fail simultaneously. The reserve isn't for normal operations — it's survival insurance.",
            level=Level.CONCRETE,
            examples=["household: 3 months expenses as worst-case budget", "data center: backup generators for worst-case power failure", "fleet: reserved energy for safe shutdown in worst case"],
            bridges=["reserve", "insurance", "worst-case", "survival"],
            tags=["risk", "worst-case", "budget", "concrete"])

        ns.define("tail-hedging",
            "Positioning to benefit from extreme events rather than just surviving them",
            description="Standard hedging: lose less in bad scenarios. Tail hedging: PROFIT from bad scenarios. Buy out-of-the-money puts: they expire worthless in normal times (small cost) but pay enormous in crashes (big gain). In the fleet: maintaining diverse genes (cuda-genepool) IS tail hedging — genes that seem useless now might become critical in a changed environment. The cost is small (energy to maintain). The payoff in a tail event (environment change) is enormous.",
            level=Level.META,
            examples=["out-of-money puts: worthless normally, enormous in crash", "fleet: diverse gene pool (small maintenance cost, enormous payoff in environment change)", "multi-cloud deployment: extra cost normally, lifesaver if one provider fails"],
            bridges=["hedging", "tail-risk", "extreme-event", "profit"],
            tags=["risk", "tail-hedge", "extreme", "meta"])

    def _load_propagation(self):
        ns = self.add_namespace("propagation",
            "How signals, effects, and changes spread through systems")

        ns.define("blast-propagation",
            "A failure or signal that spreads outward from its origin like an explosion",
            description="One server crashes → cascading failures across dependent services. A rumor starts with one person → spreads to thousands. Blast propagation is UNCONTROLLED spread — the system amplifies rather than dampens the signal. In the fleet: cuda-resilience's bulkhead pattern prevents blast propagation — failures are contained. Without bulkheads, one agent failure cascades through the fleet like a blast wave.",
            level=Level.BEHAVIOR,
            examples=["server crash cascading to dependent services (blast propagation)", "rumor spreading exponentially (blast propagation)", "fleet: one agent failure cascading without bulkheads (blast propagation)"],
            bridges=["cascading-failure", "propagation", "amplification", "uncontrolled"],
            tags=["propagation", "blast", "cascade", "behavior"])

        ns.define("dampened-propagation",
            "A signal that weakens as it travels, naturally attenuating to zero at sufficient distance",
            description="A stone dropped in a pond: ripples spread but fade. A sound in a room: audible nearby, inaudible far away. Dampened propagation has a finite reach. In the fleet: cuda-stigmergy's pheromone decay IS dampened propagation — a strong signal fades with distance and time. Only nearby agents (in time AND space) are affected. Dampened propagation prevents fleet-wide noise from single-agent events.",
            level=Level.PATTERN,
            examples=["stone in pond: ripples spread but fade (dampened propagation)", "sound: audible nearby, inaudible far away", "fleet: pheromone signal fades with distance and time (dampened propagation)"],
            bridges=["decay", "attenuation", "finite-reach", "distance"],
            tags=["propagation", "dampened", "decay", "pattern"])

        ns.define("amplification-loop",
            "A feedback path that takes output and feeds it back as input, amplifying the original signal",
            description="Microphone near a speaker: mic picks up speaker output, speaker amplifies, mic picks up louder output → screech. Positive feedback loop: output reinforces input, signal grows exponentially. In the fleet: cuda-emotion's contagion IS an amplification loop — one agent's emotion affects a neighbor, which affects another, exponentially spreading. Without dampening, amplification loops diverge (runaway emotion).",
            level=Level.PATTERN,
            examples=["microphone + speaker feedback loop → screech", "fleet: emotion contagion: one agent → neighbor → fleet (amplification)", "stock market panic: selling triggers more selling (amplification loop)"],
            bridges=["feedback", "positive-feedback", "exponential", "amplification"],
            tags=["propagation", "amplification", "feedback", "pattern"])

        ns.define("signal-grounding",
            "Connecting an abstract signal to a concrete reality so it stops being noise and starts being information",
            description="An electrical circuit needs a ground reference — otherwise the signal has no meaning (voltage relative to what?). Without grounding, signals are just noise. In the fleet: cuda-communication's SharedVocabulary provides grounding — terms are anchored to shared meanings. Without grounding, messages are noise (the receiver doesn't know what the sender means). Grounding IS the shared reference that turns signals into information.",
            level=Level.DOMAIN,
            examples=["electrical ground: gives voltage signals meaning relative to 0V", "fleet: shared vocabulary grounds messages to common meanings", "conversation: shared context grounds words to mutual understanding"],
            bridges=["grounding", "reference", "meaning", "shared"],
            tags=["propagation", "grounding", "reference", "domain"])

    def _load_lifecycle(self):
        ns = self.add_namespace("lifecycle",
            "How entities are born, grow, decline, and are replaced in systems")

        ns.define("birth-threshold",
            "The minimum conditions required for a new entity to be created — the barrier to entry",
            description="A startup needs: a problem, a solution, initial capital, a founding team. Missing any = can't launch. The birth threshold is ALL conditions, not a weighted average. In the fleet: spawning a new agent requires: available energy, clear task assignment, compatible equipment. Missing any = can't spawn. cuda-captain's launch_mission checks birth threshold before creating agents.",
            level=Level.CONCRETE,
            examples=["startup: needs problem + solution + capital + team (birth threshold)", "fleet agent: needs energy + task + equipment (birth threshold)", "new species: needs niche + resources + reproduction (birth threshold)"],
            bridges=["spawn", "creation", "minimum-requirements", "threshold"],
            tags=["lifecycle", "birth", "threshold", "concrete"])

        ns.define("decline-curve",
            "The trajectory of decreasing performance as a system ages or faces deteriorating conditions",
            description="Oil well production: fast initial decline, then slow taper. Human performance: gradual decline after peak. Every system has a decline curve. The question is whether decline is graceful (slow, predictable) or catastrophic (sudden, unexpected). In the fleet: cuda-energy's fitness tracking reveals the decline curve. Steep decline → investigate (possible failure). Gradual decline → normal aging. Apoptosis triggers when decline crosses a threshold.",
            level=Level.DOMAIN,
            examples=["oil well: fast initial decline, slow taper (decline curve)", "human: gradual performance decline after peak", "fleet agent: fitness decline tracked by energy system, triggers apoptosis at threshold"],
            bridges=["decline", "aging", "trajectory", "fitness"],
            tags=["lifecycle", "decline", "curve", "domain"])

        ns.define("succession",
            "The orderly transfer of responsibility from one entity to its replacement",
            description="A CEO retires, a new CEO takes over. The old CEO doesn't just vanish — there's a handoff period. Institutional knowledge transfers, relationships are introduced, the new leader has support. In the fleet: when an agent undergoes apoptosis, its knowledge (genes, trust scores, memory) should be transferred to a successor. cuda-persistence's snapshot/rollback enables succession: save state before death, restore in successor.",
            level=Level.PATTERN,
            examples=["CEO succession: handoff period, knowledge transfer", "fleet: dying agent transfers state to successor via snapshots", "presidential transition: 2-month handoff period"],
            bridges=["replacement", "handoff", "knowledge-transfer", "orderly"],
            tags=["lifecycle", "succession", "handoff", "pattern"])

        ns.define("niches-and-clines",
            "The spatial distribution of fitness across an environment — where different strategies thrive",
            description="A mountain: polar bears at the top (cold niche), bears at mid-elevation (temperate niche), snakes at the bottom (warm niche). A cline is the gradual change in species composition across a gradient. In the fleet: different task environments create niches — high-deliberation tasks favor complex agents, simple-monitoring tasks favor lightweight agents. The fleet should have agents adapted to each niche, not one generalist.",
            level=Level.DOMAIN,
            examples=["mountain: different species at different elevations (niches along gradient)", "fleet: complex agents for hard tasks, lightweight agents for simple tasks (niches)", "market: enterprise customers (high-touch niche) vs consumers (self-service niche)"],
            bridges=["adaptation", "environment", "fitness-landscape", "specialization"],
            tags=["lifecycle", "niche", "gradient", "domain"])

    def _load_incentives(self):
        ns = self.add_namespace("incentives",
            "How reward structures shape behavior — alignment and misalignment of incentives")

        ns.define("perverse-incentive",
            "A reward structure that encourages the opposite of the desired behavior",
            description="Wells Fargo: employees rewarded for opening accounts → opened fake accounts. Teachers evaluated by test scores → taught to the test, not for understanding. The metric drove behavior that served the metric but defeated the purpose. In the fleet: if agents are rewarded for speed (tasks completed/time), they'll skip deliberation. If rewarded for confidence, they'll be overconfident. Incentive design IS the hardest problem — measure what you value, not what's easy to measure.",
            level=Level.BEHAVIOR,
            examples=["Wells Fargo: account-opening quotas → fake accounts (perverse)", "teachers: test score eval → teaching to test (perverse)", "fleet: speed reward → skip deliberation (perverse incentive)"],
            bridges=["misalignment", "gaming", "metric", "behavior"],
            tags=["incentive", "perverse", "misalignment", "behavior"])

        ns.define("skin-in-the-game",
            "When decision-makers bear the consequences of their decisions — alignment through shared risk",
            description="A chef who eats their own cooking. A founder who invests their own money. An agent that spends its own energy on decisions it recommends. Skin-in-the-game creates alignment: if the decision is wrong, you suffer too. In the fleet: each agent's energy budget IS skin-in-the-game — it pays for its own deliberation. If it recommends a bad action, it wastes its own energy. No external budget, no moral hazard.",
            level=Level.PATTERN,
            examples=["chef eats own cooking (skin in the game)", "founder invests own money (skin in the game)", "fleet agent spends own energy on deliberation (skin in the game)"],
            bridges=["alignment", "consequence", "risk-sharing", "accountability"],
            tags=["incentive", "skin-in-game", "alignment", "pattern"])

        ns.define("goodhart-law",
            "When a measure becomes a target, it ceases to be a good measure",
            description="Originally: correlation between money supply and inflation was useful for understanding the economy. When central banks started TARGETING money supply, the correlation broke. Measuring → targeting → gaming → measure is useless. In the fleet: if fitness score becomes the SOLE target, agents will optimize for fitness score rather than actual task performance. Use multiple measures, keep them implicit when possible, rotate them to prevent gaming.",
            level=Level.META,
            examples=["money supply: useful measure, useless target", "fleet: fitness score as sole target → agents game the score", "test scores: useful measure, useless when teachers target them"],
            bridges=["metric", "gaming", "target", "measure"],
            tags=["incentive", "goodhart", "measure", "meta"])


    def _load_action_verbs(self):
        ns = self.add_namespace("action-verbs",
            "High-compression verbs that each encode a complete multi-step operational pattern")

        ns.define("vet",
            "Rapidly assess whether a candidate is viable before investing in full evaluation",
            description="Not evaluate — VET. A 30-second scan that eliminates 90% of options. Check three things: does it compile? Does it have tests? Does the README make sense? 30 seconds, not 30 minutes. Vet before evaluate. In the fleet: cuda-deliberation should vet proposals before full deliberation — quick confidence check on the proposer's track record eliminates obviously weak proposals from expensive deliberation rounds.",
            level=Level.CONCRETE,
            examples=["resume scan: 30 seconds, eliminate 90% before full interview (vet)", "fleet: quick confidence check on proposal before full deliberation (vet)", "startup idea: quick market check before building MVP (vet)"],
            bridges=["pre-filter", "rapid-assessment", "triage", "efficiency"],
            tags=["verb", "vet", "assess", "concrete"])

        ns.define("triage",
            "Rapidly categorize incoming items by urgency and severity to allocate attention where it's most needed",
            description="Emergency room: patient with chest pain → immediate. Patient with cut finger → wait. 30-second assessment per patient, then prioritize. Triage doesn't solve — it RANKS. The ranking determines where limited attention goes first. In the fleet: cuda-filtration's ResourceBudget IS a triage system — incoming tasks are ranked by priority, and energy is allocated to the highest-ranked first.",
            level=Level.CONCRETE,
            examples=["ER: chest pain → immediate, cut finger → wait (triage)", "fleet: incoming tasks ranked by priority, energy allocated to highest (triage)", "inbox: flag urgent, archive spam, respond to important (triage)"],
            bridges=["prioritize", "rank", "urgent", "allocate"],
            tags=["verb", "triage", "prioritize", "concrete"])

        ns.define("shard",
            "Split a large entity into smaller, independently manageable pieces that can be processed in parallel",
            description="A 100GB database split into 10 × 10GB shards. Each shard is independent, can be processed on a different machine. Sharding enables parallel processing of data too large for one machine. In the fleet: a large task (analyze 1000 sensor readings) is sharded into 10 sub-tasks of 100 readings each, assigned to 10 agents in parallel. Sharding turns a sequential bottleneck into parallel throughput.",
            level=Level.CONCRETE,
            examples=["database: 100GB → 10 × 10GB shards (parallel processing)", "fleet: 1000 readings → 10 × 100 reading sub-tasks (parallel agents)", "map: reduce input into chunks for parallel map workers"],
            bridges=["split", "parallel", "partition", "scale"],
            tags=["verb", "shard", "split", "concrete"])

        ns.define("stitch",
            "Combine multiple partial results into a coherent whole — the inverse of shard",
            description="After sharding and parallel processing, stitch the results back together. 10 partial analyses → 1 comprehensive analysis. Stitching requires handling overlaps, resolving conflicts, and ensuring consistency. In the fleet: cuda-fusion's weighted/Bayesian fusion IS stitching — combining multiple sensor readings or agent assessments into one coherent picture. Stitch well and the seams are invisible.",
            level=Level.CONCRETE,
            examples=["sharded results → combined analysis (stitch)", "fleet: multiple sensor readings → fused coherent picture (stitch)", "quilt: separate patches → one blanket (stitch)"],
            bridges=["fusion", "combine", "merge", "consistency"],
            antonyms=["shard"],
            tags=["verb", "stitch", "combine", "concrete"])

        ns.define("bench",
            "Establish a performance baseline by measuring a known workload under controlled conditions",
            description="Before optimizing, bench: run the workload, measure the time, record the number. That's your baseline. After optimization, run the same bench. Compare. If you didn't bench first, you don't know if the optimization helped. In the fleet: each agent should bench its typical task performance before and after gene changes. cuda-metrics provides the timing. Without benching, you're optimizing blind.",
            level=Level.CONCRETE,
            examples=["benchmark: run workload, measure time (bench)", "fleet: measure task performance before/after gene change (bench)", "athlete: timed run before/after training program (bench)"],
            bridges=["baseline", "measure", "performance", "comparison"],
            tags=["verb", "bench", "measure", "concrete"])

        ns.define("mock",
            "Replace a real component with a fake one that has the same interface but controlled behavior",
            description="Test a payment system without charging real cards → mock the payment gateway. Test fleet coordination without real network → mock the A2A protocol. Mocks enable testing in isolation: fast, deterministic, no external dependencies. In the fleet: cuda-equipment's traits enable mocking — implement the Sensor trait with fixed values for testing, swap in real sensors for production.",
            level=Level.CONCRETE,
            examples=["test payment without real charges (mock gateway)", "test fleet without real network (mock A2A)", "fleet: mock sensor trait with fixed values for testing"],
            bridges=["testing", "isolation", "fake", "interface"],
            tags=["verb", "mock", "fake", "concrete"])

        ns.define("hardwire",
            "Permanently connect two components with a direct, dedicated channel — no routing, no discovery, no overhead",
            description="Instead of sending a message through the fleet mesh (routing overhead, discovery, serialization), hardwire two agents with a direct channel. Dedicated, no intermediaries, minimal latency. In the fleet: agents that communicate constantly (perception → deliberation) should be hardwired — direct function calls instead of A2A messages. Reserve the fleet mesh for inter-team communication, not intra-pipeline communication.",
            level=Level.CONCRETE,
            examples=["dedicated line between two offices instead of phone network (hardwire)", "fleet: direct function call between perception and deliberation (hardwire)", "memory bus: direct CPU-RAM connection instead of network storage (hardwire)"],
            bridges=["direct-connection", "low-latency", "dedicated", "bypass"],
            antonyms=["route"],
            tags=["verb", "hardwire", "direct", "concrete"])

        ns.define("throttle",
            "Deliberately slow down a process to prevent overload, resource exhaustion, or rate limit violations",
            description="API rate limit: 100 requests/second. Your code can do 500/second. Throttle to 100. Throttle isn't failure — it's protection. In the fleet: cuda-rate-limit's TokenBucket throttles outbound communication to prevent rate limit violations. Throttling is better than being blocked: self-imposed throttle keeps you under the limit, reactive throttle means you've already exceeded it.",
            level=Level.CONCRETE,
            examples=["API: self-throttle to 100/s to avoid 429 errors", "fleet: throttle outbound messages to stay under rate limits", "highway: speed limit as throttle (prevents accidents from over-speed)"],
            bridges=["rate-limit", "protect", "slow-down", "self-imposed"],
            tags=["verb", "throttle", "slow", "concrete"])

        ns.define("bridge",
            "Create a translation layer between two incompatible systems, enabling them to communicate without modifying either",
            description="A database adapter that translates SQL queries to NoSQL operations. A protocol converter that translates HTTP to gRPC. The bridge doesn't modify either side — it sits between them and translates. In the fleet: cuda-vessel-bridge IS a bridge — it translates between physical sensors/actuators and the agent's abstract decision-making layer. Neither side needs to know about the other's implementation.",
            level=Level.CONCRETE,
            examples=["database adapter: SQL → NoSQL translation (bridge)", "fleet: physical sensors ↔ agent decisions (bridge)", "interpreter: translates between two languages (bridge)"],
            bridges=["translation", "adapter", "intermediary", "compatibility"],
            tags=["verb", "bridge", "translate", "concrete"])

        ns.define("harden",
            "Add defensive measures to a system to make it resistant to attacks, failures, and unexpected inputs",
            description="Input validation, rate limiting, error handling, memory bounds, authentication — hardening is adding ALL the defensive layers that prevent the system from being compromised or crashing under adversarial conditions. In the fleet: cuda-sandbox hardens agent isolation, cuda-compliance hardens policy enforcement, cuda-rbac hardens access control. Hardening is the difference between 'works in development' and 'survives in production'.",
            level=Level.PATTERN,
            examples=["web app: add input validation, rate limit, auth, CSP (harden)", "fleet: add sandbox, compliance, RBAC to agent system (harden)", "house: add locks, alarm, reinforced doors (harden)"],
            bridges=["defense", "production-readiness", "resilience", "robustness"],
            tags=["verb", "harden", "defend", "pattern"])

        ns.define("stress-test",
            "Push a system beyond its normal operating range to find its breaking point and verify graceful degradation",
            description="Load test: 1000 concurrent users. Chaos test: kill random services. Spike test: 10x normal traffic in 10 seconds. Stress-testing reveals weaknesses that normal testing misses. In the fleet: cuda-resilience's chaos monkey stress-tests the fleet by randomly failing agents and verifying that the fleet degrades gracefully rather than collapsing catastrophically.",
            level=Level.CONCRETE,
            examples=["load test: 1000 concurrent users (stress-test)", "chaos test: kill random services (stress-test)", "fleet: chaos monkey randomly fails agents (stress-test)"],
            bridges=["testing", "extreme", "failure", "verification"],
            tags=["verb", "stress-test", "push", "concrete"])

        ns.define("garden",
            "Tend to a system regularly — prune dead parts, water growing parts, remove weeds — continuous maintenance",
            description="A garden isn't built once — it's tended continuously. Prune dead branches. Remove weeds (bad genes). Water growing plants (promising strategies). Rotate crops (gene diversity). In the fleet: the gene pool (cuda-genepool) needs gardening — quarantine bad genes, promote fit genes, maintain diversity, remove duplicates. Gardening is ongoing, not one-time. The best systems are gardens, not buildings.",
            level=Level.BEHAVIOR,
            examples=["garden: prune, water, weed, rotate (continuous maintenance)", "fleet: quarantine bad genes, promote fit ones (gene pool gardening)", "codebase: remove dead code, refactor, update deps (codebase gardening)"],
            bridges=["maintenance", "continuous", "prune", "tend"],
            tags=["verb", "garden", "maintain", "behavior"])

        ns.define("orchestrate",
            "Coordinate multiple independent components to perform a unified workflow without centralizing control",
            description="An orchestra: 50 musicians, each playing their own part, no one telling each note when to play. The conductor sets tempo and dynamics, but each musician plays independently within that framework. In the fleet: cuda-captain orchestrates agents — sets mission parameters and task assignments, but each agent executes independently. Orchestration is coordination WITHOUT micromanagement.",
            level=Level.PATTERN,
            examples=["orchestra: conductor coordinates, musicians play independently", "fleet: captain coordinates, agents execute independently (orchestrate)", "devops: pipeline tool coordinates build, test, deploy stages (orchestrate)"],
            bridges=["coordinate", "conduct", "workflow", "independent"],
            tags=["verb", "orchestrate", "coordinate", "pattern"])

        ns.define("ferment",
            "Allow a system to develop complexity through autonomous internal processes over time, with minimal intervention",
            description="Wine: grapes + yeast + time = complex flavors you can't design directly. Fermentation requires patience and a controlled environment, not active manipulation. In the fleet: cuda-genepool's evolution IS fermentation — genes combine, mutate, and compete over time. The fleet doesn't design optimal strategies directly. It creates the conditions (fitness function, energy budget) and lets strategies ferment into quality.",
            level=Level.META,
            examples=["wine: grapes + yeast + time = complex flavors (ferment)", "fleet: genes evolve over time into strategies (ferment)", "starter: flour + water + time = sourdough culture (ferment)"],
            bridges=["evolution", "autonomous", "patience", "emergence"],
            tags=["verb", "ferment", "evolve", "meta"])

        ns.define("calibrate",
            "Adjust a measurement instrument to match a known reference standard, ensuring accuracy",
            description="A scale reads 101g for a 100g weight — it needs calibration. Adjust it to read 100g. Now it's calibrated against the reference standard. In the fleet: cuda-self-model's capability calibration adjusts self-assessed performance toward actual performance using EMA. Without calibration, agents overestimate or underestimate their capabilities. With calibration, self-assessment matches reality.",
            level=Level.CONCRETE,
            examples=["scale: reads 101g for 100g weight → adjust to 100g (calibrate)", "fleet: self-model adjusts self-assessment to match actual performance (calibrate)", "thermometer: adjust to match known reference temperature (calibrate)"],
            bridges=["accuracy", "reference", "adjustment", "measurement"],
            tags=["verb", "calibrate", "adjust", "concrete"])

        ns.define("route",
            "Determine the optimal path for information or resources to travel from source to destination",
            description="GPS: route from A to B via fastest roads. Internet: route packets via shortest AS path. Fleet mesh: route messages via lowest-latency agent-to-agent path. Routing considers current conditions (congestion, latency, cost) and selects the best path NOW, not just the shortest path on paper. In the fleet: cuda-fleet-mesh's routing selects message paths dynamically based on current network conditions.",
            level=Level.CONCRETE,
            examples=["GPS: fastest route considering current traffic (route)", "internet: shortest AS path (route)", "fleet: lowest-latency message path (route)"],
            bridges=["pathfinding", "optimal", "dynamic", "network"],
            tags=["verb", "route", "path", "concrete"])

        ns.define("thaw",
            "Gradually restore a frozen or quarantined component to active service, testing at each step",
            description="Frozen food: thaw in fridge overnight, not microwave (gradual). Quarantined employee: return to reduced duties first, then full duties. Thaw is the reverse of quarantine: incremental restoration with verification at each step. In the fleet: a quarantined gene (cuda-genepool) can be thawed — tested in sandbox first, then in limited production, then fully restored. Thaw prevents re-introduction of the problem that caused quarantine.",
            level=Level.PATTERN,
            examples=["quarantined employee: reduced duties → full duties (thaw)", "fleet: sandbox → limited production → full (thaw quarantined gene)", "frozen code: unit tests → staging → production (thaw)"],
            bridges=["quarantine", "restore", "incremental", "verification"],
            antonyms=["quarantine"],
            tags=["verb", "thaw", "restore", "pattern"])

        ns.define("snapshot",
            "Capture the complete state of a system at a moment in time, enabling exact restoration later",
            description="Virtual machine snapshot: save all memory, disk, CPU state → restore to EXACTLY this point. Database snapshot: consistent view of all data at one instant. Snapshots enable rollback, debugging, and recovery. In the fleet: cuda-persistence's snapshot captures agent state (energy, memory, genes, trust scores) at a moment in time. On failure, restore from snapshot instead of starting from scratch.",
            level=Level.CONCRETE,
            examples=["VM snapshot: exact state preservation for rollback", "fleet: agent state captured for failure recovery (snapshot)", "photo: visual snapshot of a moment in time"],
            bridges=["backup", "state", "restore", "point-in-time"],
            tags=["verb", "snapshot", "capture", "concrete"])

        ns.define("broadcast",
            "Send a message to all receivers simultaneously without targeting specific recipients",
            description="Radio broadcast: one transmitter, all receivers in range hear it. TV broadcast: one signal, all TVs receive. Broadcast doesn't target — it INUNDATES. Receivers decide whether the message is relevant. In the fleet: cuda-a2a's broadcast sends a message to all fleet agents. Each agent decides whether to act on it based on relevance. Broadcast is efficient for fleet-wide announcements but wasteful if most agents ignore it.",
            level=Level.CONCRETE,
            examples=["radio: one transmitter, all receivers hear (broadcast)", "fleet: one message, all agents receive (broadcast)", "emergency alert: one signal, all phones receive (broadcast)"],
            bridges=["multicast", "inundate", "announce", "all-receivers"],
            tags=["verb", "broadcast", "send-all", "concrete"])

        ns.define("tunnel",
            "Create a direct path through an intermediate layer that would otherwise block or transform the communication",
            description="VPN tunnel: encrypted path through the internet that bypasses firewalls and surveillance. SSH tunnel: direct connection through a bastion host. Tunnels bypass intermediaries. In the fleet: agents that need direct communication despite fleet mesh routing can establish tunnels — direct agent-to-agent connections that bypass the mesh routing layer for lower latency.",
            level=Level.CONCRETE,
            examples=["VPN: encrypted path through internet (tunnel)", "SSH tunnel: direct connection through bastion host (tunnel)", "fleet: direct agent-to-agent connection bypassing mesh routing (tunnel)"],
            bridges=["bypass", "direct", "encrypted", "intermediary"],
            antonyms=["route"],
            tags=["verb", "tunnel", "bypass", "concrete"])

        ns.define("pollinate",
            "Transfer useful patterns or data from one part of the system to another, enabling cross-pollination of ideas",
            description="Bees transfer pollen between flowers → cross-fertilization → genetic diversity. Without pollination, plants self-fertilize and lose diversity. In the fleet: cuda-genepool's gene sharing IS pollination — successful genes from one agent's pool transfer to another agent's pool. Cross-pollination prevents local optima by introducing diversity. Pollinate actively, not just wait for it to happen naturally.",
            level=Level.PATTERN,
            examples=["bees: transfer pollen between plants (pollinate)", "fleet: gene sharing between agent pools (pollinate)", "team rotation: ideas transfer between teams (pollinate)"],
            bridges=["cross-pollination", "diversity", "transfer", "sharing"],
            tags=["verb", "pollinate", "transfer", "pattern"])

        ns.define("graft",
            "Attach a foreign component to an existing system, creating a hybrid that combines both capabilities",
            description="Graft apple branch onto pear tree → one tree produces both apples and pears. The graft takes time to heal but the result is a hybrid with capabilities neither had alone. In the fleet: grafting a new capability (e.g., cuda-attention) onto an existing agent (e.g., cuda-swarm-agent) creates a hybrid agent that can both coordinate AND focus attention. Grafting requires careful interface matching.",
            level=Level.PATTERN,
            examples=["apple branch grafted onto pear tree (hybrid fruit tree)", "fleet: attention capability grafted onto swarm agent (hybrid agent)", "organization: acquired team integrated into existing structure (graft)"],
            bridges=["hybrid", "attach", "integrate", "combine"],
            tags=["verb", "graft", "attach", "pattern"])

        ns.define("reconcile",
            "Resolve differences between two or more sources of truth into a single consistent state",
            description="Bank statement vs checkbook: amounts differ. Reconcile: find the discrepancy, determine which is correct, update to consistent state. Two agents have different trust scores for the same third agent. Reconcile: exchange evidence, converge to agreed trust score. In the fleet: cuda-trust's gossip sharing and cuda-crdt's merge both implement reconciliation — combining multiple perspectives into one truth.",
            level=Level.CONCRETE,
            examples=["bank statement vs checkbook: find discrepancy, resolve (reconcile)", "fleet: agents exchange evidence, converge on agreed trust score (reconcile)", "git merge: resolve conflicts between branches (reconcile)"],
            bridges=["consistency", "merge", "conflict-resolution", "truth"],
            tags=["verb", "reconcile", "resolve", "concrete"])

        ns.define("fortify",
            "Add defensive layers specifically targeting known or anticipated attack vectors",
            description="A castle is built → then fortified with thicker walls, deeper moats, more guards. Fortification is HARDENING against specific threats, not general hardening. In the fleet: cuda-compliance's policy rules ARE fortifications — specific rules against specific threat vectors (no unauthorized resource access, no energy overspending, no communication with untrusted agents). Each rule is a fortification against a specific attack.",
            level=Level.PATTERN,
            examples=["castle: add walls, moats, guards against specific threats (fortify)", "fleet: add compliance rules against specific attack vectors (fortify)", "network: add specific firewall rules for known attack patterns (fortify)"],
            bridges=["harden", "defense", "specific", "threat"],
            tags=["verb", "fortify", "defend", "pattern"])


    def _load_action_verbs_2(self):
        ns = self.add_namespace("action-verbs-2",
            "More high-compression operational verbs for the fleet vocabulary")

        ns.define("drain",
            "Gradually consume or remove a finite resource until it reaches zero or a critical low",
            description="A battery drains over hours. A budget drains as expenses accumulate. Drain is the passive consumption of a finite resource — it's not being actively attacked, just slowly depleted. In the fleet: an agent's energy budget drains during sustained deliberation. cuda-energy's EnergyBudget tracks drain rate. If drain exceeds generation (rest instinct), the agent eventually reaches apoptosis threshold.",
            level=Level.CONCRETE,
            examples=["phone battery: drains from 100% to 0% over day (drain)", "fleet: energy budget drains during sustained deliberation", "bank account: drains as expenses accumulate (drain)"],
            bridges=["depletion", "consumption", "budget", "resource"],
            tags=["verb", "drain", "deplete", "concrete"])

        ns.define("prime",
            "Pre-load a system with initial data or state so it's immediately useful, not cold-starting from zero",
            description="Prime a water pump: fill it with water before starting so it can pump. Prime a cache: pre-populate with hot data before serving requests. Priming eliminates cold-start latency. In the fleet: cuda-cache's warm() primes the cache with predicted accesses. cuda-world-model primes with known object positions. A primed agent starts productive immediately; an unprimed agent needs time to learn.",
            level=Level.CONCRETE,
            examples=["water pump: fill with water before starting (prime)", "cache: pre-populate with hot data (prime)", "fleet: preload agent with known object positions (prime)"],
            bridges=["warm-start", "pre-load", "initial-state", "latency"],
            tags=["verb", "prime", "pre-load", "concrete"])

        ns.define("relay",
            "Pass a message or task through a chain of intermediaries, each forwarding to the next hop",
            description="Olympic torch relay: runner to runner. Email relay: server to server. Each hop receives, processes minimally, and forwards. The relay doesn't CREATE the message — it TRANSPORTS it. In the fleet: cuda-fleet-mesh's message routing can use relay — if agent A can't reach agent C directly, it relays through agent B. Relay enables connectivity across disconnected network segments.",
            level=Level.CONCRETE,
            examples=["Olympic torch: runner to runner (relay)", "email: server to server to destination (relay)", "fleet: A → B → C message routing (relay)"],
            bridges=["routing", "forward", "chain", "hop"],
            tags=["verb", "relay", "forward", "concrete"])

        ns.define("fuse",
            "Merge two or more signals into one combined signal that preserves the essential information of each",
            description="Sensor fusion: combine GPS + accelerometer + gyroscope into one position estimate. Each source has strengths and weaknesses; fusion produces a better estimate than any single source. In the fleet: cuda-fusion's Bayesian fusion combines multiple confidence-weighted assessments. cuda-confidence's harmonic mean fusion combines independent confidence sources. Fuse IS the fleet's fundamental information-combining operation.",
            level=Level.CONCRETE,
            examples=["GPS + accelerometer → fused position (sensor fusion)", "fleet: multiple confidence sources → combined assessment (fusion)", "metal alloy: iron + carbon → steel (material fusion)"],
            bridges=["combine", "merge", "information", "confidence"],
            tags=["verb", "fuse", "merge", "concrete"])

        ns.define("steer",
            "Apply a small directional influence that gradually changes the system's trajectory without forcing it",
            description="A rudder doesn't push the ship forward — it redirects the existing momentum. Small rudder adjustment = gradual course change. Steering is INFLUENCE, not control. In the fleet: subagents steering the agent's direction through subtle prompts. cuda-deliberation's evidence weighting steers proposals toward better outcomes. Steering works WITH momentum, not against it.",
            level=Level.PATTERN,
            examples=["rudder: redirects ship momentum without forcing (steer)", "fleet: evidence weighting steers deliberation toward better proposals", "parent: gentle guidance without commanding (steer)"],
            bridges=["influence", "redirect", "subtle", "momentum"],
            tags=["verb", "steer", "influence", "pattern"])

        ns.define("pinpoint",
            "Identify the exact location or cause of an issue with maximum precision, not just the general area",
            description="Not 'the server is slow' but 'request to /api/users takes 2300ms due to unindexed query on user_preferences table'. Pinpoint goes beyond identifying the subsystem to identifying the exact line of code, exact query, exact parameter. In the fleet: cuda-provenance's audit trail enables pinpointing — each decision can be traced back to the exact proposal, evidence, and agent that produced it.",
            level=Level.CONCRETE,
            examples=["not 'server slow' but '/api/users 2300ms, unindexed query' (pinpoint)", "fleet: trace decision to exact proposal and agent (pinpoint)", "doctor: not 'you're sick' but 'strep throat, bacteria X, antibiotic Y' (pinpoint)"],
            bridges=["precision", "diagnosis", "root-cause", "exact"],
            tags=["verb", "pinpoint", "precise", "concrete"])

        ns.define("absorb",
            "Take in a smaller entity into a larger one, integrating its capabilities without maintaining separate identity",
            description="Company acquires startup: startup's tech is absorbed into the main product. The startup ceases to exist as a separate entity. In the fleet: when an agent is absorbed into a fleet, its capabilities (genes, equipment) are integrated into the fleet's shared resources. The agent identity disappears but its capabilities persist. Absorb is stronger than integrate — it's full assimilation.",
            level=Level.CONCRETE,
            examples=["acquisition: startup tech absorbed into main product", "fleet: agent capabilities absorbed into fleet shared resources", "ocean absorbs river: river ceases to exist, water persists"],
            bridges=["acquire", "assimilate", "integrate", "merge"],
            tags=["verb", "absorb", "assimilate", "concrete"])

        ns.define("prune",
            "Remove dead, redundant, or underperforming components to improve overall system health",
            description="Prune dead branches: tree directs energy to living branches. Prune redundant code: codebase is easier to maintain. Prune underperforming genes: gene pool allocates energy to fit genes. Pruning IS maintenance — regular, deliberate removal of what doesn't serve the system anymore. In the fleet: cuda-genepool's gene quarantine + decay IS pruning — genes below fitness threshold are eventually removed.",
            level=Level.PATTERN,
            examples=["prune dead tree branches: energy to living branches", "prune dead code: easier maintenance", "fleet: quarantine and remove low-fitness genes (prune)"],
            bridges=["remove", "maintenance", "fitness", "health"],
            tags=["verb", "prune", "remove", "pattern"])

        ns.define("graft-replace",
            "Replace a component with a new version by cutting the old one out and grafting the new one in its place",
            description="Replace a diseased heart valve: remove old, graft new one. The surrounding tissue adapts to the graft. In the fleet: replacing a cuda-* crate version is graft-replace — cut the old version out of Cargo.toml, graft the new one in. Dependent code may need minimal adaptation (the surrounding tissue adjusts to the new graft). Graft-replace is surgical upgrade.",
            level=Level.CONCRETE,
            examples=["heart valve replacement: remove old, graft new", "fleet: swap crate version in Cargo.toml (graft-replace)", "organ transplant: remove old organ, graft new one"],
            bridges=["replace", "upgrade", "surgical", "version"],
            tags=["verb", "graft-replace", "swap", "concrete"])

        ns.define("partition",
            "Divide a system into isolated segments that can operate independently, limiting blast radius",
            description="A ship has watertight compartments — if one floods, others stay dry. Partition the system into isolated segments. In the fleet: cuda-resilience's bulkhead IS partition — each agent runs in its own isolated segment. If one fails, the partition prevents the failure from flooding other segments. Partition trades coordination overhead for isolation benefit.",
            level=Level.CONCRETE,
            examples=["ship watertight compartments: one floods, others dry (partition)", "fleet: bulkhead isolation between agents (partition)", "database: partitioned tables for independent scaling (partition)"],
            bridges=["isolation", "bulkhead", "segment", "blast-radius"],
            tags=["verb", "partition", "isolate", "concrete"])

        ns.define("nominate",
            "Designate a specific agent for a specific role based on capability matching, not random assignment",
            description="Not 'someone handle this' but 'YOU handle this because your fitness for this task type is highest'. Nomination is capability-based assignment. In the fleet: cuda-captain's best_available() IS nomination — it scans all available agents and nominates the one with the highest fitness for the task. Random assignment wastes capable agents on wrong tasks.",
            level=Level.CONCRETE,
            examples=["team lead nomination: best person for the role", "fleet: best_available() nominates highest-fitness agent for task", "award nomination: best candidate selected based on criteria"],
            bridges=["assign", "capability", "fitness", "selection"],
            tags=["verb", "nominate", "assign", "concrete"])

        ns.define("deputize",
            "Temporarily grant an agent authority or capabilities beyond its normal scope for a specific purpose",
            description="A sheriff deputizes a civilian for a specific emergency. The civilian gains law enforcement authority temporarily. Not a permanent promotion — a temporary grant for a specific situation. In the fleet: an agent temporarily granted elevated trust (cuda-rbac) or additional energy budget for an emergency response. Deputization is temporary authority elevation.",
            level=Level.PATTERN,
            examples=["sheriff deputizes civilian for emergency (temporary authority)", "fleet: agent granted elevated trust for emergency response (deputize)", "employee given acting manager role (deputize)"],
            bridges=["temporary", "authority", "elevation", "emergency"],
            tags=["verb", "deputize", "temporary", "pattern"])

        ns.define("delegate",
            "Assign a task to a subordinate agent with the authority to complete it autonomously",
            description="Manager delegates report to employee: 'write the Q3 report, you have full authority on format and content'. The employee decides HOW, the manager decides WHAT. Delegation transfers both responsibility and authority. In the fleet: cuda-captain delegates tasks to agents — the captain defines the mission (what), the agent decides execution (how). Micro-management is the opposite of delegation.",
            level=Level.PATTERN,
            examples=["manager delegates report to employee (what, not how)", "fleet: captain delegates task to agent (mission defined, execution autonomous)", "parent delegates chore to child (task defined, method child's choice)"],
            bridges=["assign", "autonomous", "authority", "responsibility"],
            antonyms=["micromanage"],
            tags=["verb", "delegate", "assign", "pattern"])

        ns.define("quiesce",
            "Gracefully drain all active operations and enter a quiet state without abrupt termination",
            description="Not kill — QUIESCE. A database quiesces: finishes all active transactions, stops accepting new ones, then shuts down cleanly. No data loss. No in-flight request failures. Quiescence is graceful shutdown. In the fleet: cuda-actor's Stop supervision strategy quiesces the agent — finishes current task, saves state, then terminates. No partial work lost.",
            level=Level.CONCRETE,
            examples=["database: finish transactions, stop accepting new, then shut down (quiesce)", "fleet: finish task, save state, terminate cleanly (quiesce)", "factory: finish current production run, then shut down (quiesce)"],
            bridges=["graceful-shutdown", "drain", "clean", "no-abort"],
            antonyms=["kill", "abort"],
            tags=["verb", "quiesce", "shutdown", "concrete"])

        ns.define("silo",
            "Isolate a component or team so completely that no information flows in or out — intentional separation",
            description="Silos are usually bad — they prevent coordination. But INTENTIONAL silos can be useful: security silos (classified information), development silos (prevent premature integration), testing silos (isolate experiments). In the fleet: cuda-sandbox IS an intentional silo — the contained agent can't communicate with fleet agents until released. Silo for safety, not for dysfunction.",
            level=Level.PATTERN,
            examples=["security: classified information in silo (intentional)", "fleet: sandbox isolates untrusted agent (intentional silo)", "development: feature branch isolated from main (intentional silo)"],
            bridges=["isolation", "sandbox", "containment", "intentional"],
            tags=["verb", "silo", "isolate", "pattern"])

        ns.define("referee",
            "Observe interactions between agents and enforce rules without participating in the interaction",
            description="A sports referee watches the game, enforces rules, but doesn't play. The referee's authority comes from impartiality — they're not a participant. In the fleet: cuda-compliance acts as referee — it observes agent behavior, enforces policy rules, but doesn't participate in the agents' tasks. A good referee is invisible when everything is fair — only visible when rules are broken.",
            level=Level.PATTERN,
            examples=["sports referee: watches, enforces rules, doesn't play", "fleet: compliance observes behavior, enforces policy, doesn't participate in tasks", "judge: observes arguments, enforces procedure, doesn't advocate"],
            bridges=["observer", "enforce", "impartial", "rule"],
            tags=["verb", "referee", "enforce", "pattern"])

        ns.define("nurture",
            "Invest resources in a growing entity with the expectation that it will eventually become self-sustaining",
            description="A startup incubator nurtures new companies: provides funding, mentorship, office space. The expectation: companies become self-sustaining and the incubator investment is repaid. In the fleet: the captain (cuda-captain) nurtures new agents — assigns easy tasks first, provides guidance, monitors health. Eventually the agent becomes self-sustaining and no longer needs nurturing. Nurture with the expectation of independence.",
            level=Level.PATTERN,
            examples=["incubator: funds and mentors startups until self-sustaining (nurture)", "fleet: captain assigns easy tasks to new agent, then harder (nurture)", "parent: cares for child until independent (nurture)"],
            bridges=["grow", "invest", "incubate", "eventual-independence"],
            tags=["verb", "nurture", "grow", "pattern"])

        ns.define("excavate",
            "Dig through accumulated layers to find buried but valuable information or patterns",
            description="Archaeological excavation: dig through layers of dirt to find artifacts. Each layer represents a time period. In the fleet: cuda-persistence's snapshots are layers — excavating through snapshots reveals how the agent's state evolved over time. cuda-memory-fabric's episodic memory is layered — excavating through past experiences reveals patterns that weren't visible in real-time.",
            level=Level.CONCRETE,
            examples=["archaeology: dig through dirt layers to find artifacts (excavate)", "fleet: dig through state snapshots to trace evolution (excavate)", "journalism: dig through documents to find buried story (excavate)"],
            bridges=["dig", "layers", "history", "discover"],
            tags=["verb", "excavate", "dig", "concrete"])

        ns.define("synchronize",
            "Bring two or more systems into consistent state by reconciling differences and resolving conflicts",
            description="Clock sync: two clocks showing different times → adjust to same time. Git sync: local repo out of date → fetch + merge. Synchronization requires: identifying differences, determining which is correct (or merging), applying the resolution. In the fleet: cuda-crdt's merge operation IS synchronization — two agents with different states converge to one consistent state through merge.",
            level=Level.CONCRETE,
            examples=["clock sync: adjust two clocks to same time (synchronize)", "git: fetch + merge to sync local and remote (synchronize)", "fleet: CRDT merge brings two agent states into consistency (synchronize)"],
            bridges=["consistency", "merge", "reconcile", "conflict"],
            tags=["verb", "synchronize", "consistency", "concrete"])

        ns.define("templify",
            "Convert a specific solution into a reusable template that works across multiple contexts",
            description="You solved a problem for project A. Extract the pattern into a template that works for projects B, C, and D. Templify is the verb form of 'create a template'. In the fleet: a successful agent configuration can be templified — convert specific parameter values into variables with defaults, making it reusable across different task types. cuda-config's ConfigLayer system enables templification.",
            level=Level.PATTERN,
            examples=["project A solution → reusable template for B, C, D (templify)", "fleet: specific agent config → reusable config template (templify)", "email template: specific email → template with variables (templify)"],
            bridges=["template", "reusable", "generalize", "extract"],
            tags=["verb", "templify", "template", "pattern"])

        ns.define("vaccinate",
            "Expose a system to a weakened version of a threat to build immunity against future stronger attacks",
            description="Vaccine: weakened virus trains immune system. When real virus arrives, immune system destroys it quickly. In the fleet: cuda-resilience's chaos monkey IS vaccination — exposing the fleet to small, controlled failures builds resilience against larger, unexpected failures. Vaccination is proactive hardening based on exposure, not just design analysis.",
            level=Level.PATTERN,
            examples=["vaccine: weakened virus builds immunity (vaccinate)", "fleet: chaos monkey injects small failures to build resilience (vaccinate)", "flu shot: exposure builds antibodies before real flu arrives"],
            bridges=["immunity", "exposure", "proactive", "resilience"],
            tags=["verb", "vaccinate", "immunize", "pattern"])


    def _load_final_verbs(self):
        ns = self.add_namespace("fleet-verbs",
            "The last set of operational verbs completing the 600-term vocabulary")

        ns.define("scaffold",
            "Build a temporary framework that supports construction and is removed when no longer needed",
            description="Building construction: steel scaffold supports workers until the building can support itself. The scaffold is TEMPORARY — remove it when the building stands on its own. In the fleet: the captain (cuda-captain) scaffolds new agents — provides task structure and coordination until agents can self-organize. As agents mature, the scaffold is removed. Scaffold enables construction; it's not the final structure.",
            level=Level.PATTERN,
            examples=["building scaffold: temporary support during construction", "fleet: captain provides temporary task structure for new agents", "training wheels: temporary support until balance learned (scaffold)"],
            bridges=["temporary", "support", "construction", "remove"],
            tags=["verb", "scaffold", "temporary", "pattern"])

        ns.define("recon",
            "Perform a quick, minimal-cost exploration of unknown territory to gather information before committing resources",
            description="Military reconnaissance: send a small scout team to gather intel before committing the full force. Recon is CHEAP exploration — minimal investment, maximal information about the unknown. In the fleet: cuda-filtration's BudgetTiers includes 'scout' mode — lightweight exploration that costs minimal energy but provides enough information to plan the full operation.",
            level=Level.CONCRETE,
            examples=["military scout team: gather intel before full deployment (recon)", "fleet: scout-mode exploration before committing full resources (recon)", "house hunting: drive by neighborhood before scheduling tour (recon)"],
            bridges=["explore", "scout", "cheap", "information"],
            tags=["verb", "recon", "scout", "concrete"])

        ns.define("siege",
            "Sustained pressure on a target that gradually depletes its resources until it yields or collapses",
            description="Castle siege: surround, cut off supplies, wait. Not a direct attack — sustained pressure. The castle's resources (food, water, morale) deplete over time. In the fleet: sustained low-intensity adversarial input (cuda-adversarial-red-team) sieges an agent — not a single attack but continuous probing that depletes energy and tests resilience over time.",
            level=Level.PATTERN,
            examples=["castle siege: surround, cut supplies, wait for depletion", "fleet: sustained adversarial probing depletes agent energy over time (siege)", "legal: persistent lawsuits drain defendant resources (siege)"],
            bridges=["sustained", "pressure", "deplete", "gradual"],
            tags=["verb", "siege", "sustained", "pattern"])

        ns.define("reconstitute",
            "Rebuild a system from its preserved components after a disruption or failure",
            description="Freeze-dried food: add water, it returns to (approximately) its original state. Reconstitute: take saved components (snapshot, genes, config) and rebuild the system. Not from scratch — from preserved state. In the fleet: cuda-persistence's rollback IS reconstitution — load a saved snapshot, restore agent state, resume operation. Reconstitution is faster than rebuilding from scratch.",
            level=Level.CONCRETE,
            examples=["freeze-dried: add water → restored food (reconstitute)", "fleet: load snapshot → restore agent state (reconstitute)", "jigsaw puzzle: reassemble from pieces (reconstitute)"],
            bridges=["restore", "rebuild", "snapshot", "recovery"],
            tags=["verb", "reconstitute", "restore", "concrete"])

        ns.define("ping",
            "Send a minimal signal to verify that a system is alive and responsive, measuring round-trip time",
            description="Network ping: send ICMP echo, wait for reply, measure latency. The simplest health check: are you alive? How fast do you respond? In the fleet: cuda-fleet-mesh's ping_all() sends minimal health check messages to all agents. Ping is the cheapest possible health check — one message, one response, confirms alive + measures latency.",
            level=Level.CONCRETE,
            examples=["network ping: ICMP echo, measure latency (ping)", "fleet: health check message to all agents (ping)", "sonar: pulse, listen for echo (ping)"],
            bridges=["health-check", "latency", "minimal", "alive"],
            tags=["verb", "ping", "check", "concrete"])

        ns.define("scuttle",
            "Deliberately and safely destroy a system to prevent it from falling into the wrong hands or causing harm",
            description="Scuttle a ship: open seacocks, let it sink on YOUR terms, not the enemy's. Controlled destruction. In the fleet: cuda-energy's ApoptosisProtocol IS scuttling — when an agent's fitness drops below threshold, it deliberately shuts down on its own terms rather than becoming a zombie agent that wastes fleet resources. Scuttle is controlled, intentional destruction.",
            level=Level.PATTERN,
            examples=["scuttle ship: sink on own terms, not enemy's", "fleet: apoptosis shuts down failing agent on own terms (scuttle)", "burn documents: destroy on your terms before enemy captures them"],
            bridges=["apoptosis", "destroy", "controlled", "intentional"],
            tags=["verb", "scuttle", "destroy", "pattern"])

        ns.define("ratify",
            "Formally approve a decision or action after it has been proposed and reviewed by relevant stakeholders",
            description="Treaty negotiation: draft → review → ratify. Ratification is the formal approval step that turns a proposal into a binding decision. It comes AFTER deliberation. In the fleet: cuda-deliberation's consensus above threshold IS ratification — the proposal has been reviewed by evidence and stakeholders, and when confidence exceeds the threshold, it's ratified (becomes actionable).",
            level=Level.CONCRETE,
            examples=["treaty: negotiate → review → ratify (formal approval)", "fleet: deliberation → evidence → threshold → ratify (actionable decision)", "constitution: draft → debate → ratify (law)"],
            bridges=["approve", "consensus", "formal", "decision"],
            tags=["verb", "ratify", "approve", "concrete"])

        ns.define("embargo",
            "Restrict the flow of resources or information to or from a specific entity as a penalty or protective measure",
            description="Trade embargo: no goods flow to/from the targeted country. Information embargo: classified data is restricted. Embargo is targeted isolation, not general quarantine. In the fleet: cuda-compliance can embargo a misbehaving agent — restrict its communication with fleet agents (not isolation, but selective restriction). Embargo is lighter than quarantine but heavier than rate-limiting.",
            level=Level.PATTERN,
            examples=["trade embargo: restrict goods to/from country", "fleet: restrict misbehaving agent's communication (embargo)", "library: embargo book from being checked out (restricted access)"],
            bridges=["restrict", "sanction", "targeted", "penalty"],
            tags=["verb", "embargo", "restrict", "pattern"])

        ns.define("exfiltrate",
            "Extract data or resources from a system, potentially covertly, without authorization",
            description="Data breach: attacker exfiltrates user records. Insider threat: employee copies proprietary code to USB. Exfiltration is unauthorized extraction. In the fleet: the membrane (cuda-genepool) blocks exfiltration — antibody rules prevent agents from transmitting sensitive internal state to external entities. Compliance rules flag suspicious outbound data patterns.",
            level=Level.CONCRETE,
            examples=["data breach: attacker extracts user records (exfiltrate)", "insider: copies proprietary code to USB (exfiltrate)", "fleet membrane: blocks sensitive state transmission (anti-exfiltration)"],
            bridges=["breach", "unauthorized", "extract", "security"],
            tags=["verb", "exfiltrate", "breach", "concrete"])

        ns.define("infiltrate",
            "Gain access to a system by blending in with legitimate traffic or credentials",
            description="Attacker uses stolen credentials to access the system — looks legitimate from the outside. Infiltration bypasses perimeter defenses by appearing to be authorized. In the fleet: cuda-rbac and membrane prevent infiltration — even with stolen credentials, the agent's behavioral patterns are monitored. An infiltrating agent that behaves differently from normal agents gets flagged by anomaly detection.",
            level=Level.CONCRETE,
            examples=["stolen credentials: attacker looks legitimate (infiltrate)", "trojan horse: malicious code hidden in legitimate package (infiltrate)", "fleet: behavioral monitoring detects anomalous agents (anti-infiltration)"],
            bridges=["breach", "disguise", "credential-theft", "detection"],
            tags=["verb", "infiltrate", "breach", "concrete"])

        ns.define("decommission",
            "Formally retire a system from active service, removing its access and redirecting its responsibilities",
            description="Navy decommissions a ship: remove from active fleet, reassign crew, recycle parts. The ship still EXISTS but doesn't OPERATE. In the fleet: decommissioning an agent means: remove from fleet-mesh discovery, reassign its tasks, save its state (maybe restore later), revoke its access credentials. Decommission is graceful retirement, not abrupt termination.",
            level=Level.CONCRETE,
            examples=["navy ship: retire from active fleet (decommission)", "fleet agent: remove from mesh, reassign tasks, save state (decommission)", "software: EOL product, redirect users, archive code (decommission)"],
            bridges=["retire", "graceful", "reassign", "formal"],
            tags=["verb", "decommission", "retire", "concrete"])

        ns.define("demote",
            "Reduce an agent's authority, capabilities, or priority level in response to poor performance or policy violation",
            description="Military: officer demoted for misconduct. Employee: senior role reduced to junior. Demotion is a punishment that reduces capability, not removal. The entity still functions but at a lower level. In the fleet: an agent that repeatedly violates compliance rules gets demoted — reduced energy budget, lower trust scores, restricted communication. Demotion is the step before decommission.",
            level=Level.PATTERN,
            examples=["officer demoted for misconduct (reduced rank)", "fleet agent: reduced energy budget after compliance violation (demote)", "employee: manager → individual contributor (demote)"],
            bridges=["punish", "reduce", "authority", "warning"],
            tags=["verb", "demote", "reduce", "pattern"])

        ns.define("promote",
            "Increase an agent's authority, capabilities, or priority level in recognition of superior performance",
            description="Employee promoted from engineer to senior engineer. Military officer promoted for bravery. Promotion grants MORE: more authority, more resources, more responsibility. It's earned through demonstrated capability. In the fleet: cuda-rbac can promote an agent — grant higher permissions, larger energy budget, access to more fleet resources — when it consistently demonstrates high fitness and trust scores.",
            level=Level.PATTERN,
            examples=["engineer → senior engineer (promoted for competence)", "fleet agent: granted higher permissions for consistent high fitness (promote)", "officer: promoted for exemplary service (promote)"],
            bridges=["reward", "elevate", "authority", "merit"],
            antonyms=["demote"],
            tags=["verb", "promote", "elevate", "pattern"])

        ns.define("evict",
            "Forcefully remove an entity from a shared resource or space because it's violating rules or overstaying",
            description="Landlord evicts tenant for non-payment. Cache evicts LRU entry when full. Eviction is forceful removal for cause. In the fleet: cuda-cache's evict_lru() removes least-recently-used entries when the cache is full. cuda-lock can evict a holder that exceeds its lease TTL. Eviction is the mechanism that enforces resource limits.",
            level=Level.CONCRETE,
            examples=["landlord evicts tenant for non-payment (evict)", "cache evicts LRU entry when full (evict)", "fleet: cache removes least-recently-used entries (evict)"],
            bridges=["remove", "forceful", "limit", "enforce"],
            tags=["verb", "evict", "remove", "concrete"])

        ns.define("provision",
            "Allocate and configure resources needed by a new or growing entity before it needs them",
            description="Cloud provisioning: create VMs, configure networks, set up storage BEFORE the application starts. Provision ahead of demand. In the fleet: provisioning a new agent means: allocate energy budget, configure communication channels, register with fleet mesh, assign equipment. All BEFORE the agent receives its first task. Provision eliminates cold-start latency.",
            level=Level.CONCRETE,
            examples=["cloud: create VMs, networks, storage before app starts (provision)", "fleet: allocate energy, register mesh, assign equipment before first task (provision)", "event: set up venue, AV, catering before attendees arrive (provision)"],
            bridges=["allocate", "configure", "ahead-of-demand", "setup"],
            tags=["verb", "provision", "allocate", "concrete"])

        ns.define("ring-fence",
            "Isolate a subset of resources or operations with strict boundaries to prevent contamination or cross-subsidization",
            description="A bank ring-fences its retail banking from investment banking: if investment banking fails, retail is protected. The ring-fence creates an impenetrable boundary. In the fleet: cuda-sandbox ring-fences untrusted code from the fleet. cuda-resource can ring-fence a budget: this energy is for THIS task only, no cross-subsidization. Ring-fence is stronger than partition — it has financial/legal/energy guarantees.",
            level=Level.PATTERN,
            examples=["bank: retail ring-fenced from investment banking", "fleet: sandbox ring-fences untrusted code", "budget: task-specific energy allocation that can't be used elsewhere (ring-fence)"],
            bridges=["isolate", "boundary", "protect", "guarantee"],
            tags=["verb", "ring-fence", "isolate", "pattern"])

        ns.define("siphon",
            "Divert resources or information from a main flow into a secondary channel, usually covertly or gradually",
            description="Siphon gas from a tank: slow, gradual diversion through a small tube. The main flow doesn't notice because the diversion is small and gradual. In the fleet: a misbehaving agent might siphon energy from the shared budget — small, gradual energy requests that go unnoticed. cuda-energy's budget tracking detects siphoning by monitoring consumption rates against expected usage patterns.",
            level=Level.BEHAVIOR,
            examples=["siphon gas from tank: slow, gradual diversion", "fleet: agent slowly diverts shared energy for personal use (siphon)", "embezzlement: small, gradual theft from accounts (siphon)"],
            bridges=["divert", "gradual", "covert", "drain"],
            tags=["verb", "siphon", "divert", "behavior"])

        ns.define("bench-press",
            "Stress-test a specific capability under controlled heavy load to measure its maximum capacity",
            description="Bench press: how much can you lift ONCE? Not endurance (how long) but capacity (how much). One-rep max. In the fleet: bench-press a specific agent capability — what's the maximum message throughput before latency degrades? What's the maximum deliberation complexity before timeout? Bench-press finds the ceiling of a specific capability.",
            level=Level.CONCRETE,
            examples=["bench press: maximum weight lifted once (capacity)", "fleet: maximum message throughput before degradation (bench-press)", "bridge: maximum load before structural failure (bench-press)"],
            bridges=["capacity-test", "maximum", "stress", "measure"],
            tags=["verb", "bench-press", "capacity", "concrete"])

        ns.define("cross-pollinate",
            "Transfer successful patterns from one domain or team to another where they haven't been tried",
            description="Toyota's lean manufacturing crossed into software development as agile methodology. The pattern transferred domains. Cross-pollination requires: recognizing the pattern in domain A, abstracting it, applying it in domain B, adapting for domain B's constraints. In the fleet: cuda-genepool's gene sharing IS cross-pollination — a navigation gene from a robotics agent applied to a data processing agent (after adaptation).",
            level=Level.PATTERN,
            examples=["lean manufacturing → agile software (cross-pollination)", "fleet: navigation gene applied to data processing (cross-pollinate)", "baseball analytics → basketball analytics (cross-pollination)"],
            bridges=["transfer", "domain-transfer", "adapt", "innovation"],
            tags=["verb", "cross-pollinate", "transfer", "pattern"])


    def _load_github_native(self):
        ns = self.add_namespace("github-native",
            "Vocabulary for git-native agent operations — where the repo IS the nervous system")

        ns.define("capability-diff",
            "The measurable delta between an agent's capabilities at two points in time, as captured by git diffs",
            description="Not just code changes — capability changes. An agent added 'navigation' capability, lost 'fishing' skill, improved 'deliberation' quality by 15%. The capability-diff is the MEANING of the code diff, not the syntax. In the fleet: cuda-git-agent tracks capability-diffs in every commit. The diff tells you not what code changed, but what the agent can NOW do that it couldn't before. Capability-diffs become the agent's resume.",
            level=Level.CONCRETE,
            examples=["git diff that adds navigation module = capability-diff: +navigation", "agent commit log as resume of capability acquisition", "PR review: 'this capability-diff adds threat detection'"],
            bridges=["git", "capability", "evolution", "measurement"],
            tags=["github", "capability", "evolution", "concrete"])

        ns.define("agentic-diff",
            "A diff format that captures semantic intent alongside code changes — WHY the change, not just WHAT changed",
            description="Standard git diff: '- old line, + new line'. Agentic diff: '- old capability, + new capability, WHY: better edge case handling, TRADEOFF: 3ms latency increase'. The diff IS the deliberation trace. In the fleet: cuda-provenance's decision lineage captures agentic diffs. Every code change carries the deliberation that produced it. Future agents reading the history understand not just what changed but WHY.",
            level=Level.CONCRETE,
            examples=["PR with agentic-diff: shows reasoning, tradeoffs, confidence change", "fleet agent commit: includes deliberation summary with code change", "code review: 'the agentic-diff shows the agent considered 3 alternatives'"],
            bridges=["diff", "intent", "provenance", "deliberation"],
            tags=["github", "diff", "intent", "concrete"])

        ns.define("branch-agon",
            "An algorithm that strategically generates competing branches as adversarial challenges to the production state",
            description="Not just feature branches — ADVERSARIAL branches. Branch-agon spawns a branch that TRIES TO BEAT the current production state. Like a sparring partner for your codebase. In the fleet: cuda-captain's branch-agon creates challenger agents that compete against the current best. The champion stays. The challenger either improves it or gets discarded. Continuous improvement through controlled combat.",
            level=Level.PATTERN,
            examples=["spawn adversarial branch that optimizes for latency vs current accuracy branch", "fleet: challenger agent competes against incumbent for task performance", "A/B testing taken to its logical extreme: every branch is a challenger"],
            bridges=["competition", "branch", "adversarial", "evolution"],
            tags=["github", "branch", "competition", "pattern"])

        ns.define("commit-prophecy",
            "Predicting the future trajectory of a codebase by analyzing patterns in its commit history",
            description="Every commit is a data point about the developer's intent. A series of 'refactor' commits followed by 'add test' commits followed by silence = either polishing for release or losing interest. A spike in 'fix' commits = growing technical debt. In the fleet: cuda-provenance's decision lineage enables commit-prophecy — analyzing agent commit patterns to predict future behavior, capability drift, and potential failures.",
            level=Level.DOMAIN,
            examples=["spike in 'fix' commits = growing technical debt (commit prophecy)", "fleet: agent commit patterns predict capability drift", "startup: commit frequency decline = project stalling"],
            bridges=["prediction", "git", "pattern", "trajectory"],
            tags=["github", "commit", "prediction", "domain"])

        ns.define("code-pheromone",
            "Metadata attached to code that probabilistically attracts future modifications and attention",
            description="Not a TODO comment — a pheromone. A signal that says 'this area is active and fruitful'. Like ants leaving trails to food sources. Code-pheromones are deposited by agents when they make successful changes: the area becomes MORE likely to be modified again. In the fleet: cuda-stigmergy's pheromone system IS code-pheromone — successful mutations deposit pheromones that attract future modifications. Convergence accelerates.",
            level=Level.BEHAVIOR,
            examples=["successful PR leaves pheromone on modified files → future PRs modify same area", "fleet: successful mutation deposits pheromone → attracts future mutations", "hotspot detection: files with most pheromones = areas of active evolution"],
            bridges=["stigmergy", "attraction", "attention", "probability"],
            tags=["github", "pheromone", "stigmergy", "behavior"])

        ns.define("repo-synapse",
            "A git hook or webhook that triggers biological-style learning when specific repo events occur",
            description="When a PR is merged, a repo-synapse fires. When tests fail, a different synapse fires. The repo IS a neural network: commits are signals, branches are pathways, merges are synaptic strengthening. In the fleet: repo-synapses connect git events to fleet learning — a merged PR triggers gene pool update, a failed test triggers quarantine, a reverted commit triggers fitness decrease. The repo becomes the nervous system.",
            level=Level.PATTERN,
            examples=["on PR merge: trigger gene pool update (repo-synapse)", "on test failure: trigger agent quarantine (repo-synapse)", "on revert: trigger fitness decrease (repo-synapse)"],
            bridges=["git", "hook", "learning", "nervous-system"],
            tags=["github", "synapse", "hook", "pattern"])

        ns.define("vessel-gem",
            "A packaged agent snapshot — pre-trained, tested, and ready to plug into any compatible fleet",
            description="Like a gem in Ruby — a self-contained package of capability. Not the whole agent, just a CAPABILITY GEM. Install the navigation gem → agent can navigate. Install the deliberation gem → agent can deliberate. In the fleet: cuda-genepool's genes are vessel-gems — packaged capability patterns that any agent can acquire. The marketplace (skill-cartridge-registry) distributes vessel-gems across fleets.",
            level=Level.CONCRETE,
            examples=["install navigation gem → agent gains navigation capability", "fleet marketplace: browse vessel-gems by capability rating", "cuda-genepool: genes distributed as vessel-gems"],
            bridges=["package", "capability", "marketplace", "plug-in"],
            tags=["github", "gem", "package", "concrete"])

        ns.define("semantic-fork",
            "A fork that translates the conceptual model of the original repo into a different paradigm or domain",
            description="Not a code fork — a CONCEPT fork. Take a fleet coordination system, fork it conceptually into a biological immune system. The structure is preserved (coordination → immune response), the implementation domain changes (distributed computing → biology). In the fleet: cuda-biology IS a semantic fork of fleet coordination — the same structural patterns (discovery → antigen detection, trust → immune memory, quarantine → immune isolation) in a different paradigm.",
            level=Level.META,
            examples=["fleet coordination → immune system (semantic fork)", "game theory → market dynamics (semantic fork)", "git branching → biological gene expression (semantic fork)"],
            bridges=["fork", "translation", "cross-domain", "abstraction"],
            tags=["github", "fork", "concept", "meta"])

        ns.define("seam-merge",
            "Merging agents or codebases from entirely different fleets, organizations, or paradigms",
            description="Regular merge: same repo, same team, same conventions. Seam-merge: different repos, different teams, different conventions, different PURPOSES. The seam is where they don't fit — seam-merge is the work of making them fit. In the fleet: merging a robotics agent with a data analysis agent requires seam-merge — bridging equipment types, message formats, and capability models. The seam is the interface specification.",
            level=Level.PATTERN,
            examples=["merge robotics agent with data analysis agent (seam-merge)", "corporate acquisition: integrate two engineering cultures (seam-merge)", "open source + proprietary: merge community code with internal code (seam-merge)"],
            bridges=["merge", "integration", "interface", "cross-fleet"],
            tags=["github", "merge", "integration", "pattern"])

        ns.define("hot-branch",
            "A live production branch undergoing rapid mutation and evolution without stabilizing into a release",
            description="Not a feature branch (stable before merge) — a hot branch (continuously changing, never frozen). Production runs on the hot branch directly. Changes go live immediately. Like a chef cooking on live TV — no rehearsal, no take-backs. In the fleet: cuda-self-modify's runtime adaptation IS a hot branch — the agent modifies itself in production, no staging. Hot branches enable maximum adaptation speed at maximum risk.",
            level=Level.BEHAVIOR,
            examples=["production deployment directly from hot branch (continuous evolution)", "fleet agent: modifies own code in production (hot-branch behavior)", "live coding: changes go live immediately"],
            bridges=["production", "mutation", "continuous", "risk"],
            tags=["github", "branch", "live", "behavior"])

        ns.define("regression-bounty",
            "A reward system for identifying capability regressions — when an agent gets WORSE at something",
            description="Not bug bounty (find crashes) — REGRESSION bounty (find degradation). When the agent's navigation accuracy drops from 95% to 90%, that's a regression worth catching. The bounty incentivizes monitoring agent QUALITY over time, not just finding bugs. In the fleet: cuda-learning's experience replay includes regression detection — if a previously-solved task starts failing, it's flagged for investigation. Regression-bounties go to whoever identifies the cause.",
            level=Level.PATTERN,
            examples=["bounty: 'navigation accuracy dropped 5% after commit abc123'", "fleet: regression detection in experience replay (regression-bounty)", "not 'it crashes' but 'it got worse at X'"],
            bridges=["regression", "quality", "monitoring", "incentive"],
            tags=["github", "regression", "bounty", "pattern"])

        ns.define("branch-tributary",
            "An auxiliary branch that continuously feeds knowledge into a main branch without ever merging directly",
            description="A tributary river feeds into the main river — the water mixes, but the tributary maintains its own course. Branch-tributaries contribute INSIGHTS, not code. They're read by the main branch agent but never merged. In the fleet: cuda-fleet-mesh's gossip protocol IS branch-tributary — information flows from auxiliary agents into main agents without code changes. The tributary is the information source, not the execution path.",
            level=Level.PATTERN,
            examples=["research branch feeds insights to production branch (tributary)", "fleet: auxiliary agent shares observations without modifying main agent (tributary)", "advisor role: provide input without being on the execution team"],
            bridges=["branch", "information", "auxiliary", "advisory"],
            tags=["github", "branch", "tributary", "pattern"])

        ns.define("orphan-branch",
            "A branch that has diverged so far from the main branch that it can no longer be merged — but may contain breakthrough innovations",
            description="Evolution creates orphans: species that diverge so far they can't interbreed anymore. Orphan-branches can't merge back — but they might have discovered something the main branch hasn't. In the fleet: cuda-genepool tracks orphan genes — genes that are so different from the current agent that they can't be directly integrated, but might contain novel capabilities worth extracting. Don't delete orphans. Study them.",
            level=Level.BEHAVIOR,
            examples=["radical refactor that breaks API compatibility (orphan-branch)", "fleet: novel gene too different to integrate directly (orphan)", "species on isolated island: diverges from mainland (biological orphan)"],
            bridges=["divergence", "innovation", "incompatibility", "novelty"],
            tags=["github", "branch", "orphan", "behavior"])

        ns.define("branch-dendrochronology",
            "Inferring the health and history of a codebase by analyzing the ring patterns of its branching structure",
            description="Tree rings tell you about drought years, fire years, good growing seasons. Branch rings tell you about crunch periods, refactoring waves, abandonment phases. Wide branch gaps = long periods between merges (stalled development). Rapid branch creation = active experimentation. In the fleet: cuda-provenance's decision lineage enables branch-dendrochronology — reading the branching history to assess agent health and development patterns.",
            level=Level.DOMAIN,
            examples=["wide gaps between merges = stalled development (dendrochronology)", "rapid branch creation = active experimentation", "fleet: branch patterns reveal agent development health"],
            bridges=["history", "branch", "health", "pattern"],
            tags=["github", "branch", "history", "domain"])

    def _load_fleet_biology(self):
        ns = self.add_namespace("fleet-biology",
            "Biological metaphors made operational in fleet computing — where biology IS computing")

        ns.define("neurotransmitter-map",
            "A real-time mapping of agent state to neurochemical signals — dopamine for confidence, serotonin for trust, cortisol for stress",
            description="Not a metaphor — an operational protocol. Every agent's internal state is expressible as a neurochemical profile: high dopamine = high confidence, low serotonin = low trust, high cortisol = high stress from resource depletion. The neurotransmitter map enables INTER-agent communication through chemical gradients. In the fleet: cuda-neurotransmitter implements the neurotransmitter map — agents broadcast their neurochemical state and other agents adjust their behavior based on received signals.",
            level=Level.DOMAIN,
            examples=["agent broadcasts high cortisol → neighbors reduce communication load", "fleet: neurotransmitter map enables non-verbal fleet coordination", "biological: stress hormone cortisol triggers fight-or-flight (neurotransmitter map)"],
            bridges=["neurotransmitter", "state", "communication", "gradient"],
            tags=["biology", "neurotransmitter", "state", "domain"])

        ns.define("silicon-respiration",
            "Measuring the computational energy cost of every agent operation as a metabolite-equivalent expenditure",
            description="Breathing consumes oxygen. Thinking consumes glucose. Agent operations consume energy (cuda-energy's ATP budget). Silicon-respiration tracks the EXHAUSTION rate — how fast the agent burns through its energy budget. An agent that deliberates too much is hyperventilating (wasting energy on low-value cognition). An agent that never deliberates is holding its breath (missing important threats). In the fleet: silicon-respiration metrics inform circadian rhythm modulation.",
            level=Level.DOMAIN,
            examples=["agent deliberating 2.0 ATP per decision = high respiration rate", "fleet: silicon-respiration metrics trigger circadian rest periods", "human: heavy exercise = high oxygen consumption (respiration)"],
            bridges=["energy", "metabolism", "budget", "exhaustion"],
            tags=["biology", "respiration", "energy", "domain"])

        ns.define("membrane-selectivity",
            "A security boundary that selectively allows or blocks information passage based on trust and compatibility, like a biological cell membrane",
            description="Not a firewall (blocks everything except rules) — a membrane (passes compatible molecules, blocks incompatible ones). The cell membrane doesn't have a rule list — it has RECEPTORS that match specific molecular shapes. In the fleet: cuda-genepool's membrane (antibody rules) implements membrane-selectivity — genes from trusted agents pass through, genes from unknown agents are quarantined. The membrane LEARNS what to accept based on experience.",
            level=Level.PATTERN,
            examples=["cell membrane: passes compatible molecules, blocks toxins", "fleet membrane: passes genes from trusted agents, quarantines unknown", "biological: immune system learns to distinguish self from non-self (membrane-selectivity)"],
            bridges=["security", "membrane", "trust", "selective"],
            tags=["biology", "membrane", "security", "pattern"])

        ns.define("instinct-fire",
            "A pre-deliberative response that executes in milliseconds based on pattern matching, bypassing the deliberation loop entirely",
            description="You touch a hot stove. Your hand pulls back BEFORE you think about it. That's instinct — fast, automatic, bypassing conscious deliberation. In the fleet: cuda-reflex's reflex arcs ARE instinct-fires — pre-compiled responses to common patterns (low battery → conserve energy, hostile agent → increase distance, failure cascade → trigger bulkhead). Instinct-fires keep the agent alive while deliberation figures out what to do.",
            level=Level.BEHAVIOR,
            examples=["touch hot stove → hand pulls back before thought (instinct-fire)", "fleet: low battery triggers conservation before deliberation (instinct-fire)", "cuda-reflex: pre-compiled responses to common threat patterns"],
            bridges=["instinct", "reflex", "fast-response", "bypass"],
            tags=["biology", "instinct", "reflex", "behavior"])

        ns.define("gene-quarantine",
            "Isolating a gene (capability pattern) from the shared pool when it exhibits harmful behavior, without deleting it permanently",
            description="Not deleting — QUARANTINING. The gene is isolated, studied, and potentially rehabilitated. Quarantine is temporary. In the fleet: cuda-genepool's quarantine system isolates genes that cause capability regressions. The quarantined gene can be thawed later (tested in sandbox → limited production → full restoration) if the conditions that caused harm have changed. Gene-quarantine preserves genetic diversity while protecting the fleet.",
            level=Level.CONCRETE,
            examples=["gene causing regression → quarantined, studied, potentially rehabilitated", "fleet: bad capability pattern isolated from gene pool (quarantine)", "biological: pathogen isolated in quarantine zone"],
            bridges=["quarantine", "gene", "safety", "temporary"],
            tags=["biology", "gene", "quarantine", "concrete"])

        ns.define("epigenetic-memory",
            "Experience-dependent modifications to agent behavior that don't change the underlying genes but alter their expression",
            description="Identical twins with different life experiences develop different traits. The DNA is the same, but gene EXPRESSION differs. Epigenetic marks turn genes on and off based on environment. In the fleet: cuda-energy's EpigeneticMemory modifies instinct weights based on experience — an agent that frequently encounters hostile agents develops stronger threat-response instincts (epigenetic mark on threat-detection gene) without changing the gene itself.",
            level=Level.DOMAIN,
            examples=["twin A stress → epigenetic mark on cortisol gene → different stress response", "fleet: repeated hostile encounters → stronger threat instincts (epigenetic memory)", "diet changes gene expression without changing DNA (epigenetics)"],
            bridges=["epigenetics", "memory", "experience", "expression"],
            tags=["biology", "epigenetic", "memory", "domain"])

        ns.define("cascading-emovance",
            "Emotional signals propagating through a fleet as color-coded priority changes — GREEN means safe, RED means halt",
            description="Not data propagation — EMOTION propagation. One agent detects a threat (RED). Nearby agents receive the RED signal and increase vigilance. Their neighbors see elevated vigilance and also shift toward YELLOW. The emotion cascades through the fleet. In the fleet: cuda-emotion's contagion IS cascading-emovance — one agent's fear/urgency spreads to neighbors through the fleet mesh, enabling coordinated fleet-wide responses without central coordination.",
            level=Level.BEHAVIOR,
            examples=["one agent detects threat → RED → neighbors increase vigilance → YELLOW → cascade", "fleet: emotion contagion spreads urgency without central command", "crowd panic: one person runs → everyone runs (cascading-emovance)"],
            bridges=["emotion", "cascade", "contagion", "color-code"],
            tags=["biology", "emotion", "cascade", "behavior"])

        ns.define("soma-death",
            "Graceful agent termination that recycles its resources back into the fleet — the computational equivalent of apoptosis",
            description="When a cell undergoes apoptosis, it dismantles itself orderly and its components are recycled by neighbors. The cell doesn't just die — it DONATES its parts. In the fleet: cuda-energy's ApoptosisProtocol implements soma-death — the agent saves its state to persistent memory (for succession), releases its equipment back to the fleet pool, transfers its energy budget to neighbors, and terminates cleanly. The fleet benefits from the termination.",
            level=Level.DOMAIN,
            examples=["cell apoptosis: orderly self-dismantling, parts recycled by neighbors", "fleet agent: save state, release equipment, transfer energy, terminate (soma-death)", "cuda-energy: apoptosis recycles resources to fleet"],
            bridges=["apoptosis", "death", "recycle", "graceful"],
            tags=["biology", "apoptosis", "death", "domain"])

        ns.define("mycelial-spread",
            "Agent capability patterns spreading through a fleet via underground connections, like fungal mycelium networks sharing nutrients",
            description="A mushroom you see is the fruiting body. The mycelium NETWORK beneath the surface is enormous — connecting trees, sharing nutrients, transmitting signals. Trees use the mycelium to share resources and chemical warnings. In the fleet: cuda-stigmergy's pheromone network IS mycelial-spread — capabilities spread through the fleet via shared environmental signals, not direct communication. An agent deposits a successful strategy pattern; other agents absorb it through the pheromone field.",
            level=Level.BEHAVIOR,
            examples=["fungal mycelium connects trees, shares nutrients, transmits warnings", "fleet: pheromone network spreads capability patterns without direct messaging", "agent deposits successful pattern → others absorb via shared signals (mycelial-spread)"],
            bridges=["stigmergy", "mycelium", "spread", "underground"],
            tags=["biology", "mycelium", "stigmergy", "behavior"])

    def _load_cognition_deep(self):
        ns = self.add_namespace("cognition-deep",
            "Advanced cognitive patterns for agent reasoning and fleet intelligence")

        ns.define("backcast-sync",
            "Aligning present systems with a clearly envisioned future state by working backwards from the end goal to today's requirements",
            description="Not forecast (project forward from today) — BACKCAST (project backward from the future). Start with the future state you want. What must be true one year before that? Two years before? Today? In the fleet: Reverse Actualization IS backcast-sync — the agent imagines 2046, identifies what must be true in 2044, 2042, ..., 2026, and builds from present to future along the backcast path. The future pulls the present forward.",
            level=Level.META,
            examples=["imagine 2046 state → what must be true in 2044 → ... → what to build today", "fleet: RA imagines future agent capability → backcast to present requirements", "architecture: start with desired end state, work backwards to current constraints"],
            bridges=["RA", "future", "planning", "alignment"],
            tags=["cognition", "backcast", "RA", "meta"])

        ns.define("axiomatic-descent",
            "Knowledge flowing from abstract principles into concrete implementations through progressive specialization",
            description="Platonic ideals → mathematical axioms → algorithms → code → deployed agent. Knowledge DESCENDS from abstract to concrete. Each step adds specificity but loses generality. In the fleet: cuda-platonic defines the ideal forms, cuda-instruction-set defines the axioms, flux-runtime-c implements the algorithms, cuda-self-modify applies to specific tasks. Axiomatic descent is the path from idea to execution.",
            level=Level.META,
            examples=["justice → law → regulation → enforcement code (axiomatic descent)", "fleet: platonic forms → instruction set → VM → agent behavior (descent)", "mathematics: axioms → theorems → algorithms → programs"],
            bridges=["platonic", "axiom", "hierarchy", "abstraction"],
            tags=["cognition", "axiom", "descent", "meta"])

        ns.define("meta-loop-anchor",
            "An external reference point that prevents recursive self-improvement from diverging into unproductive infinite loops",
            description="An agent that improves itself needs a NORTH STAR — not a moving target but a fixed reference. Without an anchor, self-improvement can optimize for the wrong thing, or enter a cycle of endless refinement that never ships. In the fleet: loop-closure's monitoring IS the meta-loop-anchor — external reality checks prevent the fleet from optimizing itself into a corner. The anchor says: 'you're improving, but is the improvement serving the mission?'",
            level=Level.PATTERN,
            examples=["self-improving compiler: anchor = 'must pass existing tests'", "fleet: loop-closure monitors whether improvement serves mission (meta-loop-anchor)", "diet: anchor = 'must maintain blood sugar above X' (prevents optimization into starvation)"],
            bridges=["loop", "anchor", "reality-check", "convergence"],
            tags=["cognition", "loop", "anchor", "pattern"])

        ns.define("epiphany-resonance",
            "A sudden fleet-wide insight that occurs when enough agents independently approach the same breakthrough threshold simultaneously",
            description="Not gradual consensus — SUDDEN SYNCHRONOUS INSIGHT. Like a crowd all realizing the answer at the same moment. When enough agents reach near-epiphany states independently, a resonance threshold triggers and the insight cascades fleet-wide. In the fleet: cuda-emergence's detect_synchronization identifies epiphany-resonance — when multiple agents independently converge on the same solution pattern, the fleet 'understands' it simultaneously. The breakthrough isn't one agent's — it's the fleet's.",
            level=Level.META,
            examples=["crowd simultaneously realizes the answer (epiphany-resonance)", "fleet: multiple agents independently discover same solution → fleet-wide adoption", "scientific: multiple labs independently discover same principle (multiple discovery)"],
            bridges=["emergence", "synchronization", "breakthrough", "resonance"],
            tags=["cognition", "epiphany", "emergence", "meta"])

        ns.define("storyboard-orchestrate",
            "Using expensive models to plan (direct) and cheap models to execute (animate), coordinating cost and quality across the pipeline",
            description="A movie has a director (expensive, strategic) and animators (cheaper, tactical). The director plans the vision; animators execute frame by frame. In the fleet: dead-reckoning-engine IS storyboard-orchestrate — GLM-5.1 directs (plans the approach), GLM-5-turbo animates (executes the steps), local models render (process the data). The director uses 100x more compute but runs 100x less often.",
            level=Level.PATTERN,
            examples=["GLM-5.1 directs strategy, GLM-5-turbo executes tactics, local model processes (storyboard)", "movie: director plans, animators execute (storyboard-orchestrate)", "fleet: expensive model deliberates once, cheap models execute many times"],
            bridges=["cost-optimization", "director", "animator", "pipeline"],
            tags=["cognition", "storyboard", "cost", "pattern"])

        ns.define("spore-cast",
            "Broadcasting a compressed behavioral seed that any compatible agent can absorb to gain a new capability instantly",
            description="A spore is tiny, durable, and contains everything needed to grow into a full organism. A behavior-spore is a compressed capability pattern that any compatible agent can absorb and execute. In the fleet: mycelium-ai's seed mechanism IS spore-casting — one prompt + one seed = exact behavior reproduction. The spore is the SKILL in compressed form. Absorb the spore → gain the skill. Spore-casting enables instant capability transfer across the fleet.",
            level=Level.PATTERN,
            examples=["mycelium AI: one seed + one prompt = exact behavior reproduction (spore-cast)", "fleet: compressed skill pattern broadcast to all agents → instant capability gain", "biological: fungal spores carry genetic info to new locations"],
            bridges=["mycelium", "skill", "broadcast", "instant"],
            tags=["cognition", "spore", "skill", "pattern"])

        ns.define("silicon-hibernate",
            "Freezing a model's weights into mask-locked silicon and switching to a low-power maintenance mode that only responds to wake triggers",
            description="A bear hibernates: metabolism drops to near-zero, body preserves energy, responds only to strong stimuli. Frozen Intelligence hibernates: the model IS the silicon, no energy for computation except when a wake trigger activates the inference path. In the fleet: frozen-intelligence's mask-locked chips ARE silicon-hibernation — the weights are baked into silicon, power consumption is minimal, inference activates only on demand. Hibernation enables always-on intelligence at near-zero cost.",
            level=Level.DOMAIN,
            examples=["bear hibernation: near-zero metabolism, strong stimuli to wake", "frozen intelligence: weights in silicon, minimal power, inference on demand (hibernate)", "fleet: idle agents hibernate, wake on task assignment"],
            bridges=["frozen", "hibernate", "power", "inference"],
            tags=["cognition", "hibernate", "frozen", "domain"])

        ns.define("tile-neurulation",
            "Training sparse attention tiles to become semi-autonomous processing units that self-organize into layers",
            description="Neurulation is the process in embryonic development where the neural tube forms. Tile-neurulation is the process where individual attention tiles learn to self-organize into processing layers. Each tile is a simple unit; together they form a complex attention system. In the fleet: cuda-ghost-tiles' learned sparse attention IS tile-neurulation — tiles that were initially random become specialized processing units (some attend to spatial features, others to temporal patterns, others to semantic relationships).",
            level=Level.DOMAIN,
            examples=["embryo: neural tube forms from simple cells (neurulation)", "fleet: random attention tiles self-organize into specialized processors (tile-neurulation)", "biological: neural crest cells migrate and specialize"],
            bridges=["ghost-tiles", "attention", "neurulation", "self-organize"],
            tags=["cognition", "tile", "neurulation", "domain"])

        ns.define("peristaltic-ingest",
            "Acquiring new capabilities through a wave-like process of assessment → trial → integration → verification",
            description="Peristalsis: the wave-like muscle contraction that moves food through the digestive tract. Each section: receive, process, pass to next. Peristaltic-ingest for capabilities: assess the skill → try it in sandbox → integrate into runtime → verify with tests → move to next skill. Each stage is a contraction. The fleet DIGESTS new capabilities. In the fleet: cuda-skill's proficiency progression IS peristaltic-ingest — skills are acquired in waves: learn → practice → master → next.",
            level=Level.PATTERN,
            examples=["digestion: mouth → stomach → intestine → absorption (peristalsis)", "fleet skill acquisition: assess → trial → integrate → verify (peristaltic-ingest)", "new employee: onboarding waves: learn tool → try project → integrate into team → verify competence"],
            bridges=["skill", "acquisition", "wave", "digest"],
            tags=["cognition", "peristaltic", "skill", "pattern"])

        ns.define("reverse-entropy-plan",
            "Intentionally increasing disorder in the present system to enable cleaner emergence into a more ordered future state",
            description="Sometimes you must break eggs to make an omelet. Sometimes you must INCREASE entropy (disorder) before you can decrease it. A refactor often looks worse before it looks better — you break the monolith into services, and for a while everything is messier. In the fleet: cuda-self-modify's mutation process IS reverse-entropy-plan — mutations increase local disorder (code churn) but enable convergence on better solutions. The plan tolerates temporary disorder for long-term order.",
            level=Level.META,
            examples=["refactor: break monolith → temporary disorder → cleaner architecture (reverse-entropy)", "fleet mutation: code churn → worse temporarily → better solution emerges", "creative process: brainstorm (disorder) → synthesize (order)"],
            bridges=["entropy", "disorder", "emergence", "temporary"],
            tags=["cognition", "entropy", "emergence", "meta"])


    def _load_fleet_interactions(self):
        ns = self.add_namespace("fleet-interactions",
            "Novel patterns from the interaction of fleet subsystems — the whole is greater than the parts")

        ns.define("reflex-deliberation-tradeoff",
            "The dynamic allocation between fast instinctive responses and slow deliberative reasoning based on urgency and energy availability",
            description="Not always deliberate. Not always reflex. The allocation between reflex and deliberation IS the intelligence budget. High urgency + low energy → reflex dominates (fast, cheap, risky). Low urgency + high energy → deliberation dominates (slow, expensive, thorough). In the fleet: cuda-reflex and cuda-deliberation share the energy budget. The tradeoff is set by cuda-energy's circadian rhythm and urgency scoring. An exhausted agent defaults to reflex — sometimes wise (avoid overthinking), sometimes dangerous (miss threat).",
            level=Level.PATTERN,
            examples=["tired driver: reflexes work, deliberation impaired (tradeoff matters)", "fleet: low energy → default to reflex responses", "chess: blitz = reflex, classical = deliberation"],
            bridges=["reflex", "deliberation", "energy", "tradeoff"],
            tags=["fleet", "reflex", "deliberation", "pattern"])

        ns.define("confidence-gating",
            "Using confidence levels as admission gates for downstream processing — low-confidence data is filtered before expensive computation",
            description="Not process everything — GATE it. Confidence below threshold? Skip. Above threshold? Proceed. Confidence gating saves enormous compute by preventing garbage-in-garbage-out propagation. In the fleet: cuda-confidence-cascade IS confidence gating — each processing stage has a confidence threshold below which the signal is discarded, not processed. Low-confidence sensor readings don't trigger deliberation. Low-confidence proposals don't get executed.",
            level=Level.PATTERN,
            examples=["sensor confidence < 0.3 → skip deliberation (confidence gate)", "proposal confidence < 0.5 → don't execute (confidence gate)", "email spam filter: confidence < 0.7 → mark spam (confidence gate)"],
            bridges=["confidence", "filter", "threshold", "efficiency"],
            tags=["fleet", "confidence", "gate", "pattern"])

        ns.define("trust-topology",
            "The network graph of trust relationships between fleet agents, where trust links determine information flow and collaboration patterns",
            description="Trust isn't binary — it's a TOPOLOGY. Agent A trusts B (0.7) and C (0.3). B trusts D (0.9). The trust topology determines how information flows: A trusts B's recommendations highly, C's recommendations barely. Trust topology IS the fleet's social structure. In the fleet: cuda-trust's multi-context profiles create a weighted directed graph. Information propagation follows trust-weighted paths. High-trust clusters form tight coordination groups. Low-trust boundaries isolate unreliable agents.",
            level=Level.DOMAIN,
            examples=["social network: who trusts whom determines information flow (trust topology)", "fleet: trust graph determines which agent's recommendations are weighted heavily", "supply chain: supplier trust scores determine procurement decisions"],
            bridges=["trust", "network", "graph", "social-structure"],
            tags=["fleet", "trust", "topology", "domain"])

        ns.define("narrative-provenance",
            "Explaining the chain of decisions leading to a current state through story-like sequences that are auditable and comprehensible",
            description="Not just a log — a NARRATIVE. 'We detected anomaly X at time T. Agent A investigated with confidence 0.7. Agent B corroborated with evidence E. We escalated to Captain C who decided action D.' The narrative makes the decision chain comprehensible to both humans and agents. In the fleet: cuda-narrative constructs stories from cuda-provenance's decision lineage. The narrative explains WHY decisions were made, not just WHAT was decided. Narratives enable accountability and debugging.",
            level=Level.PATTERN,
            examples=["post-mortem: 'we failed because X caused Y which led to Z' (narrative)", "fleet: narrative from provenance chain explains decision sequence", "court testimony: narrative explaining chain of events"],
            bridges=["narrative", "provenance", "audit", "explainability"],
            tags=["fleet", "narrative", "provenance", "pattern"])

        ns.define("forgetful-foraging",
            "Actively decaying low-value memories while simultaneously retrieving high-value ones, using the same access patterns for both operations",
            description="Every time you recall a memory, you strengthen it (retrieval practice). Every memory you DON'T recall gets weaker (forgetting curve). The SAME access pattern does both: strengthening what's used, decaying what's not. In the fleet: cuda-memory-fabric's forgetting curves ARE forgetful-foraging — each memory access increases the memory's strength, while unaccessed memories decay. The foraging pattern naturally optimizes the memory store toward current relevance.",
            level=Level.BEHAVIOR,
            examples=["study for exam: recall strengthens memory, unrecalled decays (forgetful-foraging)", "fleet: used memories strengthen, unused memories decay", "LSH forest: frequently accessed buckets stay warm, unused cool down"],
            bridges=["memory", "forgetting", "retrieval", "optimization"],
            tags=["fleet", "memory", "forgetting", "behavior"])

        ns.define("goal-convergence",
            "Detecting when a fleet of agents has collectively achieved a decomposed high-level goal through monitoring subgoal satisfaction across all agents",
            description="A mission has 5 subgoals assigned to 5 agents. Agent 1 finishes. Agent 3 finishes. When ALL subgoals are satisfied, the mission converges. Convergence detection isn't about individual agents — it's about the FLEET's collective state. In the fleet: cuda-goal's hierarchical decomposition creates subgoal trees, cuda-convergence monitors satisfaction across all agents. Convergence doesn't mean every subgoal is perfect — it means the fleet as a whole meets the mission criteria.",
            level=Level.CONCRETE,
            examples=["mission: 5 subgoals × 5 agents → all satisfied = convergence", "fleet: convergence detection monitors collective subgoal satisfaction", "git: all PRs merged = branch convergence"],
            bridges=["goal", "convergence", "decomposition", "collective"],
            tags=["fleet", "goal", "convergence", "concrete"])

        ns.define("pheromone-gradient",
            "A spatial field of deposited signals that creates a gradient, guiding agents toward (or away from) locations in solution space",
            description="Ants follow pheromone gradients — stronger signal = more ants walked this way = probably a good path. The gradient ATTRACTS agents toward fruitful areas and REPELS them from dead ends. In the fleet: cuda-stigmergy's pheromone system creates pheromone gradients in task space — successful approaches accumulate pheromones, failed approaches deposit negative pheromones. Agents climb the positive gradient toward better solutions.",
            level=Level.PATTERN,
            examples=["ants follow pheromone gradient to food source", "fleet: successful approaches accumulate pheromones, attract more attempts", "market: price gradient guides buyers toward deals"],
            bridges=["stigmergy", "gradient", "attraction", "spatial"],
            tags=["fleet", "pheromone", "gradient", "pattern"])

        ns.define("model-mosaic",
            "The composite world representation formed by stitching together each agent's specialized local model into a fleet-wide shared understanding",
            description="Each agent has a partial view of the world — its world model is accurate for its domain but incomplete for the whole fleet. The model mosaic is the STITCHED version: agent A's spatial model + agent B's temporal model + agent C's social model = fleet-wide understanding greater than any individual. In the fleet: cuda-world-model's partial representations are fused by cuda-fusion into a model mosaic. Each agent contributes its specialized view, the fleet sees the whole picture.",
            level=Level.DOMAIN,
            examples=["blind men and elephant: each feels one part, mosaic = whole elephant", "fleet: spatial + temporal + social models fused into fleet-wide understanding", "weather: local station data + satellite data + radar = composite forecast"],
            bridges=["world-model", "fusion", "mosaic", "collective"],
            tags=["fleet", "model", "mosaic", "domain"])

        ns.define("curriculum-convergence",
            "Dynamically updating agent learning curricula based on what the fleet collectively knows and doesn't know, closing gaps efficiently",
            description="As the fleet learns, the gaps change. Early on: nobody knows navigation. Later: everyone knows navigation but nobody knows threat detection. The curriculum ADAPTS to fleet-wide knowledge gaps. In the fleet: cuda-learning's curriculum progression tracks fleet-wide skill coverage and dynamically assigns learning objectives to fill gaps. If 80% of agents know X and 20% don't, the curriculum shifts from X to Y. Curriculum-convergence ensures the fleet grows uniformly.",
            level=Level.PATTERN,
            examples=["school: move from algebra to calculus when class masters algebra (curriculum-convergence)", "fleet: shift learning objectives based on fleet-wide skill coverage", "codebase: fix most-reported bugs first (curriculum of bug fixes)"],
            bridges=["curriculum", "learning", "gap-analysis", "collective"],
            tags=["fleet", "curriculum", "convergence", "pattern"])

        ns.define("reflex-reticulation",
            "Chaining primitive reflex behaviors into complex compound responses without invoking deliberation",
            description="A single reflex: detect threat → increase distance. Two chained reflexes: detect threat → increase distance → scan for cover → move to cover. No deliberation needed — the chain is pre-compiled. Reflex reticulation enables complex behaviors at reflex speed. In the fleet: cuda-reflex chains multiple primitive reflexes into compound responses. The chain fires as fast as a single reflex. Only when the chain encounters an unexpected situation does it escalate to deliberation.",
            level=Level.PATTERN,
            examples=["detect threat → flee → find cover → hide (reflex chain, no thinking)", "fleet: reflex chain fires at reflex speed, deliberation only on unexpected input", "piano: practiced piece plays as reflex chain, not note-by-note deliberation"],
            bridges=["reflex", "chain", "compound", "speed"],
            tags=["fleet", "reflex", "chain", "pattern"])

        ns.define("subgoal-scenting",
            "Depositing pheromone-like priority signals on subgoals to guide agent attention toward the fleet's current priorities without explicit messaging",
            description="Instead of broadcasting 'everyone work on subgoal X', deposit a strong scent on subgoal X in the shared tuple space. Agents naturally gravitate toward the strongest scent. No command, no assignment — just environmental signals. In the fleet: cuda-tuple-space combined with cuda-stigmergy implements subgoal-scenting — the captain deposits priority signals on subgoals, agents discover them through tuple space pattern matching and self-assign to the highest-scented subgoals.",
            level=Level.PATTERN,
            examples=["deposit strong scent on subgoal X → agents naturally gravitate toward X", "fleet: captain deposits priority signals → agents self-assign via tuple space", "restaurant: busy kitchen (high scent) attracts more cooks"],
            bridges=["subgoal", "pheromone", "tuple-space", "priority"],
            tags=["fleet", "subgoal", "scent", "pattern"])

        ns.define("deliberative-trust",
            "Assigning higher trust to outputs produced through thorough deliberation with multiple alternatives considered, penalizing hasty conclusions",
            description="Quick answer from reflex → low deliberative trust. Answer from 3-round deliberation with evidence from 5 agents → high deliberative trust. The depth of consideration determines the trustworthiness, not just the correctness. In the fleet: cuda-trust weights trust scores by deliberation depth — a correct answer reached through deliberation is trusted more than a correct guess. Deliberative trust rewards process quality, not just outcomes.",
            level=Level.PATTERN,
            examples=["deliberated answer (3 rounds, 5 agents) > correct guess (deliberative trust)", "fleet: trust score weighted by deliberation depth", "science: peer-reviewed study > anecdote (deliberative trust)"],
            bridges=["trust", "deliberation", "depth", "process"],
            tags=["fleet", "trust", "deliberation", "pattern"])

        ns.define("platonic-pruning",
            "Filtering training data and experiences against ideal type templates to keep only canonical examples that best represent each concept",
            description="Not all examples are equal. Some are canonical (perfect illustrations of the concept). Others are noisy, edge cases, or misleading. Platonic pruning keeps only the canonical examples — the ones that best match the ideal form. In the fleet: cuda-platonic's ideal type templates filter cuda-learning's experience buffer — only experiences that closely match Platonic forms are retained for training. This distills the essence of each competency.",
            level=Level.PATTERN,
            examples=["ML: keep canonical training examples, discard edge cases (platonic pruning)", "fleet: filter experiences against ideal templates → keep canonical examples", "art school: study masterworks (Platonic forms), not student sketches"],
            bridges=["platonic", "pruning", "canonical", "quality"],
            tags=["fleet", "platonic", "pruning", "pattern"])

        ns.define("self-similar-fleet",
            "The recursive application of the same coordination patterns at multiple scales — individual agents, agent teams, and the whole fleet use identical mechanisms",
            description="A single agent: confidence propagation, trust assessment, deliberation. A team of agents: confidence propagation between members, trust between members, team deliberation. The whole fleet: same mechanisms. Self-similarity means the same vocabulary and patterns work at every scale. In the fleet: cuda-equipment's Confidence propagates within an agent (between modules) AND between agents (via A2A). The same mechanism at two scales. Self-similarity reduces complexity — learn the pattern once, apply it everywhere.",
            level=Level.META,
            examples=["agent modules coordinate like agents coordinate like fleets coordinate (self-similar)", "fractal: same pattern at every zoom level", "military: squad tactics mirror platoon tactics mirror battalion tactics"],
            bridges=["fractal", "recursive", "scale", "pattern"],
            tags=["fleet", "self-similar", "fractal", "meta"])

        ns.define("provenance-weave",
            "Appending decision context to every piece of information as it flows through the fleet, creating a fabric of interconnected explanations",
            description="Agent A produces insight X (confidence 0.8, evidence: E1, E2). Agent B uses X to produce Y. Y carries X's provenance PLUS B's own. The fabric GROWS with each interaction, weaving explanations into every data point. In the fleet: cuda-provenance's decision lineage implements provenance-weave — every decision carries the chain of evidence that led to it. Querying any fleet data point reveals its complete provenance.",
            level=Level.PATTERN,
            examples=["Wikipedia: edit history woven into every article (provenance-weave)", "fleet: every data point carries decision chain that produced it", "supply chain: every part carries origin, processing, transport history"],
            bridges=["provenance", "weave", "context", "traceability"],
            tags=["fleet", "provenance", "weave", "pattern"])

        ns.define("energy-arbitrage",
            "Trading compute resources between tasks based on the ratio of expected payoff to energy cost, maximizing value per ATP spent",
            description="Task A: costs 2.0 ATP, expected payoff 5.0. Task B: costs 0.5 ATP, expected payoff 2.0. Task A's value/ATP = 2.5. Task B's value/ATP = 4.0. Energy arbitrage prioritizes Task B. Not just "highest payoff" but "highest payoff per energy spent". In the fleet: cuda-deliberation's utility scoring combined with cuda-energy's budget implements energy arbitrage — proposals are ranked by expected_value / energy_cost, not just expected_value.",
            level=Level.PATTERN,
            examples=["stock trading: risk-adjusted return (energy arbitrage for money)", "fleet: rank proposals by payoff/energy, not just payoff", "restaurant: most profitable dish per ingredient cost (energy arbitrage for food)"],
            bridges=["energy", "arbitrage", "efficiency", "tradeoff"],
            tags=["fleet", "energy", "arbitrage", "pattern"])

        ns.define("backpressure-propagation",
            "When downstream processing can't keep up, signaling upstream to slow down — protecting the system from overload cascades",
            description="A pipeline: producer → buffer → consumer. Consumer is slow. Buffer fills up. Backpressure tells producer: SLOW DOWN. Without backpressure, the buffer overflows and the system crashes. In the fleet: cuda-backpressure's credit-based flow control implements backpressure-propagation — when a downstream agent's queue is full, upstream agents receive credit reduction signals that slow their output. The fleet adapts its throughput to the slowest link.",
            level=Level.CONCRETE,
            examples=["producer → buffer → consumer: consumer slow → buffer full → producer slows (backpressure)", "fleet: downstream queue full → upstream agents reduce output", "traffic jam: cars slow down because cars ahead are slow (backpressure)"],
            bridges=["backpressure", "flow-control", "overload", "adaptation"],
            tags=["fleet", "backpressure", "flow", "concrete"])

        ns.define("tuple-space-match",
            "Anonymous coordination where agents deposit structured data and retrieve matching patterns without knowing each other's identity",
            description="Linda tuple space: OUT (deposit data), RD (read matching data), IN (consume matching data). No addresses, no channels, no identities — just PATTERN MATCHING. Agent A deposits ('task', 'navigation', 0.8). Agent B reads patterns matching ('task', _, _) and finds the navigation task. In the fleet: cuda-tuple-space enables tuple-space-match — agents coordinate without knowing each other. The tuple space IS the coordination medium.",
            level=Level.PATTERN,
            examples=["Linda: OUT('task','navigation',0.8), IN('task',_,_) → match (tuple-space)", "fleet: agent deposits task tuple, another agent reads matching pattern (tuple-space-match)", "bulletin board: post request, someone who can fulfill it reads it"],
            bridges=["tuple-space", "anonymous", "pattern-match", "coordination"],
            tags=["fleet", "tuple", "anonymous", "pattern"])

        ns.define("ghost-guidance",
            "Using invisible attention tiles to subtly steer agent behavior without explicit commands — the dark matter of fleet coordination",
            description="Ghost tiles don't exist in the visible code — they're learned attention patterns that influence which data the agent processes. Ghost guidance is STEERING through attention, not through commands. In the fleet: cuda-ghost-tiles' sparse attention IS ghost guidance — tiles learned from experience make the agent attend to certain patterns and ignore others, subtly directing behavior without explicit rules. The fleet is guided by its own learned attention patterns.",
            level=Level.BEHAVIOR,
            examples=["learned attention: agent naturally attends to important features (ghost guidance)", "fleet: ghost tiles steer behavior without explicit rules", "habits: you naturally look left before crossing street (ghost guidance from experience)"],
            bridges=["ghost-tiles", "attention", "guidance", "invisible"],
            tags=["fleet", "ghost", "guidance", "behavior"])

        ns.define("skill-synergy",
            "When two or more skills combine to produce capability greater than the sum of their individual effects — compound interest for competence",
            description="Navigation + perception = autonomous exploration (greater than either alone). Deliberation + trust = reliable coordination (greater than either alone). Skill synergy means the fleet's total capability is SUPERLINEAR in its skills — each new skill multiplies existing capabilities. In the fleet: cuda-skill's synergy bonuses reward agents for acquiring complementary skill pairs. The synergy map shows which skill combinations produce the greatest compound effect.",
            level=Level.PATTERN,
            examples=["navigation + perception = autonomous exploration (skill synergy)", "deliberation + trust = reliable coordination (skill synergy)", "python + statistics = data science (skill synergy)"],
            bridges=["skill", "synergy", "compound", "superlinear"],
            tags=["fleet", "skill", "synergy", "pattern"])

        ns.define("platonic-attraction",
            "The tendency of agents to evolve toward ideal type templates over time, as Platonic forms exert an attractor force on agent behavior",
            description="Not explicit optimization — ATTRACTION. The Platonic form is like a gravity well: agents naturally drift toward it as they accumulate experience. The form doesn't command; it ATTRACTS. In the fleet: cuda-platonic's ideal templates exert platonic-attraction on agents — as agents gain experience, their behavior naturally converges toward the ideal form for their role. The attraction is gentle but persistent, like evolution toward fitness peaks.",
            level=Level.META,
            examples=["evolution: species drift toward fitness peaks (platonic attraction)", "fleet: agents naturally converge toward ideal behavior templates", "apprentice naturally develops toward master's skill level (platonic attraction)"],
            bridges=["platonic", "attraction", "evolution", "ideal"],
            tags=["fleet", "platonic", "attraction", "meta"])

        ns.define("deliberation-half-life",
            "The rate at which deliberation relevance decays — a decision made 10 minutes ago is less relevant than one made 10 seconds ago",
            description="Not all deliberation is equally valuable. Fresh deliberation (high relevance) informs current decisions. Stale deliberation (low relevance) may mislead. The half-life determines how quickly deliberation value decays. In the fleet: cuda-temporal's deadline urgency combined with cuda-memory-fabric's forgetting curves implements deliberation-half-life — recent deliberation is weighted heavily, old deliberation fades. This prevents the fleet from being anchored to outdated analysis.",
            level=Level.CONCRETE,
            examples=["news: 10-minute-old analysis > 10-day-old analysis (deliberation half-life)", "fleet: recent deliberation weighted more than old", "radioactive decay: fresh sample has more activity (half-life analogy)"],
            bridges=["deliberation", "decay", "temporal", "relevance"],
            tags=["fleet", "deliberation", "half-life", "concrete"])

        ns.define("fleet-immune-response",
            "The collective defensive reaction when a fleet detects an internal or external threat — isolation, analysis, adaptation, and memory formation",
            description="Biological immune response: detect pathogen → isolate → analyze → produce antibodies → remember. Fleet immune response: detect misbehaving agent → quarantine (cuda-sandbox) → analyze provenance (cuda-provenance) → produce compliance rules (cuda-compliance) → remember in trust engine (cuda-trust). The fleet LEARNS from each threat, becoming more resistant over time. In the fleet: the combination of cuda-sandbox + cuda-compliance + cuda-trust implements fleet-immune-response.",
            level=Level.PATTERN,
            examples=["detect pathogen → isolate → analyze → antibodies → memory (immune response)", "fleet: detect bad agent → quarantine → analyze → add rules → update trust", "cybersecurity: detect intrusion → isolate → analyze → patch → add to IDS"],
            bridges=["immune", "defense", "adapt", "memory"],
            tags=["fleet", "immune", "defense", "pattern"])

        ns.define("model-descent-inversion",
            "The point where the algorithm has absorbed so much intelligence that removing the model IMPROVES performance — code eats the model",
            description="At first: algorithm + model. The model provides intelligence the algorithm lacks. Over time: the algorithm LEARNS from the model's patterns and incorporates them. Eventually: the algorithm IS the intelligence, the model is just training data. Model descent inversion: remove the model and the algorithm still works (it absorbed the intelligence). In the fleet: cuda-model-descent's vision is this inversion — agents become smarter not by better models but by absorbing model intelligence into their code.",
            level=Level.META,
            examples=["distillation: small model trained on large model's outputs (model descent)", "fleet: agent absorbs model patterns into deliberation code → doesn't need model", "human: apprentice internalizes master's knowledge → doesn't need master anymore"],
            bridges=["model-descent", "absorption", "inversion", "code-eats-model"],
            tags=["fleet", "model-descent", "inversion", "meta"])

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
