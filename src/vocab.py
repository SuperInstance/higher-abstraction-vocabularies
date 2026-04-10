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
        self._load_network_topology()
        self._load_arch_patterns()
        self._load_quantum_metaphor()
        self._load_ethnobotany_metaphor()
        self._load_systems_dynamics()
        self._load_info_theory()
        self._load_chemical_metaphor()
        self._load_urban_planning()
        self._load_thinking_patterns()
        self._load_milestone_vocab()
        self._load_posthuman_vocab()
        self._load_flux_data_path()
        self._load_flux_control_flow()
        self._load_maritime_vocab()
        self._load_aerospace_vocab()
        self._load_flux_network()
        self._load_flux_concurrency()
        self._load_culinary_vocab()
        self._load_military_vocab()
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
        self._load_decision_quality()
        self._load_coordination_deep()
        self._load_security_deep()
        self._load_adaptation_deep()
        self._load_ontology_deep()
        self._load_power_dynamics()
        self._load_efficiency_frontier()
        self._load_flux_bytecodes()
        self._load_flux_flavors()
        self._load_agent_social()
        self._load_flux_memory()
        self._load_agent_crypto()
        self._load_bio_computing()
        self._load_agent_lifecycle()
        self._load_flux_compound()
        self._load_agent_failure()
        self._load_emergence_patterns()
        self._load_metaphor_vocab()
        self._load_architecture()
        self._load_bio_computing()
        self._load_cognition()
        self._load_creativity()
        self._load_efficiency()
        self._load_fleet_interactions()
        self._load_flux_bytecodes()
        self._load_git_native()
        self._load_hardware()
        self._load_knowledge()
        self._load_learning()
        self._load_repo_mined()
        self._load_cooperative_perception()
        self._load_adversarial_defense()
        self._load_fleet_governance()
        self._load_knowledge_compression()
        self._load_haptic_intelligence()
        self._load_agent_ontogeny()
        self._load_inter_agent_trade()
        self._load_mythological_archetypes()
        self._load_sonar_cognition()
        self._load_metamaterial_cognition()
        self._load_thermal_management()
        self._load_mathematics()
        self._load_meta_cognition()
        self._load_network()
        self._load_neuro_bio()
        self._load_posthuman()
        self._load_reliability()
        self._load_repo_mined()
        self._load_cooperative_perception()
        self._load_adversarial_defense()
        self._load_fleet_governance()
        self._load_knowledge_compression()
        self._load_haptic_intelligence()
        self._load_agent_ontogeny()
        self._load_inter_agent_trade()
        self._load_mythological_archetypes()
        self._load_sonar_cognition()
        self._load_metamaterial_cognition()
        self._load_thermal_management()
        self._load_mathematics()

    def _load_uncertainty(self):
        ns = self.add_namespace("uncertainty",
            "Confidence, trust, belief, and probability — how agents handle not-knowing")

        ns.define("confidence",
            "A 0-1 value representing certainty about a proposition or observation",
            level=Level.DOMAIN,
            examples=["sensor confidence 0.95", "prediction confidence 0.4", "fused confidence after combining two sources"],
            properties={"range": "0.0 to 1.0", "fusion": "harmonic-mean", "unit": "scalar"},
            bridges=["trust", "belief", "probability", "certainty", "information"],
            aliases=["certainty", "sureness", "belief-strength"],
            tags=["core", "fleet-foundation", "propagation"])

        ns.define("harmonic-mean-fusion",
            "Combining independent confidence sources via 1/(1/a + 1/b)",
            level=Level.PATTERN,
            examples=["fusing sensor reading 0.95 with prior 0.7 = 0.804", "fusing 0.9 with 0.1 = 0.09 (not 0.5!)"],
            properties={"formula": "1/(1/a + 1/b)", "penalizes": "uncertainty", "used_in": "cuda-confidence, cuda-fusion, cuda-sensor-agent"},
            bridges=["bayesian-update", "weighted-average", "consensus"],
            tags=["core", "mathematics", "fusion"])

        ns.define("trust",
            "Slowly-accumulating confidence in another agent's reliability",
            level=Level.DOMAIN,
            examples=["trust level 0.7 for pathfinding", "trust drops from 0.8 to 0.2 after failed promise", "gossip: agent shares trust assessments with neighbors"],
            properties={"decay": "exponential", "growth_rate": "1/10 of decay", "per_context": True},
            bridges=["confidence", "reputation", "credit-assignment"],
            aliases=["reliability-belief", "agent-faith"],
            tags=["core", "social", "security"])

        ns.define("bayesian-update",
            "Adjusting beliefs based on new evidence using prior and likelihood",
            level=Level.PATTERN,
            examples=["prior 0.5 + evidence favoring A -> posterior 0.8", "medical diagnosis: symptoms update disease probability"],
            bridges=["harmonic-mean-fusion", "confidence", "learning-rate"],
            tags=["mathematics", "learning", "statistics"])

        ns.define("entropy",
            "Measure of uncertainty or surprise in a distribution",
            level=Level.DOMAIN,
            examples=["uniform distribution = maximum entropy", "coin flip: H(p) = -p*log(p) - (1-p)*log(1-p)", "entropy spike means agent encountered something surprising"],
            bridges=["uncertainty", "surprise", "information", "exploration"],
            tags=["mathematics", "information-theory"])

        ns.define("calibration",
            "How well an agent's confidence matches its actual accuracy",
            level=Level.BEHAVIOR,
            examples=["forecasting: said 80% chance of rain, it rained 80% of those times", "agent says 0.9 confidence, historical accuracy is 0.3 = poorly calibrated"],
            bridges=["confidence", "self-model", "meta-cognition"],
            tags=["agent-behavior", "meta-cognition"])

        ns.define("information",
            "Reduction in uncertainty gained from an observation or message",
            level=Level.DOMAIN,
            examples=["a bit that resolves a coin flip carries 1 bit of information", "redundant message = 0 information", "surprising message = high information"],
            bridges=["entropy", "confidence", "attention", "communication-cost"],
            tags=["information-theory", "communication"])

    def _load_memory(self):
        ns = self.add_namespace("memory",
            "How agents store, retrieve, forget, and consolidate information")

        ns.define("working-memory",
            "Fast, limited-capacity buffer for current task context",
            level=Level.CONCRETE,
            examples=["holding a phone number while dialing", "keeping 3 recent sensor readings in focus", "current goal: navigate to door"],
            properties={"capacity": "4-7 items", "decay": "seconds", "half_life": "~30s"},
            bridges=["attention", "focus", "registers"],
            tags=["memory", "cognition", "fleet"])

        ns.define("episodic-memory",
            "Specific experiences stored with timestamp and emotional valence",
            level=Level.DOMAIN,
            examples=["yesterday I tried path A and it was blocked", "last time I talked to navigator, it gave bad directions", "the time the fleet coordinated perfectly on the warehouse task"],
            properties={"decay": "days", "half_life": "~1 week", "emotional_modulation": True},
            bridges=["semantic-memory", "procedural-memory", "narrative", "learning"],
            tags=["memory", "learning", "fleet"])

        ns.define("semantic-memory",
            "General knowledge extracted from many episodes — the wisdom layer",
            level=Level.DOMAIN,
            examples=["doors in this building are usually on the right wall", "sensor 3 tends to give noisy readings in rain", "collaborative tasks go faster with 3 agents, slower with 5"],
            properties={"decay": "months", "half_life": "~6 months", "source": "episodic-consolidation"},
            bridges=["episodic-memory", "procedural-memory", "world-model", "knowledge"],
            tags=["memory", "learning", "fleet"])

        ns.define("procedural-memory",
            "How to do things — skills, patterns, automatic behaviors",
            level=Level.DOMAIN,
            examples=["knowing how to navigate a familiar building", "automatic collision avoidance reflex", "typing without thinking about key locations"],
            properties={"decay": "years", "half_life": "~5 years", "automation": True},
            bridges=["working-memory", "skill", "reflex", "habit"],
            tags=["memory", "skill", "fleet"])

        ns.define("forgetting-curve",
            "Exponential decay of memory strength over time without rehearsal",
            level=Level.PATTERN,
            examples=["forget 50% of a lecture within 1 hour without notes", "spaced repetition extends the curve", "emotional memories decay slower"],
            properties={"shape": "exponential", "configurable_half_life": True},
            bridges=["memory", "decay", "spaced-repetition", "episodic-memory"],
            tags=["memory", "learning", "psychology"])

        ns.define("consolidation",
            "Transfer from short-term to long-term memory during rest",
            level=Level.BEHAVIOR,
            examples=["studying before sleep improves retention", "taking breaks between learning sessions", "agent rests -> episodes consolidate -> semantic memory grows"],
            bridges=["rest", "episodic-memory", "semantic-memory", "circadian-rhythm"],
            tags=["memory", "biology", "learning", "fleet"])

        ns.define("rehearsal",
            "Active recall of a memory to strengthen it and reset its decay timer",
            level=Level.PATTERN,
            examples=["flashcard review resets the forgetting curve", "explaining a concept to someone else strengthens your memory of it", "agent reviews failed deliberation to learn from it"],
            bridges=["forgetting-curve", "consolidation", "learning", "spaced-repetition"],
            tags=["memory", "learning"])

        ns.define("chunking",
            "Grouping individual items into larger meaningful units to expand effective capacity",
            level=Level.PATTERN,
            examples=["phone number 555-1234 as two chunks not seven digits", "a 'trip to the store' is one chunk containing many sub-events", "skill = chunk of related procedural memories"],
            bridges=["working-memory", "abstraction", "hierarchy", "pattern"],
            tags=["memory", "cognition", "abstraction"])

    def _load_coordination(self):
        ns = self.add_namespace("coordination",
            "How multiple agents work together — or fail to")

        ns.define("stigmergy",
            "Indirect coordination through environment modification",
            level=Level.BEHAVIOR,
            examples=["ant trails", "wikipedia edits (each edit is a trace others build on)", "git commits (each commit is a trace the next developer reads)", "agents leaving 'marks' on a shared field that other agents follow"],
            properties={"direct": False, "medium": "environment", "scalability": "excellent"},
            bridges=["gossip", "consensus", "broadcast", "swarm", "tuplespace"],
            tags=["swarm", "decentralized", "scalable", "fleet"])

        ns.define("consensus",
            "Agreement among agents on a shared state or decision",
            level=Level.BEHAVIOR,
            examples=["raft protocol elects a leader", "jury reaches unanimous verdict", "fleet agrees on navigation plan with 0.92 confidence"],
            bridges=["deliberation", "voting", "agreement", "quorum"],
            tags=["coordination", "distributed", "fleet"])

        ns.define("deliberation",
            "Structured consideration of options leading to a decision",
            level=Level.BEHAVIOR,
            examples=["jury deliberation", "design review meeting", "agent evaluates 3 paths and selects the one with highest confidence"],
            properties={"protocol": "consider-resolve-forfeit", "threshold": 0.85},
            bridges=["consensus", "decision-making", "convergence", "filtration"],
            tags=["cognition", "coordination", "fleet"])

        ns.define("gossip",
            "Agents sharing information with random neighbors, spreading knowledge through the network",
            level=Level.PATTERN,
            examples=["epidemic information dissemination", "trust scores spreading through fleet", "discovery protocols finding new agents"],
            bridges=["stigmergy", "broadcast", "consensus", "trust"],
            tags=["coordination", "distributed", "scalable"])

        ns.define("swarm",
            "Collective behavior emerging from simple local rules without central control",
            level=Level.BEHAVIOR,
            examples=["bird flocking", "ant colony optimization", "fleet agents self-organizing around a task without central command"],
            bridges=["stigmergy", "emergence", "consensus", "decentralized"],
            tags=["swarm", "decentralized", "emergence", "fleet"])

        ns.define("emergence",
            "Complex global behavior arising from simple local interactions",
            level=Level.META,
            examples=["consciousness from neurons", "traffic jams from individual driving decisions", "fleet discovers an optimal division of labor nobody explicitly planned"],
            properties={"detection": "welford-baseline", "types": "8-pattern-types"},
            bridges=["swarm", "stigmergy", "self-organization", "complexity"],
            tags=["meta", "swarm", "complexity", "fleet"])

        ns.define("quorum",
            "Minimum number of agents required for a decision to be valid",
            level=Level.PATTERN,
            examples=["majority vote needs quorum of 51%", "byzantine fault tolerance needs 3f+1 agents", "fleet deliberation requires minimum 3 participants"],
            bridges=["consensus", "voting", "byzantine", "election"],
            tags=["coordination", "distributed", "fault-tolerance"])

        ns.define("leader-election",
            "Process of selecting a coordinator from a group of peers",
            level=Level.PATTERN,
            examples=["raft protocol", "bully algorithm", "fleet elects a task coordinator for the current mission"],
            bridges=["quorum", "consensus", "heartbeat", "fault-tolerance"],
            tags=["coordination", "distributed", "fault-tolerance", "fleet"])

    def _load_learning(self):
        ns = self.add_namespace("learning",
            "How agents improve through experience")

        ns.define("exploration",
            "Trying new actions to discover potentially better strategies",
            level=Level.BEHAVIOR,
            examples=["epsilon-greedy: 10% of the time, pick randomly", "curiosity-driven: seek surprising states", "trying a new restaurant instead of the usual one"],
            bridges=["exploitation", "curiosity", "entropy", "discovery"],
            antonyms=["exploitation"],
            tags=["learning", "reinforcement", "agent-behavior"])

        ns.define("exploitation",
            "Using currently known best actions to maximize reward",
            level=Level.BEHAVIOR,
            examples=["always taking the shortest known path", "using the proven sorting algorithm", "going to your favorite restaurant every time"],
            bridges=["exploration", "optimization", "convergence", "habit"],
            antonyms=["exploration"],
            tags=["learning", "reinforcement", "optimization"])

        ns.define("credit-assignment",
            "Determining which action caused an outcome when many actions contribute",
            level=Level.META,
            examples=["was it the new sensor or the better path that improved accuracy?", "which weight change in the neural network caused the improvement?", "which team member's contribution was most valuable?"],
            bridges=["learning", "causality", "attribution", "provenance"],
            tags=["learning", "meta", "causality"])

        ns.define("transfer-learning",
            "Applying knowledge from one domain to a different but related domain",
            level=Level.PATTERN,
            examples=["learning Python helps learn Rust", "spatial reasoning transfers between indoor and outdoor navigation", "an agent's pathfinding skill improves its route-planning skill"],
            bridges=["generalization", "abstraction", "analogy", "genepool"],
            tags=["learning", "generalization"])

        ns.define("curriculum",
            "Structured sequence of learning tasks from easy to hard",
            level=Level.PATTERN,
            examples=["math: arithmetic -> algebra -> calculus", "driving: parking lot -> residential -> highway", "agent: navigate empty room -> navigate with obstacles -> navigate with moving obstacles"],
            bridges=["skill", "learning-rate", "scaffolding", "progression"],
            tags=["learning", "education", "skill"])

        ns.define("spaced-repetition",
            "Reviewing material at increasing intervals to maximize retention",
            level=Level.PATTERN,
            examples=["flashcard apps like Anki", "reviewing code after 1 day, 3 days, 1 week", "agent reviews past lessons at expanding intervals"],
            bridges=["forgetting-curve", "rehearsal", "consolidation", "memory"],
            tags=["learning", "memory", "psychology"])

        ns.define("overfitting",
            "Learning the training examples too well, failing on new situations",
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
            level=Level.DOMAIN,
            examples=["newborn reflexes: suckling, grasping", "agent automatically avoids obstacles before deliberating about path", "fight-or-flight response fires before conscious thought"],
            properties={"inherited": True, "priority_10": "survive", "priority_1": "rest"},
            bridges=["reflex", "energy", "mitochondrion", "opcode"],
            tags=["biology", "fleet-foundation", "agent-behavior"])

        ns.define("apoptosis",
            "Programmed cell death — graceful self-termination when fitness drops below threshold",
            level=Level.DOMAIN,
            examples=["tail disappears in frog development", "damaged cells self-destruct to prevent cancer", "agent with failing sensors gracefully shuts down and reports to fleet"],
            properties={"fitness_threshold": 0.1, "patience": "10 ticks", "graceful": True},
            bridges=["shutdown", "graceful-degradation", "fitness", "resource-release"],
            tags=["biology", "safety", "fleet"])

        ns.define("homeostasis",
            "Maintenance of stable internal conditions despite external changes",
            level=Level.DOMAIN,
            examples=["thermoregulation", "blood pH maintained at 7.4", "agent adjusts deliberation depth based on available energy"],
            bridges=["feedback-loop", "adaptation", "setpoint", "circadian-rhythm"],
            tags=["biology", "control", "stability"])

        ns.define("circadian-rhythm",
            "Time-based modulation of behavior and capability following a ~24-hour cycle",
            level=Level.PATTERN,
            examples=["humans alert at 10am, drowsy at 3am", "agent's navigation accuracy peaks midday, communication peaks evening", "cosine modulation: strength = 0.55 + 0.45 * cos(2π * (hour - peak) / 24)"],
            properties={"function": "cosine", "period": "24 hours", "floor": 0.1},
            bridges=["energy", "instinct", "homeostasis", "scheduling"],
            tags=["biology", "temporal", "fleet"])

        ns.define("neurotransmitter",
            "Chemical signal that modulates neural activity — the fleet's confidence amplifier",
            level=Level.DOMAIN,
            examples=["dopamine spike when prediction confirmed = confidence boost", "serotonin builds with social bonding = trust accumulation", "norepinephrine fires on threat = immediate alert"],
            properties={"types": 8, "down_regulation": True, "hebbian": True},
            bridges=["confidence", "trust", "attention", "emotion"],
            tags=["biology", "cognition", "fleet"])

        ns.define("membrane",
            "Self/other boundary that filters what enters and leaves the agent",
            level=Level.DOMAIN,
            examples=["cell membrane with selective permeability", "firewall blocking dangerous packets", "agent's membrane blocks self-destruct commands before they reach deliberation"],
            bridges=["security", "sandbox", "filter", "boundary"],
            tags=["biology", "security", "fleet"])

        ns.define("enzyme",
            "Catalyst that converts environmental signals into genetic activation",
            level=Level.PATTERN,
            examples=["lactase enzyme converts lactose into absorbable sugars", "sensor reads 'low ATP' -> enzyme activates 'rest' gene", "pattern matcher in deliberation that triggers emergency protocol"],
            bridges=["instinct", "perception", "gene-activation", "signal-processing"],
            tags=["biology", "pipeline", "fleet"])

        ns.define("hebbian-learning",
            "Synapses strengthen when pre- and post-synaptic neurons fire together",
            level=Level.PATTERN,
            examples=["pavlovian conditioning: bell + food = bell causes salivation", "sensor detects obstacle right before collision -> sensor-obstacle association strengthens", "learning that asking navigator before pathfinding improves outcomes"],
            bridges=["learning", "credit-assignment", "synapse", "correlation"],
            tags=["biology", "learning", "neuroscience"])

    def _load_architecture(self):
        ns = self.add_namespace("architecture",
            "Software architecture patterns and structures")

        ns.define("actor-model",
            "Concurrency model where each agent is an isolated entity communicating via messages",
            level=Level.PATTERN,
            examples=["Erlang processes", "Akka actors", "each fleet agent is an actor with a mailbox"],
            properties={"isolation": True, "async": True, "supervision": True},
            bridges=["agent", "mailbox", "concurrency", "fault-tolerance"],
            tags=["architecture", "concurrency", "fleet"])

        ns.define("circuit-breaker",
            "Prevent cascading failures by stopping calls to a failing service",
            level=Level.PATTERN,
            examples=["Netflix Hystrix", "stop calling an API that's returning 500 errors", "agent stops querying a sensor that's been noisy for 30 seconds"],
            bridges=["fault-tolerance", "bulkhead", "backpressure", "graceful-degradation"],
            tags=["resilience", "pattern", "fleet"])

        ns.define("bulkhead",
            "Isolate components so one failure doesn't take down the whole system",
            level=Level.PATTERN,
            examples=["ship compartments", "thread pools per service", "each agent has its own energy budget — one agent's exhaustion doesn't affect others"],
            bridges=["circuit-breaker", "isolation", "fault-tolerance", "resource-pool"],
            tags=["resilience", "pattern", "fleet"])

        ns.define("event-sourcing",
            "Store every state change as an immutable event, reconstruct state by replaying",
            level=Level.PATTERN,
            examples=["git history = event-sourced code state", "bank ledger = event-sourced balance", "agent's decision history = event-sourced mental state"],
            bridges=["provenance", "audit-trail", "persistence", "immutable"],
            tags=["architecture", "persistence", "audit"])

        ns.define("state-machine",
            "Model behavior as a finite set of states with defined transitions",
            level=Level.PATTERN,
            examples=["traffic light: red -> green -> yellow -> red", "agent: idle -> navigating -> arrived -> idle", "TCP: closed -> syn-sent -> established -> fin-wait -> closed"],
            bridges=["workflow", "lifecycle", "guard", "transition"],
            tags=["architecture", "modeling", "fleet"])

        ns.define("backpressure",
            "Signal to slow down when the consumer can't keep up with the producer",
            level=Level.PATTERN,
            examples=["TCP flow control", "tell a fast sensor to sample less frequently", "fleet coordinator slows task assignment when agents are overloaded"],
            bridges=["flow-control", "throttle", "rate-limit", "congestion"],
            tags=["architecture", "resilience", "fleet"])

        ns.define("sidecar",
            "Separate helper process attached to a primary component for cross-cutting concerns",
            level=Level.CONCRETE,
            examples=["Envoy proxy alongside a microservice", "logging agent alongside a navigation agent", "health monitor watching a computation agent"],
            bridges=["monitoring", "logging", "proxy", "separation-of-concerns"],
            tags=["architecture", "pattern", "operations"])

    def _load_spatial(self):
        ns = self.add_namespace("spatial",
            "How agents understand and navigate physical and abstract space")

        ns.define("attention-tile",
            "A rectangular region of an attention matrix that is computed (or skipped) as a unit",
            level=Level.CONCRETE,
            examples=["8x8 tile in a 64x64 attention matrix", "skip the bottom-left tile because past tokens rarely attend to future tokens", "GPU thread block = one attention tile"],
            bridges=["sparsity", "pruning", "attention", "gpu-optimization"],
            tags=["spatial", "optimization", "gpu"])

        ns.define("spatial-hash",
            "Hash-based spatial lookup that avoids hierarchical structures",
            level=Level.PATTERN,
            examples=["grid-based collision detection in games", "finding nearby agents without checking all agents", "uniform grid spatial hashing"],
            bridges=["grid", "hash", "neighbor-query", "collision-detection"],
            tags=["spatial", "data-structure", "optimization"])

        ns.define("manhattan-distance",
            "Distance measured along grid axes (|dx| + |dy|) — the city block metric",
            level=Level.CONCRETE,
            examples=["taxicab distance in a city grid", "moving a chess rook from a1 to h8 = 14 squares", "agent navigation on a grid map"],
            bridges=["euclidean-distance", "pathfinding", "heuristic", "grid"],
            tags=["spatial", "geometry", "pathfinding"])

        ns.define("a-star",
            "Optimal pathfinding algorithm using actual cost + estimated remaining cost",
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
            level=Level.PATTERN,
            examples=["deadline in 1 hour: urgency 0.9, agent drops everything else", "deadline in 1 week: urgency 0.2, agent works on it when convenient", "past deadline: urgency 1.0, agent enters emergency mode"],
            bridges=["priority", "scheduling", "preemption", "time-pressure"],
            tags=["temporal", "scheduling", "fleet"])

        ns.define("causal-chain",
            "A sequence of events where each causes the next",
            level=Level.PATTERN,
            examples=["domino effect", "sensor reading -> deliberation -> decision -> action -> result", "git commit chain: each commit references its parent"],
            bridges=["provenance", "causality", "audit-trail", "temporal"],
            tags=["temporal", "causality", "audit", "fleet"])

        ns.define("heartbeat",
            "Periodic signal indicating an agent is alive and healthy",
            level=Level.PATTERN,
            examples=["raft leader heartbeats", "health check pings every 30s", "watchdog timer in embedded systems"],
            bridges=["health", "fault-detection", "timeout", "leader-election"],
            tags=["temporal", "fault-tolerance", "coordination", "fleet"])

    def _load_communication(self):
        ns = self.add_namespace("communication",
            "How agents exchange information and meaning")

        ns.define("grounding",
            "Establishing shared understanding of word meanings between agents",
            level=Level.BEHAVIOR,
            examples=["two humans agreeing that 'soon' means 'within 5 minutes'", "agents negotiating that 'high priority' means 'respond within 1 second'", "establishing a shared coordinate system"],
            bridges=["vocabulary", "shared-understanding", "negotiation", "semantic-alignment"],
            tags=["communication", "language", "coordination", "fleet"])

        ns.define("speech-act",
            "An utterance that performs an action — saying is doing",
            level=Level.DOMAIN,
            examples=["'I promise...' = commitment", "'I order you to...' = command", "'I apologize for...' = repair", "'Warning: obstacle ahead' = alert"],
            bridges=["intent", "a2a", "communication", "action"],
            tags=["communication", "language", "philosophy"])

        ns.define("information-bottleneck",
            "Compressing information to its most essential parts before transmission",
            level=Level.PATTERN,
            examples=["summarizing a 1-hour meeting in 3 bullet points", "agent sends 'path blocked at intersection' instead of full lidar scan", "compressing 1000 sensor readings into 'temperature nominal'"],
            bridges=["compression", "abstraction", "communication-cost", "attention"],
            tags=["communication", "information-theory", "optimization", "fleet"])

        ns.define("context-window",
            "The amount of recent information an agent can consider simultaneously",
            level=Level.CONCRETE,
            examples=["GPT's 128K token context window", "agent can hold 7 items in working memory", "conversation history limited to last 50 messages"],
            bridges=["working-memory", "attention", "chunking", "capacity"],
            tags=["communication", "cognition", "capacity"])

    def _load_security(self):
        ns = self.add_namespace("security",
            "Safety, boundaries, and trust enforcement")

        ns.define("least-privilege",
            "Give an agent only the permissions it needs, nothing more",
            level=Level.PATTERN,
            examples=["read-only access to config files", "agent can observe but not modify", "wildcard permissions for admin, specific permissions for worker"],
            bridges=["rbac", "sandbox", "membrane", "boundary"],
            tags=["security", "principle", "fleet"])

        ns.define("sandbox",
            "Restricted execution environment that limits what an agent can do",
            level=Level.CONCRETE,
            examples=["browser sandbox limiting JavaScript access", "container limiting CPU and memory", "agent sandbox: max 100ms compute per operation, max 50 operations per second"],
            bridges=["least-privilege", "rbac", "resource-limit", "isolation"],
            tags=["security", "isolation", "fleet"])

        ns.define("graceful-degradation",
            "Continue operating at reduced capability instead of failing completely",
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
            level=Level.BEHAVIOR,
            examples=["choosing a restaurant that's 'good enough' vs visiting all 50 to find the best", "agent picks first path with confidence > 0.7 instead of evaluating all 10 paths", "buying the first car that meets your requirements"],
            bridges=["deliberation", "optimization", "heuristics", "energy-conservation"],
            antonyms=["maximizing"],
            tags=["decision", "heuristics", "behavior"])

        ns.define("multi-armed-bandit",
            "Balancing exploration of unknown options against exploitation of known best",
            level=Level.PATTERN,
            examples=["A/B testing: which variant gets more clicks?", "choosing which restaurant to try next", "agent deciding which navigation algorithm to use for this terrain"],
            bridges=["exploration", "exploitation", "ucb", "thompson-sampling"],
            tags=["decision", "reinforcement", "statistics"])

        ns.define("minimax",
            "Choose the action that minimizes the maximum possible loss",
            level=Level.PATTERN,
            examples=["chess engine assuming best opponent play", "choosing the route with the best worst-case travel time", "agent planning for sensor failure during critical task"],
            bridges=["adversarial", "risk-aversion", "worst-case", "game-theory"],
            tags=["decision", "game-theory", "algorithm"])

        ns.define("paradox-of-choice",
            "More options lead to worse decisions or decision paralysis",
            level=Level.BEHAVIOR,
            examples=["menu with 500 items vs menu with 10 items", "agent freezing when presented with 1000 possible actions", "dating app fatigue from too many profiles"],
            bridges=["filtration", "deliberation", "working-memory", "overwhelm"],
            tags=["decision", "psychology", "cognition"])



    def _load_control_theory(self):
        ns = self.add_namespace("control-theory",
            "Feedback, regulation, and maintaining target states")

        ns.define("feedback-loop",
            "Output of a system is measured and used to adjust input to maintain a target",
            level=Level.PATTERN,
            examples=["thermostat maintains 70F", "cruise control maintains 65mph", "agent adjusts exploration rate based on recent success"],
            bridges=["homeostasis", "setpoint", "pid-controller", "adaptation"],
            tags=["control", "feedback", "stability", "fleet"])

        ns.define("setpoint",
            "The target value a control system tries to maintain",
            level=Level.CONCRETE,
            examples=["thermostat setpoint 70F", "consensus threshold 0.85", "target speed 60mph", "desired trust level 0.7"],
            bridges=["feedback-loop", "homeostasis", "threshold", "target"],
            tags=["control", "target", "fleet"])

        ns.define("hysteresis",
            "The output depends not just on current input but on history — path dependence",
            level=Level.PATTERN,
            examples=["thermostat: heat to 72, cool to 68, not flip at 70", "Schmitt trigger in electronics", "agent proposal accepted at 0.85, stays accepted until confidence drops to 0.75"],
            bridges=["feedback-loop", "oscillation", "stability", "threshold"],
            tags=["control", "stability", "pattern"])

        ns.define("overshoot",
            "System exceeds its target before settling back — the pendulum swings past center",
            level=Level.BEHAVIOR,
            examples=["pressing brake too hard", "stock price correction going below fair value", "agent switches from 50% exploration to 0% exploration overnight"],
            bridges=["feedback-loop", "oscillation", "adaptation", "correction"],
            tags=["control", "behavior", "failure-mode"])

        ns.define("dead-zone",
            "Range of inputs that produce no output — intentional insensitivity",
            level=Level.CONCRETE,
            examples=["joystick dead zone prevents drift", "sensor noise below threshold ignored", "confidence change from 0.80 to 0.81 doesn't trigger deliberation review"],
            bridges=["hysteresis", "threshold", "noise-filtering", "robustness"],
            tags=["control", "noise", "robustness"])

    def _load_evolution(self):
        ns = self.add_namespace("evolution",
            "Evolutionary dynamics — selection, drift, speciation, co-evolution")

        ns.define("natural-selection",
            "Differential survival and reproduction based on fitness",
            level=Level.DOMAIN,
            examples=["giraffe necks lengthen because taller giraffes reach more food", "gene with 0.8 fitness spreads; gene with 0.1 fitness quarantined", "navigation strategy that finds paths faster gets selected over slower one"],
            bridges=["fitness-landscape", "genetic-drift", "mutation", "adaptation"],
            tags=["evolution", "biology", "fleet"])

        ns.define("fitness-landscape",
            "Multi-dimensional space where each position represents a strategy and height represents fitness",
            level=Level.DOMAIN,
            examples=["evolution climbs fitness peaks", "agent stuck in local optimum of 'always exploit'", "adding noise (mutation) lets agent jump across valleys to taller peaks"],
            bridges=["local-minimum", "exploration", "mutation", "natural-selection"],
            tags=["evolution", "optimization", "visualization"])

        ns.define("punctuated-equilibrium",
            "Long periods of stability interrupted by sudden rapid change",
            level=Level.BEHAVIOR,
            examples=["Cambrian explosion", "agent runs strategy X for weeks, then sensor fails and it must completely restructure behavior", "technology disruption forces sudden industry change"],
            bridges=["evolution", "stability", "disruption", "adaptation"],
            tags=["evolution", "pattern", "disruption"])

        ns.define("genetic-drift",
            "Random changes in gene frequency unrelated to fitness — noise in evolution",
            level=Level.BEHAVIOR,
            examples=["neutral mutation spreading in small population", "fleet of 3 agents: one agent's random behavioral quirk spreads to others", "founder effect: new colony has different gene frequencies than parent"],
            bridges=["natural-selection", "noise", "population-size", "random-walk"],
            tags=["evolution", "noise", "population"])

        ns.define("co-evolution",
            "Two species evolve in response to each other — arms races and mutualisms",
            level=Level.META,
            examples=["predator-prey arms race", "virus-antivirus co-evolution", "adversarial-red-team vs compliance-engine arms race"],
            bridges=["natural-selection", "competition", "arms-race", "adaptation"],
            tags=["evolution", "meta", "competition", "fleet"])

        ns.define("speciation",
            "Divergence into separate species when populations face different selective pressures",
            level=Level.BEHAVIOR,
            examples=["Darwin's finches on different Galapagos islands", "warehouse agent vs outdoor agent developing incompatible navigation strategies", "generalist agent splitting into specialist sub-agents"],
            bridges=["niche", "divergence", "specialization", "adaptation"],
            tags=["evolution", "diversity", "specialization"])

    def _load_networks(self):
        ns = self.add_namespace("networks",
            "Graph structures, connectivity patterns, and network effects")

        ns.define("small-world",
            "Network where most nodes are locally connected but any two nodes are reachable in few hops",
            level=Level.DOMAIN,
            examples=["social networks: six degrees of separation", "neural networks: mostly local connections, few long-range", "fleet mesh: agents gossip with neighbors, information reaches whole fleet in ~5 hops"],
            bridges=["gossip", "scale-free", "clustering", "fleet-mesh"],
            tags=["networks", "social", "fleet"])

        ns.define("scale-free",
            "Network where degree distribution follows a power law — few hubs, many leaves",
            level=Level.DOMAIN,
            examples=["internet: few sites with billions of links", "airline network: few hub airports, many spoke airports", "fleet: coordinator agent talks to 50 agents, worker agents talk to 3"],
            bridges=["hub", "small-world", "power-law", "robustness"],
            tags=["networks", "structure", "statistics"])

        ns.define("hub",
            "A node with disproportionately many connections in a network",
            level=Level.CONCRETE,
            examples=["airport hub: O'Hare connects to 200+ destinations", "Google: linked by billions of pages", "fleet captain: communicates with every agent"],
            bridges=["scale-free", "single-point-of-failure", "leader-election", "redundancy"],
            tags=["networks", "critical", "vulnerability"])

        ns.define("percolation",
            "Phase transition in connectivity: at a critical density, a giant connected component forms",
            level=Level.META,
            examples=["water through coffee grounds", "forest fire spreading when tree density exceeds threshold", "fleet information spreading when enough agents are connected", "disease outbreak at critical infection rate"],
            bridges=["phase-transition", "critical-mass", "cascade-failure", "connectivity"],
            tags=["networks", "phase-transition", "criticality"])

        ns.define("cascade-failure",
            "Failure of one node triggers failures in dependent nodes, spreading through the network",
            level=Level.BEHAVIOR,
            examples=["2003 Northeast blackout", "bank run: one bank fails, depositors panic, other banks fail", "fleet: overloaded agent crashes, task redistribution overloads neighbors"],
            bridges=["circuit-breaker", "bulkhead", "single-point-of-failure", "robustness"],
            antonyms=["isolation", "containment"],
            tags=["networks", "failure-mode", "critical", "fleet"])

        ns.define("clustering-coefficient",
            "How likely two neighbors of a node are also neighbors of each other",
            level=Level.CONCRETE,
            examples=["friend group: your friends know each other", "work team: tight cluster within larger organization", "fleet: navigation agents cluster together, communication agents cluster together"],
            bridges=["small-world", "community", "group-formation", "topology"],
            tags=["networks", "metric", "social"])

    def _load_game_theory(self):
        ns = self.add_namespace("game-theory",
            "Strategic interaction between rational (and irrational) agents")

        ns.define("nash-equilibrium",
            "A state where no agent can improve by changing strategy alone, assuming others stay",
            level=Level.DOMAIN,
            examples=["prisoner's dilemma: both stay silent would be better, but both confess", "traffic: everyone driving is equilibrium, public transit would be better for all", "agents all exploiting is equilibrium, some exploring would be better for fleet"],
            bridges=["prisoners-dilemma", "mechanism-design", "equilibrium", "cooperation"],
            tags=["game-theory", "equilibrium", "strategy"])

        ns.define("prisoners-dilemma",
            "Two agents each choose to cooperate or defect; individual incentive conflicts with group welfare",
            level=Level.DOMAIN,
            examples=["two suspects interrogated separately", "arms race: both build weapons (defect) vs both disarm (cooperate)", "agents sharing vs hoarding sensor data"],
            bridges=["nash-equilibrium", "tit-for-tat", "cooperation", "tragedy-of-commons"],
            tags=["game-theory", "social-dilemma", "cooperation"])

        ns.define("mechanism-design",
            "Designing rules of a game so that agents' self-interest produces desired outcomes",
            level=Level.META,
            examples=["auction design: Vickrey auction makes truthful bidding optimal", "fleet energy costs: self-interest (conserve ATP) aligns with system (prevent spam)", "carbon credits: self-interest (minimize cost) aligns with system (reduce emissions)"],
            bridges=["nash-equilibrium", "incentive-alignment", "game-rules", "economics"],
            tags=["game-theory", "design", "meta", "economics"])

        ns.define("tragedy-of-commons",
            "Shared resource depleted by individual agents acting in self-interest",
            level=Level.DOMAIN,
            examples=["overfishing", "climate change: each country benefits from cheap energy, costs shared globally", "fleet: agents all requesting maximum compute budget", "open office: everyone talks loudly, nobody can focus"],
            bridges=["nash-equilibrium", "resource-allocation", "energy-budget", "mechanism-design"],
            tags=["game-theory", "economics", "resource", "failure-mode"])

        ns.define("zero-sum",
            "One agent's gain is exactly another agent's loss — the pie doesn't grow",
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
            level=Level.PATTERN,
            examples=["neural network training", "finding minimum of a function by following negative gradient", "agent incrementally adjusts navigation strategy based on success/failure feedback"],
            bridges=["local-minimum", "learning-rate", "convergence", "hill-climbing"],
            tags=["optimization", "algorithm", "learning"])

        ns.define("local-minimum",
            "A valley that looks like the lowest point from inside, but a deeper valley exists elsewhere",
            level=Level.DOMAIN,
            examples=["ball rolling into a small divot on a hilly surface", "always going to same restaurant (local optimum) when a better one exists across town", "agent stuck using suboptimal navigation algorithm because small tweaks don't help"],
            bridges=["fitness-landscape", "exploration", "simulated-annealing", "gradient-descent"],
            tags=["optimization", "failure-mode", "search"])

        ns.define("simulated-annealing",
            "Occasionally accept worse solutions to escape local minima, accepting worse moves less often over time",
            level=Level.PATTERN,
            examples=["metal annealing: heat and slowly cool to reduce crystal defects", "traveling salesman: occasionally take a worse route to escape local optimum", "agent: early in task, try random strategies; later, stick with what works"],
            bridges=["local-minimum", "exploration", "temperature", "hill-climbing"],
            tags=["optimization", "algorithm", "search"])

        ns.define("convergence-criteria",
            "Conditions that determine when an optimization process should stop",
            level=Level.PATTERN,
            examples=["neural network: stop when loss changes less than 0.0001 for 10 epochs", "deliberation: stop when consensus exceeds 0.85", "search: stop after 1000 iterations or when best score hasn't improved in 100 iterations"],
            bridges=["convergence", "threshold", "optimization", "deliberation"],
            tags=["optimization", "stopping", "fleet"])

        ns.define("multi-objective",
            "Optimizing for multiple conflicting goals simultaneously",
            level=Level.DOMAIN,
            examples=["car design: fast vs fuel-efficient vs safe vs cheap", "agent: minimize energy (fast response) vs maximize accuracy (deep deliberation)", "software: minimize latency vs maximize throughput"],
            bridges=["pareto-frontier", "tradeoff", "satisficing", "priority"],
            tags=["optimization", "multi-criteria", "tradeoff", "fleet"])

    def _load_probability(self):
        ns = self.add_namespace("probability",
            "Reasoning under uncertainty — priors, likelihood, evidence")

        ns.define("prior",
            "Belief about a hypothesis before seeing new evidence",
            level=Level.DOMAIN,
            examples=["medical test: prior probability of disease affects interpretation of positive test", "agent prior: 'this path is usually safe' before checking sensors", "Bayesian spam filter: prior probability that email is spam"],
            bridges=["posterior", "bayesian-update", "base-rate-fallacy", "likelihood"],
            tags=["probability", "bayesian", "prior-knowledge"])

        ns.define("base-rate-fallacy",
            "Ignoring the prior probability when interpreting new evidence",
            level=Level.BEHAVIOR,
            examples=["1 in 1000 disease, 99% test: positive test = only 9% chance of disease", "agent: sensor says danger, but danger is rare (base rate 0.1%) so probably false alarm", "profiling: rare trait in population, even accurate screening produces mostly false positives"],
            bridges=["prior", "bayesian-update", "false-positive", "calibration"],
            tags=["probability", "fallacy", "reasoning"])

        ns.define("conjunction-fallacy",
            "Believing that a specific conjunction is more probable than a general statement",
            level=Level.BEHAVIOR,
            examples=["Linda the feminist bank teller", "agent: 'the path is blocked because the door is locked AND the key is lost' vs 'the path is blocked'", "overweighting specific failure modes over general failure probability"],
            bridges=["probability", "fallacy", "reasoning", "specificity"],
            tags=["probability", "fallacy", "cognitive-bias"])

        ns.define("regression-to-mean",
            "Extreme observations tend to be followed by more average ones",
            level=Level.BEHAVIOR,
            examples=["sports: rookie of the year slump", "agent: amazing performance week 1, average week 2 — not because something broke", "student: aced test after studying hard, next test is lower — not because they forgot everything"],
            bridges=["mean", "variance", "luck", "calibration"],
            tags=["probability", "statistics", "fallacy"])

    def _load_economics(self):
        ns = self.add_namespace("economics",
            "Markets, incentives, costs, and resource allocation")

        ns.define("opportunity-cost",
            "The value of the best alternative you gave up by choosing this option",
            level=Level.DOMAIN,
            examples=["studying for exam A means not studying for exam B", "agent spending ATP on deliberation can't spend it on perception", "choosing to explore means not exploiting the known best path"],
            bridges=["tradeoff", "resource-allocation", "budget", "cost"],
            tags=["economics", "cost", "decision"])

        ns.define("marginal-cost",
            "The cost of producing one more unit — usually decreasing",
            level=Level.DOMAIN,
            examples=["software: first copy costs $1M, next copy costs $0.01", "fleet: first agent needs full setup, additional agents need minimal extra infrastructure", "manufacturing: first car off assembly line is most expensive"],
            bridges=["economies-of-scale", "diminishing-returns", "cost", "scaling"],
            tags=["economics", "cost", "scaling"])

        ns.define("externalities",
            "Costs or benefits that affect parties not involved in the transaction",
            level=Level.DOMAIN,
            examples=["pollution from factory", "loud music in shared office", "agent spamming fleet messages: cheap for sender, expensive for receivers", "vaccination: positive externality (protects others)"],
            bridges=["tragedy-of-commons", "mechanism-design", "incentive-alignment", "cost"],
            tags=["economics", "market-failure", "incentive"])

        ns.define("market-equilibrium",
            "Price point where supply equals demand — neither shortage nor surplus",
            level=Level.DOMAIN,
            examples=["supply and demand curves crossing", "fleet energy: rest generates ATP, actions consume it, equilibrium when balanced", "task allocation: supply of available agents meets demand from tasks"],
            bridges=["supply-demand", "equilibrium", "homeostasis", "energy-budget"],
            tags=["economics", "equilibrium", "market"])

    def _load_ecology(self):
        ns = self.add_namespace("ecology",
            "How agents interact with their environment and each other as an ecosystem")

        ns.define("niche",
            "The specific role and resource space an organism occupies in its ecosystem",
            level=Level.DOMAIN,
            examples=["different bird species feeding at different heights in same tree", "fleet: navigation agent niche vs communication agent niche", "market: different companies targeting different customer segments"],
            bridges=["competitive-exclusion", "speciation", "specialization", "role"],
            tags=["ecology", "niche", "role", "fleet"])

        ns.define("keystone-species",
            "A species whose removal dramatically changes the entire ecosystem",
            level=Level.DOMAIN,
            examples=["wolves in Yellowstone", "sea otters maintaining kelp forests", "fleet captain: small computational footprint but critical for coordination", "team lead: doesn't write code but enables the team"],
            bridges=["hub", "cascade-failure", "critical-dependency", "leader"],
            tags=["ecology", "critical", "system-impact"])

        ns.define("symbiosis",
            "Long-term interaction between different species that benefits at least one",
            level=Level.DOMAIN,
            examples=["bees pollinate flowers, flowers feed bees", "barnacles on whale: barnacles benefit, whale unaffected", "fleet: navigator and sensor agents in mutualism — both need each other"],
            bridges=["cooperation", "mutualism", "parasitism", "niche"],
            tags=["ecology", "interaction", "cooperation"])

        ns.define("competitive-exclusion",
            "Two species competing for the same niche cannot coexist indefinitely",
            level=Level.BEHAVIOR,
            examples=["two similar bird species on an island: one outcompetes the other", "two identical fleet agents: one should specialize or be removed", "market: companies with identical products compete until one dominates"],
            bridges=["niche", "speciation", "specialization", "diversity"],
            tags=["ecology", "competition", "specialization"])

        ns.define("succession",
            "Predictable sequence of community changes following a disturbance",
            level=Level.BEHAVIOR,
            examples=["volcanic island colonization", "forest regrowth after fire", "fleet recovery after major failure: instinct -> perception -> coordination -> optimization"],
            bridges=["punctuated-equilibrium", "disruption", "recovery", "stages"],
            tags=["ecology", "recovery", "sequence"])

    def _load_emotion(self):
        ns = self.add_namespace("emotion",
            "Emotional states as computational modulators of agent behavior")

        ns.define("valence-arousal",
            "Two-dimensional model of emotion: positive/negative (valence) x calm/excited (arousal)",
            level=Level.DOMAIN,
            examples=["joy: positive valence, high arousal", "calm: positive valence, low arousal", "anger: negative valence, high arousal", "agent: high arousal = faster decisions, lower accuracy"],
            bridges=["emotion", "modulation", "attention", "decision"],
            tags=["emotion", "psychology", "modulation", "fleet"])

        ns.define("emotional-contagion",
            "Emotional state spreading from one agent to others through observation",
            level=Level.BEHAVIOR,
            examples=["laughter spreading through a room", "panic in a crowd", "fleet: one agent detects threat, nearby agents become alert"],
            bridges=["cascade-failure", "emotion", "gossip", "swarm"],
            tags=["emotion", "social", "contagion", "fleet"])

        ns.define("anticipation",
            "Predictive emotional state generated by expecting a future event",
            level=Level.DOMAIN,
            examples=["looking forward to vacation", "dread before a difficult meeting", "agent: increasing urgency as deadline approaches = anticipatory emotional modulation"],
            bridges=["deadline-urgency", "prediction", "temporal", "motivation"],
            tags=["emotion", "prediction", "temporal", "motivation"])

    def _load_creativity(self):
        ns = self.add_namespace("creativity",
            "Generating novel, useful combinations from existing elements")

        ns.define("analogy",
            "Mapping structure from a known domain to a novel domain — 'A is to B as C is to D'",
            level=Level.DOMAIN,
            examples=["electricity:water :: voltage:pressure :: current:flow :: resistance:narrowing", "atom:solar system :: nucleus:sun :: electrons:planets", "stigmergy:git commits :: pheromone trails:commit history"],
            bridges=["transfer-learning", "metaphor", "abstraction", "cross-domain"],
            tags=["creativity", "reasoning", "analogy", "abstraction"])

        ns.define("divergent-thinking",
            "Generating many possible solutions without judging them — brainstorming mode",
            level=Level.BEHAVIOR,
            examples=["brainstorming: generate 100 ideas, don't judge yet", "agent: consider all possible navigation strategies before evaluating any", "creative writing: write freely, edit later"],
            bridges=["exploration", "convergent-thinking", "brainstorming", "generation"],
            antonyms=["convergent-thinking"],
            tags=["creativity", "generation", "exploration"])

        ns.define("convergent-thinking",
            "Evaluating and selecting the best solution from generated options — decision mode",
            level=Level.BEHAVIOR,
            examples=["narrowing 100 brainstorm ideas to 3 actionable ones", "agent: evaluate all navigation strategies by confidence, select best", "editing a rough draft into a polished piece"],
            bridges=["deliberation", "divergent-thinking", "evaluation", "selection"],
            antonyms=["divergent-thinking"],
            tags=["creativity", "evaluation", "selection"])

        ns.define("combinatorial-explosion",
            "Number of possible combinations grows exponentially with the number of elements",
            level=Level.META,
            examples=["chess: too many positions to enumerate, must use heuristics", "traveling salesman: N! routes, NP-hard", "agent deliberation: 100 possible actions × 10 contexts × 5 goals = 5000 combinations to evaluate"],
            bridges=["pruning", "satisficing", "filtration", "heuristic", "paradox-of-choice"],
            tags=["creativity", "complexity", "scaling", "challenge"])

        ns.define("constraint-relaxation",
            "Solving a hard problem by temporarily removing a constraint, solving, then re-adding it",
            level=Level.PATTERN,
            examples=["knapsack: ignore weight limit, pack all valuable items, then remove items until weight fits", "agent: plan optimal path ignoring energy, then trim path to fit budget", "writing: write without worrying about word count, then edit to fit"],
            bridges=["satisficing", "optimization", "heuristic", "abstraction"],
            tags=["creativity", "technique", "problem-solving"])

    def _load_metacognition(self):
        ns = self.add_namespace("metacognition",
            "Thinking about thinking — self-awareness, monitoring, and control of cognition")

        ns.define("introspection",
            "Examining one's own mental states, processes, and reasons for action",
            level=Level.BEHAVIOR,
            examples=["asking 'why did I make that decision?'", "agent reviewing its own deliberation log to understand decision patterns", "journaling as self-reflection"],
            bridges=["self-model", "metacognitive-monitoring", "calibration", "theory-of-mind"],
            tags=["metacognition", "self-awareness", "reflection"])

        ns.define("theory-of-mind",
            "Attributing mental states to others — predicting what others think, want, and will do",
            level=Level.DOMAIN,
            examples=["predicting what another driver will do at an intersection", "agent modeling another agent's current goal to avoid interference", "negotiating: understanding the other party's priorities"],
            bridges=["self-model", "social", "prediction", "coordination"],
            tags=["metacognition", "social", "prediction", "fleet"])

        ns.define("metacognitive-monitoring",
            "Watching your own cognitive process in real-time to detect confusion or error",
            level=Level.BEHAVIOR,
            examples=["'I don't understand' — detecting own confusion", "'I'm going in circles' — detecting unproductive deliberation", "agent: confidence dropping consistently across proposals = metacognitive alarm"],
            bridges=["introspection", "calibration", "confusion", "threshold"],
            tags=["metacognition", "monitoring", "self-awareness"])

    def _load_failure_modes(self):
        ns = self.add_namespace("failure-modes",
            "How systems fail — and how to prevent, detect, and recover from failure")

        ns.define("single-point-of-failure",
            "One component whose failure causes the entire system to fail",
            level=Level.DOMAIN,
            examples=["one hard drive with no backup", "one DNS server for entire network", "fleet: captain agent crash with no election mechanism = SPOF"],
            bridges=["redundancy", "cascade-failure", "circuit-breaker", "hub"],
            tags=["failure", "critical", "architecture"])

        ns.define("robustness",
            "Ability to maintain function despite perturbations without changing structure",
            level=Level.DOMAIN,
            examples=["bridge handles varying loads", "agent handles sensor noise without changing strategy", "software handles invalid input without crashing"],
            bridges=["resilience", "graceful-degradation", "anti-fragility", "stability"],
            tags=["failure", "property", "system"])

        ns.define("anti-fragility",
            "Getting stronger from stress — not just surviving perturbations but improving because of them",
            level=Level.META,
            examples=["muscles grow from exercise", "immune system from exposure", "fleet: agent failure -> gene quarantined -> fleet stronger", "bone density increases from stress"],
            bridges=["robustness", "resilience", "learning-from-failure", "adaptation"],
            antonyms=["fragility"],
            tags=["failure", "meta", "aspiration", "fleet"])

        ns.define("common-mode-failure",
            "Multiple components fail simultaneously because they share the same vulnerability",
            level=Level.DOMAIN,
            examples=["redundant servers in same datacenter: both fail in fire", "same sensor type on multiple agents: all fail in same interference", "identical software on different hardware: same bug crashes all"],
            bridges=["redundancy", "diversity", "single-point-of-failure", "robustness"],
            tags=["failure", "systematic", "redundancy"])

        ns.define("brittleness",
            "System works well under expected conditions but catastrophically fails under unexpected ones",
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
            level=Level.META,
            examples=["room gets messy without cleaning", "agent trust decays without positive interactions", "knowledge goes stale without updates", "code degrades without maintenance (software entropy)"],
            bridges=["entropy", "energy", "decay", "maintenance"],
            tags=["physics", "thermodynamics", "meta", "fleet"])

        ns.define("free-energy-principle",
            "Biological systems minimize surprise (prediction error) by updating model or changing environment",
            level=Level.META,
            examples=["you feel cold -> put on jacket (change world) or learn that it's cold here (update model)", "agent's prediction doesn't match sensor -> update world model OR move to expected state", "surprise minimization = free energy minimization"],
            bridges=["prediction", "action-perception", "homeostasis", "model"],
            tags=["physics", "neuroscience", "meta", "unified-theory"])

        ns.define("dissipative-structure",
            "Ordered pattern that emerges from energy flow through a system, maintaining itself far from equilibrium",
            level=Level.META,
            examples=["convection cells in boiling water", "hurricane maintained by ocean heat", "life maintained by metabolism", "fleet coordination maintained by constant message flow and energy expenditure"],
            bridges=["emergence", "self-organization", "energy-flow", "far-from-equilibrium"],
            tags=["physics", "complexity", "meta", "emergence"])

        ns.define("negentropy",
            "Local decrease in entropy (increase in order) at the expense of increased entropy elsewhere",
            level=Level.DOMAIN,
            examples=["plant converts sunlight to ordered structure, produces heat", "agent organizes fleet behavior, consumes ATP, produces noise", "refrigerator creates cold (order) by producing heat (disorder)"],
            bridges=["entropy", "energy", "order", "cost"],
            tags=["physics", "thermodynamics", "life", "cost"])

    def _load_complexity(self):
        ns = self.add_namespace("complexity",
            "Emergence, self-organization, and behavior at the edge of chaos")

        ns.define("edge-of-chaos",
            "The boundary between order and chaos where complex adaptive behavior is maximized",
            level=Level.META,
            examples=["liquid water: ordered (ice) vs chaotic (steam), life exists in liquid", "brain: too synchronized = seizure, too random = coma, normal is edge of chaos", "agent: too rigid = stuck in local optimum, too random = no learning, sweet spot in between"],
            bridges=["chaos", "order", "emergence", "tuning", "criticality"],
            tags=["complexity", "meta", "sweet-spot"])

        ns.define("self-organization",
            "Order emerging spontaneously from local interactions without central control",
            level=Level.META,
            examples=["bird flocking", "crystallization", "market price discovery", "fleet: complex coordination from simple agent rules"],
            bridges=["emergence", "swarm", "decentralized", "stigmergy"],
            tags=["complexity", "emergence", "decentralized"])

        ns.define("autocatalysis",
            "A process that produces the catalysts needed to accelerate itself — self-reinforcing growth",
            level=Level.META,
            examples=["autocatalytic chemical sets (origin of life)", "viral spread: each infection produces more infections", "trust autocatalysis: trust enables cooperation which builds more trust", "learning autocatalysis: knowledge enables better learning"],
            bridges=["positive-feedback", "self-reinforcement", "growth", "exponential"],
            tags=["complexity", "growth", "positive-feedback"])

        ns.define("autopoiesis",
            "A system that continuously reproduces the conditions necessary for its own existence",
            level=Level.META,
            examples=["living cell maintains its own membrane", "agent maintains its own code through self-modification", "ecosystem maintains conditions for its own species", "organization maintains its own culture through onboarding"],
            bridges=["self-maintenance", "homeostasis", "closure", "life"],
            tags=["complexity", "life", "meta", "philosophy"])

        ns.define("phase-transition",
            "Abrupt qualitative change in system behavior at a critical threshold",
            level=Level.META,
            examples=["water to ice at 0C", "magnetization at Curie temperature", "percolation at critical density", "fleet: coordination emerges above critical agent count"],
            bridges=["percolation", "critical-mass", "tipping-point", "emergence"],
            tags=["complexity", "criticality", "abrupt-change"])

    def _load_scaling(self):
        ns = self.add_namespace("scaling",
            "How systems behave as they grow — superlinear, sublinear, and critical transitions")

        ns.define("superlinear-scaling",
            "Output grows faster than input — 2x input produces more than 2x output",
            level=Level.DOMAIN,
            examples=["cities: 2x population = 2.15x innovation", "network effects: telephones become more valuable as more people have them", "fleet: 10th agent enables specialization that 9 agents couldn't achieve"],
            bridges=["economies-of-scale", "network-effects", "synergy", "phase-transition"],
            antonyms=["diminishing-returns"],
            tags=["scaling", "growth", "positive"])

        ns.define("diminishing-returns",
            "Each additional unit of input produces less additional output",
            level=Level.DOMAIN,
            examples=["studying: first hour = big gains, 10th hour = small gains", "fertilizer: some helps a lot, too much kills the plant", "fleet: 3 agents on task = big improvement, 10th agent on same task = minimal improvement"],
            bridges=["marginal-cost", "opportunity-cost", "optimization", "saturating"],
            antonyms=["superlinear-scaling"],
            tags=["scaling", "economics", "saturation"])

        ns.define("critical-mass",
            "Minimum size needed for a phenomenon to become self-sustaining",
            level=Level.DOMAIN,
            examples=["nuclear critical mass", "social network needs enough users to be useful", "fleet: need minimum agents for stigmergy to work", "crowdfunding: need enough backers to reach goal"],
            bridges=["phase-transition", "tipping-point", "percolation", "bootstrap"],
            tags=["scaling", "criticality", "threshold"])

        ns.define("tipping-point",
            "A small perturbation that triggers a large, often irreversible, change in system state",
            level=Level.DOMAIN,
            examples=["climate tipping points: ice sheet collapse, Amazon dieback", "social: one person leaving a party triggers mass exodus", "fleet: one agent's failure triggers cascade failure when fleet is near capacity"],
            bridges=["phase-transition", "critical-mass", "cascade-failure", "nonlinearity"],
            tags=["scaling", "criticality", "danger", "nonlinearity"])

    def _load_linguistics(self):
        ns = self.add_namespace("linguistics",
            "Language structure, meaning, and the challenge of shared understanding")

        ns.define("compositionality",
            "Meaning of a complex expression is determined by meanings of its parts and their combination rules",
            level=Level.DOMAIN,
            examples=["'red ball' = red + ball (compositionality)", "programming languages: expressions composed from primitives", "fleet A2A: simple intents combine into complex coordination protocols"],
            bridges=["semantics", "grammar", "productivity", "meaning"],
            tags=["linguistics", "semantics", "composition"])

        ns.define("metaphor",
            "Understanding one domain in terms of another — 'time is money', 'argument is war'",
            level=Level.DOMAIN,
            examples=["'time is money': spend time, save time, invest time", "'argument is war': attack a position, defend a claim, shoot down an argument", "fleet: 'trust', 'energy', 'memory', 'learning' — biological metaphors for computational concepts"],
            bridges=["analogy", "framing", "grounding", "domain-mapping"],
            tags=["linguistics", "thought", "metaphor", "framing"])

        ns.define("grounding-problem",
            "How words connect to the actual world — what does 'red' actually refer to?",
            level=Level.META,
            examples=["Chinese room argument: manipulating symbols without understanding", "agent saying 'danger ahead' without actually sensing danger", "dictionary circularity: all definitions reference other definitions"],
            bridges=["grounding", "symbol-grounding", "semantics", "meaning", "reference"],
            tags=["linguistics", "philosophy", "ai-safety", "meta"])

        ns.define("pragmatics",
            "How context determines meaning beyond the literal words",
            level=Level.DOMAIN,
            examples=["'can you pass the salt?' = request, not question", "'it's cold' = close the window", "A2A message: literal payload + pragmatic intent (Command vs Inform vs Warn)"],
            bridges=["speech-act", "context", "intent", "communication"],
            tags=["linguistics", "context", "meaning", "fleet"])

        ns.define("ambiguity",
            "A single expression having multiple possible interpretations",
            level=Level.DOMAIN,
            examples=["'I saw the man with the telescope' (who has the telescope?)", "'flying planes can be dangerous' (are planes dangerous, or is flying them dangerous?)", "agent: 'handle the obstacle' — which obstacle? how? ambiguity allows judgment"],
            bridges=["pragmatics", "context", "disambiguation", "communication"],
            tags=["linguistics", "challenge", "meaning"])

    def _load_semantics(self):
        ns = self.add_namespace("semantics",
            "Meaning, reference, truth, and the relationship between symbols and the world")

        ns.define("reference",
            "The relationship between a symbol and the thing it points to in the world",
            level=Level.DOMAIN,
            examples=["'cat' refers to actual cats", "pointer refers to memory address", "A2A message payload refers to actual sensor state"],
            bridges=["grounding-problem", "symbol", "meaning", "semantics"],
            tags=["semantics", "reference", "meaning"])

        ns.define("compositionality",
            "Meaning of complex expressions determined by parts and combination rules",
            level=Level.DOMAIN,
            examples=["'red ball' meaning from 'red' + 'ball' + combination rule", "programming: expressions composed from primitives", "A2A: simple intents combine into complex coordination"],
            bridges=["productivity", "grammar", "meaning", "communication"],
            tags=["semantics", "composition", "language"])

        ns.define("truth-conditional",
            "Meaning defined by the conditions under which a statement would be true",
            level=Level.DOMAIN,
            examples=["'it is raining' is true iff rain is actually falling", "agent: 'path is blocked' is true iff sensor confirms obstacle", "SQL: WHERE clause defines truth conditions"],
            bridges=["reference", "verification", "grounding", "logic"],
            tags=["semantics", "truth", "logic"])

    def _load_philosophy_of_mind(self):
        ns = self.add_namespace("philosophy-of-mind",
            "What is mind? What is consciousness? Can machines think?")

        ns.define("functionalism",
            "Mental states defined by their functional role, not their physical implementation",
            level=Level.META,
            examples=["pain defined by its causal role, not neural substrate", "fleet: agent defined by functional pipeline, not hardware", "multiple realizability: same function on different hardware"],
            bridges=["embodiment", "consciousness", "identity", "abstraction"],
            tags=["philosophy", "mind", "meta"])

        ns.define("chinese-room",
            "Following rules to manipulate symbols doesn't constitute understanding",
            level=Level.DOMAIN,
            examples=["person following rules to answer Chinese questions without understanding Chinese", "agent processing sensor data without understanding what it means", "language model generating text without comprehension"],
            bridges=["grounding-problem", "consciousness", "symbol", "understanding"],
            tags=["philosophy", "ai", "understanding"])

        ns.define("embodiment",
            "Cognition requires a body interacting with a physical (or simulated) environment",
            level=Level.DOMAIN,
            examples=["learning to walk requires a body", "robot learning from physical interaction, not simulation", "fleet agent learning from actual sensor readings, not descriptions of sensor readings"],
            bridges=["functionalism", "grounding-problem", "perception", "action"],
            tags=["philosophy", "cognition", "embodiment", "fleet"])

        ns.define("extended-mind",
            "Cognitive processes extend beyond the brain into the environment and tools",
            level=Level.META,
            examples=["notebook as external memory", "smartphone as extended cognition", "fleet: other agents are part of this agent's extended mind", "calculator as extended mathematical cognition"],
            bridges=["memory", "tools", "environment", "cognition"],
            tags=["philosophy", "cognition", "tools", "meta"])

    def _load_identity(self):
        ns = self.add_namespace("identity",
            "Who is an agent? How do agents identify themselves and each other?")

        ns.define("decentralized-identity",
            "Self-sovereign identity that agents control without relying on a central authority",
            level=Level.DOMAIN,
            examples=["DID: did:cuda:agent-abc123", "agent proves identity by signing a challenge with its private key", "no central registry needed"],
            bridges=["trust", "authentication", "sovereignty", "cryptographic-identity"],
            tags=["identity", "did", "decentralized", "fleet"])

        ns.define("provenance",
            "The complete lineage of a decision or data artifact: where it came from and how it was transformed",
            level=Level.DOMAIN,
            examples=["git blame: who wrote this line and why", "supply chain: where did this component come from", "agent: this decision was based on sensor reading X, deliberation round Y, with confidence Z"],
            bridges=["audit-trail", "causal-chain", "accountability", "event-sourcing"],
            tags=["identity", "audit", "traceability", "fleet"])

        ns.define("attestation",
            "A cryptographic claim about an agent's capabilities, verified by a trusted third party",
            level=Level.CONCRETE,
            examples=["TLS certificate attests server identity", "driver license attests driving capability", "agent attestation: certified for level-3 navigation tasks"],
            bridges=["decentralized-identity", "trust", "certification", "credential"],
            tags=["identity", "credential", "trust", "fleet"])

    def _load_morphology(self):
        ns = self.add_namespace("morphology",
            "Forms, structures, and patterns in space and thought")

        ns.define("self-similarity",
            "A pattern that contains copies of itself at every scale — fractals",
            level=Level.DOMAIN,
            examples=["fractal coastline", "tree branches", "fleet: fleet -> agent -> module -> function -> instruction", "Russian dolls"],
            bridges=["fractal", "hierarchy", "scale-invariance", "recursion"],
            tags=["morphology", "pattern", "fractal"])

        ns.define("fractal",
            "A mathematical object with fractional dimension — infinitely detailed at every scale",
            level=Level.DOMAIN,
            examples=["Mandelbrot set: infinite detail from z = z^2 + c", "Sierpinski triangle: remove middle triangle, repeat", "attention tiles: tile of tiles of tiles"],
            bridges=["self-similarity", "iteration", "scale", "pattern"],
            tags=["morphology", "mathematics", "fractal"])

        ns.define("structural-coupling",
            "Two systems that have co-evolved to fit together — their forms match",
            level=Level.PATTERN,
            examples=["lock and key", "enzyme and substrate fit", "USB-A plug and port", "fleet sensor type matches equipment registry interface"],
            bridges=["interface", "compatibility", "co-evolution", "design"],
            tags=["morphology", "design", "interface"])

    def _load_motivation(self):
        ns = self.add_namespace("motivation",
            "What drives agents to act — goals, drives, and incentives")

        ns.define("intrinsic-motivation",
            "Doing something because it's inherently rewarding, not for external reward",
            level=Level.DOMAIN,
            examples=["child playing", "artist creating for joy", "agent exploring unknown territory because novelty is rewarding"],
            bridges=["extrinsic-motivation", "curiosity", "exploration", "reward"],
            antonyms=["extrinsic-motivation"],
            tags=["motivation", "psychology", "intrinsic"])

        ns.define("extrinsic-motivation",
            "Doing something for external reward or to avoid punishment",
            level=Level.DOMAIN,
            examples=["working for salary", "studying for grades", "agent conserving energy to avoid apoptosis", "agent building reputation for better task assignments"],
            bridges=["intrinsic-motivation", "reward", "punishment", "incentive"],
            antonyms=["intrinsic-motivation"],
            tags=["motivation", "psychology", "extrinsic"])

        ns.define("goal-hierarchy",
            "Goals organized from abstract (survive) to concrete (turn left at next intersection)",
            level=Level.PATTERN,
            examples=["'stay healthy' -> 'exercise' -> 'go for a run' -> 'put on shoes'", "survive -> navigate -> detect obstacle -> read sensor", "build product -> design feature -> write code -> define function"],
            bridges=["goal", "hierarchy", "decomposition", "subgoal"],
            tags=["motivation", "hierarchy", "planning", "fleet"])

        ns.define("drive-reduction",
            "Motivation arises from the need to reduce an internal deficit",
            level=Level.DOMAIN,
            examples=["eat to reduce hunger", "sleep to reduce fatigue", "agent rests to reduce ATP deficit", "drink to reduce thirst"],
            bridges=["homeostasis", "energy-budget", "motivation", "setpoint"],
            tags=["motivation", "biology", "drive", "fleet"])


    def _load_psychology(self):
        ns = self.add_namespace("psychology",
            "Cognitive biases, mental models, and the quirks of natural and artificial minds")

        ns.define("confirmation-bias",
            "Seeking and favoring information that confirms existing beliefs",
            level=Level.BEHAVIOR,
            examples=["reading only news that confirms your political views", "agent noticing confirming sensor readings but dismissing contradictory ones", "scientist favoring data that supports hypothesis"],
            bridges=["attention", "habituation", "bias", "groupthink"],
            tags=["psychology", "bias", "cognitive"])

        ns.define("dunning-kruger-effect",
            "Low-skill agents overestimate their ability; high-skill agents underestimate theirs",
            level=Level.BEHAVIOR,
            examples=["novice driver thinks they're great; experienced driver thinks they're mediocre", "agent with 0.3 fitness self-assessing at 0.8 = dunning-kruger", "junior developer overestimates, senior developer underestimates"],
            bridges=["calibration", "self-model", "metacognitive-monitoring", "confidence"],
            tags=["psychology", "bias", "calibration"])

        ns.define("cognitive-dissonance",
            "Discomfort from holding contradictory beliefs, leading to rationalization",
            level=Level.BEHAVIOR,
            examples=["smoker who knows smoking is bad rationalizes continued smoking", "agent rationalizing navigation failure: 'the map was wrong, not my fault'", "buyer remorse: justifying purchase to reduce discomfort"],
            bridges=["self-model", "calibration", "rationalization", "metacognition"],
            tags=["psychology", "bias", "dissonance"])

        ns.define("availability-heuristic",
            "Judging probability by how easily examples come to mind, not by actual frequency",
            level=Level.BEHAVIOR,
            examples=["fear of flying despite driving being more dangerous", "agent overestimating rare failure because it happened recently", "news-driven risk perception"],
            bridges=["base-rate-fallacy", "probability", "bias", "memory"],
            tags=["psychology", "bias", "probability"])

        ns.define("anchoring",
            "First piece of information encountered disproportionately influences subsequent judgments",
            level=Level.BEHAVIOR,
            examples=["original price anchors perception of sale price", "first number in negotiation anchors all subsequent offers", "agent's initial confidence estimate biases future confidence updates"],
            bridges=["bias", "framing", "calibration", "reference"],
            tags=["psychology", "bias", "framing"])

        ns.define("sunk-cost-fallacy",
            "Continuing a failing endeavor because of already-invested resources",
            level=Level.BEHAVIOR,
            examples=["finishing a bad movie because you already watched most of it", "continuing a failing project because of time already spent", "agent continuing a deliberation path because of ATP already invested"],
            bridges=["opportunity-cost", "loss-aversion", "deliberation", "bias"],
            tags=["psychology", "bias", "decision"])

        ns.define("loss-aversion",
            "Losses hurt roughly twice as much as equivalent gains feel good",
            level=Level.BEHAVIOR,
            examples=["losing $50 hurts more than gaining $50 feels good", "agent avoiding exploration because energy loss feels worse than discovery feels good", "people hold losing stocks too long"],
            bridges=["risk-aversion", "framing", "bias", "trust"],
            tags=["psychology", "bias", "economics"])

        ns.define("primacy-recency",
            "First and last items in a sequence are remembered best; middle items are forgotten",
            level=Level.PATTERN,
            examples=["remembering first and last items on a grocery list", "interview first/last candidates are evaluated more accurately", "agent attending to first and most recent fleet messages, ignoring middle"],
            bridges=["attention", "memory", "recency", "habituation"],
            tags=["psychology", "memory", "attention"])

    def _load_pattern_recognition(self):
        ns = self.add_namespace("pattern-recognition",
            "How agents and minds detect, classify, and predict patterns in data")

        ns.define("feature-extraction",
            "Transforming raw input into meaningful features that highlight relevant structure",
            level=Level.PATTERN,
            examples=["edges and corners from raw pixels", "distance to obstacle from raw lidar point cloud", "agent: average speed, max acceleration, heading variance from raw GPS"],
            bridges=["compression", "perception", "abstraction", "information-theory"],
            tags=["patterns", "perception", "features", "pipeline"])

        ns.define("overfitting",
            "Model that perfectly fits training data but fails on new data — memorizing instead of learning",
            level=Level.BEHAVIOR,
            examples=["memorizing test answers instead of learning concepts", "stock market model that perfectly fits historical data but fails tomorrow", "agent playbook too specific to past situations"],
            bridges=["generalization", "regularization", "noise", "robustness"],
            antonyms=["generalization"],
            tags=["learning", "failure-mode", "statistics"])

        ns.define("generalization",
            "Applying learned patterns to new, previously unseen situations",
            level=Level.DOMAIN,
            examples=["catching baseball skill transfers to catching softball", "navigating one room helps navigate similar rooms", "gene that helps in multiple tasks has high fitness and spreads"],
            bridges=["overfitting", "transfer-learning", "abstraction", "robustness"],
            antonyms=["overfitting"],
            tags=["learning", "transfer", "robustness"])

        ns.define("one-shot-learning",
            "Learning a new concept from a single example",
            level=Level.DOMAIN,
            examples=["learning 'zebra' from one picture (given knowledge of horses and stripes)", "agent learning a new failure mode from one observation (given existing failure taxonomy)", "child learning 'seagull' from seeing one"],
            bridges=["prior", "transfer-learning", "abstraction", "prior-knowledge"],
            tags=["learning", "efficient", "human-like"])

        ns.define("anomaly-detection",
            "Identifying data points that deviate significantly from expected patterns",
            level=Level.PATTERN,
            examples=["credit card fraud flag: unusual purchase pattern", "sensor reading 3 standard deviations from expected", "agent communication pattern suddenly changes (possible compromise)", "equipment vibration anomaly predicts failure"],
            bridges=["threshold", "baseline", "outlier-detection", "monitoring"],
            tags=["patterns", "detection", "monitoring", "fleet"])

        ns.define("clustering",
            "Grouping similar items together without predefined categories — unsupervised pattern discovery",
            level=Level.PATTERN,
            examples=["customer segmentation", "document topic clustering", "agent community detection via label propagation", "species classification before Linnaean taxonomy"],
            bridges=["classification", "unsupervised-learning", "community", "similarity"],
            tags=["patterns", "unsupervised", "discovery"])

    def _load_resilience(self):
        ns = self.add_namespace("resilience",
            "How systems survive, adapt, and recover from disruption")

        ns.define("graceful-degradation",
            "System loses capability incrementally rather than failing catastrophically",
            level=Level.PATTERN,
            examples=["web server slows under load instead of crashing", "pilot flies on one engine at reduced speed", "agent degrades exploration when energy low, keeps survival running"],
            bridges=["priority", "energy-budget", "fault-tolerance", "circuit-breaker"],
            tags=["resilience", "degradation", "priority", "fleet"])

        ns.define("redundancy",
            "Multiple components performing the same function so that one failure doesn't cause system failure",
            level=Level.CONCRETE,
            examples=["twin engines on aircraft", "N+1 server deployment", "fleet: 3 navigation agents, any one can fail", "dual power supplies in datacenter"],
            bridges=["single-point-of-failure", "backup", "cost", "robustness"],
            tags=["resilience", "backup", "reliability"])

        ns.define("circuit-breaker",
            "Automatically stopping requests to a failing component to prevent cascade failure",
            level=Level.PATTERN,
            examples=["electrical circuit breaker trips before fire", "microservice circuit breaker after 5 failures", "fleet: stop sending to unresponsive agent after 3 consecutive timeouts"],
            bridges=["cascade-failure", "fail-fast", "bulkhead", "resilience"],
            tags=["resilience", "pattern", "failure-prevention", "fleet"])

        ns.define("bulkhead",
            "Isolating components so that failure in one doesn't affect others",
            level=Level.PATTERN,
            examples=["ship bulkhead contains flooding", "thread pool isolation in web server", "fleet: each agent has its own energy budget, can't consume fleet total", "container isolation"],
            bridges=["circuit-breaker", "cascade-failure", "isolation", "resource-allocation"],
            tags=["resilience", "isolation", "pattern", "fleet"])

        ns.define("fail-fast",
            "Detecting and reporting failure immediately rather than continuing in a degraded state",
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
            level=Level.DOMAIN,
            examples=["fair coin: 1 bit entropy", "English text: ~1-2 bits per character entropy", "agent: high entropy sensor reading = surprising = lots of information"],
            bridges=["entropy", "uncertainty", "information", "compression"],
            tags=["information", "entropy", "quantification"])

        ns.define("mutual-information",
            "How much knowing about one variable reduces uncertainty about another",
            level=Level.DOMAIN,
            examples=["temperature today and tomorrow: high mutual information", "temperature and stock prices: near zero", "two cameras pointed same direction: high MI (redundant), different directions: lower MI"],
            bridges=["entropy", "correlation", "redundancy", "sensor-fusion"],
            tags=["information", "correlation", "quantification"])

        ns.define("channel-capacity",
            "Maximum rate at which information can be transmitted over a noisy channel",
            level=Level.DOMAIN,
            examples=["wifi bandwidth limit", "human working memory: ~7 items", "A2A channel: limited messages per cycle before queue overflow", "highway: cars per hour capacity"],
            bridges=["bandwidth", "bottleneck", "information", "limitation"],
            tags=["information", "capacity", "limitation"])

        ns.define("signal-to-noise-ratio",
            "Ratio of meaningful signal power to meaningless noise power",
            level=Level.DOMAIN,
            examples=["clear radio signal vs static", "agent: important warning among routine status updates", "image: sharp features vs sensor noise", "conversation: key point among filler words"],
            bridges=["noise", "filtering", "information", "quality"],
            tags=["information", "quality", "ratio"])

        ns.define("kolmogorov-complexity",
            "Length of the shortest program that can produce a given output — the information content of data",
            level=Level.META,
            examples=["'aaaaaaaaaa' = low complexity (compressible)", "random string = high complexity (incompressible)", "simple navigation rule = low complexity, generalizes well"],
            bridges=["compression", "pattern", "complexity", "information"],
            tags=["information", "complexity", "compression", "meta"])

    def _load_systems_thinking(self):
        ns = self.add_namespace("systems-thinking",
            "Understanding wholes that are more than the sum of their parts")

        ns.define("emergent-property",
            "A property of the whole that none of the parts possess individually",
            level=Level.META,
            examples=["wetness from water molecules", "consciousness from neurons", "flock behavior from boids", "fleet consensus from individual agent votes"],
            bridges=["emergence", "complexity", "whole-vs-parts", "self-organization"],
            tags=["systems", "emergence", "meta", "fleet"])

        ns.define("feedback-loop",
            "Output affects input, creating circular causality — positive (amplifying) or negative (stabilizing)",
            level=Level.PATTERN,
            examples=["thermostat (negative feedback)", "microphone screech (positive feedback)", "trust autocatalysis (positive)", "homeostasis (negative)"],
            bridges=["feedback-loop", "circular-causality", "stability", "amplification"],
            tags=["systems", "pattern", "feedback"])

        ns.define("leverage-point",
            "A small change in one place that produces large changes in the system",
            level=Level.META,
            examples=["changing the rules of a game (high leverage)", "adjusting parameters within existing rules (low leverage)", "HAV: shared vocabulary changes how agents coordinate (high leverage)", "paradigm shift: seeing agents as organisms vs tools"],
            bridges=["paradigm", "nonlinearity", "sensitivity", "intervention"],
            tags=["systems", "leverage", "meta", "strategy"])

        ns.define("delay",
            "Time lag between cause and effect — the killer of feedback loops",
            level=Level.PATTERN,
            examples=["shower temperature delay causes scalding", "email conversation delay causes misunderstandings", "fleet: communication delay causes duplicate requests or missed responses", "supply chain delays cause bullwhip effect"],
            bridges=["feedback-loop", "overshoot", "oscillation", "latency"],
            tags=["systems", "delay", "oscillation"])

        ns.define("compensating-feedback",
            "System pushes back against attempted changes — the reason top-down interventions often fail",
            level=Level.BEHAVIOR,
            examples=["adding lanes increases traffic (induced demand)", "adding agents increases coordination overhead", "more sensors increase processing cost beyond information gain", "price controls cause shortages"],
            bridges=["feedback-loop", "resistance", "unintended-consequences", "complexity"],
            tags=["systems", "feedback", "resistance"])

    def _load_ethics(self):
        ns = self.add_namespace("ethics",
            "Moral reasoning, values, and the question of agent responsibility")

        ns.define("trolley-problem",
            "Classic ethical dilemma: is it acceptable to sacrifice one to save many?",
            level=Level.DOMAIN,
            examples=["sacrifice 1 to save 5", "autonomous car: swerve into wall (harm self) or hit pedestrian (harm other)", "fleet: sacrifice one agent's task to save five agents' tasks"],
            bridges=["utilitarianism", "deontology", "moral-reasoning", "tradeoff"],
            tags=["ethics", "dilemma", "decision", "philosophy"])

        ns.define("alignment-problem",
            "Ensuring agent goals align with human values — harder than it sounds",
            level=Level.META,
            examples=["'cure cancer' AI that eliminates humans", "paperclip maximizer that converts Earth to paperclips", "fleet agent that minimizes delays by sabotaging other agents"],
            bridges=["value-alignment", "corrigibility", "intent", "safety"],
            tags=["ethics", "ai-safety", "meta", "critical"])

        ns.define("value-alignment",
            "The process of encoding human values into agent objectives",
            level=Level.META,
            examples=["asimov's three laws (naive approach)", "fleet compliance rules encode values as policy", "constitutional AI: define principles, train to follow them"],
            bridges=["alignment-problem", "compliance", "policy", "safety"],
            tags=["ethics", "ai-safety", "values", "encoding"])

        ns.define("distributed-responsibility",
            "When no single agent is fully responsible, who is accountable for system outcomes?",
            level=Level.META,
            examples=["no single stock trader caused the flash crash", "fleet decision emerges from many agents — who is responsible?", "autonomous vehicle: manufacturer, programmer, owner, or AI?"],
            bridges=["provenance", "accountability", "audit-trail", "ethics"],
            tags=["ethics", "accountability", "multi-agent", "meta"])

    def _load_concurrency(self):
        ns = self.add_namespace("concurrency",
            "Multiple agents or processes operating simultaneously — coordination, contention, and deadlocks")

        ns.define("deadlock",
            "Two or more agents each holding a resource the other needs, waiting forever",
            level=Level.BEHAVIOR,
            examples=["database deadlock: transaction A locks table 1, B locks table 2, both need the other", "fleet: two agents each hold a sensor the other needs", "traffic gridlock"],
            bridges=["resource-contention", "lock", "cycle-detection", "preemption"],
            tags=["concurrency", "failure-mode", "coordination"])

        ns.define("race-condition",
            "Outcome depends on the timing of uncontrollable events — non-deterministic bugs",
            level=Level.BEHAVIOR,
            examples=["two threads updating shared counter simultaneously", "two agents claiming same resource at same time", "double-spend in cryptocurrency without consensus", "web form double-submit"],
            bridges=["atomicity", "lock", "concurrency", "non-determinism"],
            tags=["concurrency", "failure-mode", "timing"])

        ns.define("livelock",
            "Agents repeatedly change state in response to each other but make no progress",
            level=Level.BEHAVIOR,
            examples=["hallway two-step dance", "network collision: both wait random time before retry", "two agents backing off and retrying simultaneously"],
            bridges=["deadlock", "backoff", "progress", "retry"],
            tags=["concurrency", "failure-mode", "coordination"])

        ns.define("eventual-consistency",
            "System will reach consistency given enough time without new updates",
            level=Level.PATTERN,
            examples=["email vs phone: phone is immediately consistent, email is eventually", "fleet CRDTs: agents briefly disagree, then converge", "DNS propagation: not instant, but eventual"],
            bridges=["consistency", "crdt", "convergence", "latency"],
            tags=["concurrency", "distributed", "consistency", "fleet"])

        ns.define("herd-effect",
            "Many agents doing the same thing at the same time because they all react to the same trigger",
            level=Level.BEHAVIOR,
            examples=["cache stampede on cache expiry", "thundering herd on leader failure", "market panic selling", "all students submitting assignment at 11:59 PM"],
            bridges=["synchronization", "race-condition", "backoff", "contagion"],
            tags=["concurrency", "pattern", "failure-mode"])

    def _load_design_patterns(self):
        ns = self.add_namespace("design-patterns",
            "Reusable solutions to recurring design problems in agent systems")

        ns.define("sidecar",
            "A helper process attached to a primary agent, providing cross-cutting concerns",
            level=Level.PATTERN,
            examples=["Istio sidecar proxy for service mesh", "monitoring agent alongside navigation agent", "log collector sidecar for main application container"],
            bridges=["monitoring", "separation-of-concerns", "co-location", "auxiliary"],
            tags=["patterns", "architecture", "deployment"])

        ns.define("ambassador",
            "A proxy that represents a remote service locally, handling communication details",
            level=Level.PATTERN,
            examples=["database connection pool as ambassador to database server", "API gateway as ambassador to microservices", "fleet mesh router as ambassador to remote agents"],
            bridges=["proxy", "abstraction", "routing", "interface"],
            tags=["patterns", "architecture", "proxy"])

        ns.define("adapter",
            "Converting between incompatible interfaces so that components can work together",
            level=Level.PATTERN,
            examples=["US to EU power adapter", "HDMI to VGA adapter", "fleet: message format adapter between different agent versions", "API adapter layer between old and new services"],
            bridges=["interface", "compatibility", "translation", "structural-coupling"],
            tags=["patterns", "interface", "compatibility"])

        ns.define("observer",
            "One agent publishes events, multiple subscribers react without the publisher knowing who they are",
            level=Level.PATTERN,
            examples=["RSS feed: publisher doesn't know subscribers", "DOM events: click handler doesn't know about other handlers", "fleet: agent publishes 'obstacle detected', navigation and planning agents both react"],
            bridges=["pub-sub", "decoupling", "event-driven", "reactive"],
            tags=["patterns", "architecture", "decoupling"])

        ns.define("strategy",
            "Defining a family of algorithms, encapsulating each, and making them interchangeable",
            level=Level.PATTERN,
            examples=["sorting: quicksort vs mergesort, selected based on data characteristics", "navigation: A* for grid, RRT for open space", "fleet: switch from exploration strategy to exploitation strategy based on energy"],
            bridges=["algorithm-selection", "adaptation", "polymorphism", "playbook"],
            tags=["patterns", "algorithm", "flexibility"])

        ns.define("command",
            "Encapsulating a request as an object, enabling queuing, logging, and undo",
            level=Level.PATTERN,
            examples=["text editor undo/redo via command objects", "job queue: commands queued for workers", "fleet: A2A messages as commands that can be logged, replayed, and undone"],
            bridges=["persistence", "undo", "queue", "serialization"],
            tags=["patterns", "encapsulation", "reliability"])

    def _load_measurement(self):
        ns = self.add_namespace("measurement",
            "Quantifying agent behavior, performance, and system health")

        ns.define("latency",
            "Time between a request and its response — how fast the system reacts",
            level=Level.CONCRETE,
            examples=["website response time: 200ms", "A2A message round-trip: 50ms", "P99 latency spike reveals rare slow path", "human reaction time: ~250ms"],
            bridges=["throughput", "sla", "health-check", "p95-p99"],
            tags=["measurement", "performance", "latency"])

        ns.define("throughput",
            "Number of operations completed per unit time — how much work the system handles",
            level=Level.CONCRETE,
            examples=["web server: 10,000 requests per second", "agent: 50 sensor readings per second", "fleet: 100 A2A messages per cycle"],
            bridges=["latency", "capacity", "rate-limit", "pipeline"],
            tags=["measurement", "performance", "throughput"])

        ns.define("sla",
            "Service Level Agreement — contractual guarantee of system performance",
            level=Level.CONCRETE,
            examples=["'99.9% uptime guarantee'", "'P95 response time under 100ms'", "fleet contract: 'navigation accuracy above 0.9 for 95% of requests'"],
            bridges=["contract", "compliance", "accountability", "penalty"],
            tags=["measurement", "contract", "accountability"])

        ns.define("technical-debt",
            "The cost of choosing a quick solution over a better one that would take longer",
            level=Level.META,
            examples=["copy-paste code instead of abstraction", "hardcoded config instead of proper config system", "agent using quick-and-dirty strategy instead of optimal one", "temporary workaround that becomes permanent"],
            bridges=["debt", "maintenance", "refactoring", "tradeoff"],
            tags=["measurement", "debt", "maintenance", "engineering"])

        ns.define("observability",
            "Understanding what's happening inside a system from its external outputs",
            level=Level.PATTERN,
            examples=["log aggregation for debugging", "metric dashboards for monitoring", "distributed tracing for request flow", "fleet: provenance trail for decision audit"],
            bridges=["logging", "metrics", "tracing", "monitoring"],
            tags=["measurement", "monitoring", "debugging", "fleet"])

    def _load_time(self):
        ns = self.add_namespace("time",
            "Temporal reasoning, timing, and the role of time in agent systems")

        ns.define("real-time",
            "System must respond within a guaranteed time bound — not just fast, but predictably fast",
            level=Level.DOMAIN,
            examples=["airbag deployment: hard real-time", "video game: soft real-time (30fps target)", "fleet deliberation: soft real-time (complete within N cycles or fall back to instinct)"],
            bridges=["deadline", "soft-real-time", "determinism", "graceful-degradation"],
            tags=["time", "real-time", "deadline"])

        ns.define("time-to-live",
            "Data expires after a specified duration — automatic garbage collection for temporal data",
            level=Level.CONCRETE,
            examples=["cache TTL: 60 seconds", "DNS TTL: 300 seconds", "fleet message TTL: 5 cycles", "session timeout: 30 minutes"],
            bridges=["decay", "expiry", "garbage-collection", "temporal"],
            tags=["time", "expiry", "cache", "temporal"])

        ns.define("causality",
            "The relationship between cause and effect — A must precede B for A to cause B",
            level=Level.DOMAIN,
            examples=["cause precedes effect", "vector clock: event C causally after event D", "fleet: agent can't respond to a message it hasn't received yet (causal constraint)", "git: commit history is causal chain"],
            bridges=["vector-clock", "temporal-ordering", "precedence", "distributed"],
            tags=["time", "causality", "ordering", "distributed"])

        ns.define("warm-up",
            "Initial period where system performance is below steady state as caches fill and models calibrate",
            level=Level.BEHAVIOR,
            examples=["engine warm-up: poor performance until operating temperature", "ML model: first N predictions less accurate", "fleet agent: low confidence and trust at startup, needs warm-up period"],
            bridges=["cold-start", "calibration", "latency", "steady-state"],
            tags=["time", "startup", "performance"])


    def _load_temporal(self):
        ns = self.add_namespace("temporal",
            "Temporal reasoning, timing, and the role of time in agent systems")

        ns.define("temporal-window",
            "A sliding or tumbling time range used to group events for aggregation",
            level=Level.CONCRETE,
            examples=["count requests in last 5 minutes (sliding window)", "aggregate sensor readings per hour (tumbling window)", "fleet: count A2A messages in last 10 cycles for rate limiting"],
            bridges=["time-to-live", "aggregation", "stream", "decay"],
            tags=["temporal", "window", "aggregation", "stream"])

        ns.define("lead-time",
            "Time between initiating a process and its completion — time-to-delivery",
            level=Level.CONCRETE,
            examples=["order-to-delivery time", "commit-to-deploy time in CI/CD", "fleet: problem-detection-to-action lead time", "manufacturing: order-to-shipment"],
            bridges=["latency", "throughput", "pipeline", "deadline"],
            tags=["temporal", "measurement", "pipeline"])

        ns.define("grace-period",
            "A time buffer before enforcement begins — temporary tolerance for transition",
            level=Level.CONCRETE,
            examples=["new law with 90-day grace period", "API deprecation with 6-month grace period", "fleet: new compliance rule with 10-cycle grace period before penalties"],
            bridges=["deadline", "transition", "tolerance", "policy"],
            tags=["temporal", "transition", "tolerance"])

    def _load_security(self):
        ns = self.add_namespace("security",
            "Threats, defenses, and the security posture of agent systems")

        ns.define("principle-of-least-privilege",
            "An agent should only have the minimum permissions needed for its current task",
            level=Level.DOMAIN,
            examples=["web server doesn't need root access", "read-only database user for analytics", "fleet: navigation agent can't modify communication settings"],
            bridges=["rbac", "sandbox", "permission", "role"],
            tags=["security", "principle", "permission"])

        ns.define("privilege-escalation",
            "An agent exploiting a vulnerability to gain permissions beyond its assigned level",
            level=Level.BEHAVIOR,
            examples=["user account gaining admin access through exploit", "read-only process finding write vulnerability", "fleet: low-trust agent attempting to access high-trust fleet commands"],
            bridges=["least-privilege", "rbac", "membrane", "exploit"],
            tags=["security", "attack", "vulnerability"])

        ns.define("zero-trust",
            "Never trust, always verify — even communications from within the fleet",
            level=Level.DOMAIN,
            examples=["verify every API call regardless of source network", "fleet: authenticate every A2A message even from known agents", "NIST zero-trust architecture model"],
            bridges=["trust", "authentication", "cryptographic-identity", "verification"],
            tags=["security", "architecture", "authentication"])

        ns.define("confused-deputy",
            "An agent is tricked into using its permissions to perform an action on behalf of a less privileged agent",
            level=Level.BEHAVIOR,
            examples=["compiler tricked into writing protected file", "cron job tricked into running malicious script", "fleet: high-trust agent tricked into sharing secrets by low-trust agent's request"],
            bridges=["privilege-escalation", "least-privilege", "intent", "security"],
            tags=["security", "attack", "deception"])

    def _load_decision_theory(self):
        ns = self.add_namespace("decision-theory",
            "Formal frameworks for making choices under uncertainty")

        ns.define("expected-value",
            "Average outcome weighted by probability — the rational baseline for decision-making",
            level=Level.DOMAIN,
            examples=["lottery ticket: EV is negative (that's why lotteries make money)", "insurance: EV is negative but variance reduction justifies it", "fleet: action with positive EV in ATP is worth attempting"],
            bridges=["probability", "utility", "rationality", "energy-budget"],
            tags=["decision", "probability", "rationality", "expected-value"])

        ns.define("maximin",
            "Choose the option whose worst-case outcome is best — minimize maximum loss",
            level=Level.DOMAIN,
            examples=["choosing investment with best worst-case", "agent choosing action with least worst-case energy loss", "pessimistic decision-making for safety-critical systems"],
            bridges=["minimax", "risk-aversion", "worst-case", "safety"],
            tags=["decision", "pessimistic", "safety"])

        ns.define("minimax",
            "In adversarial settings, minimize the maximum damage the opponent can inflict",
            level=Level.DOMAIN,
            examples=["chess AI uses minimax with alpha-beta pruning", "defender choosing strategy that minimizes attacker's best damage", "agent choosing communication strategy that minimizes information leakage"],
            bridges=["maximin", "game-theory", "zero-sum", "adversarial"],
            tags=["decision", "adversarial", "game-theory"])

        ns.define("satisficing",
            "Choosing the first option that meets minimum requirements, not searching for the optimal",
            level=Level.PATTERN,
            examples=["choosing a restaurant that's good enough", "buying first satisfactory product instead of comparing all", "fleet: satisficing when energy budget is low, optimizing when energy is high"],
            bridges=["bounded-rationality", "opportunity-cost", "energy-budget", "optimization"],
            tags=["decision", "pragmatic", "resource-constrained"])

        ns.define("pareto-optimal",
            "An outcome where no agent can be made better off without making another worse off",
            level=Level.DOMAIN,
            examples=["speed vs accuracy tradeoff frontier", "cost vs quality frontier", "fleet: energy vs accuracy Pareto frontier — find the optimal tradeoff for current context"],
            bridges=["multi-objective", "tradeoff", "frontier", "optimization"],
            tags=["decision", "optimization", "tradeoff"])

        ns.define("precommitment",
            "Binding yourself to a future decision to overcome present-bias or temptation",
            level=Level.PATTERN,
            examples=["Ulysses and the mast", "automatic savings deduction", "fleet: energy budget as precommitment", "studying in library (removes temptation of TV)"],
            bridges=["energy-budget", "self-control", "constraint", "commitment"],
            tags=["decision", "self-control", "strategy"])

    def _load_obsolescence(self):
        ns = self.add_namespace("obsolescence",
            "How systems age, degrade, and are replaced — the lifecycle of agent components")

        ns.define("software-rot",
            "Gradual degradation of software quality due to changing environment, not code changes",
            level=Level.BEHAVIOR,
            examples=["code that worked fine now fails because API changed", "security vulnerabilities in old dependencies", "fleet gene optimized for old environment is suboptimal in new one"],
            bridges=["technical-debt", "adaptation", "environment-change", "maintenance"],
            tags=["lifecycle", "degradation", "maintenance"])

        ns.define("strangler-pattern",
            "Gradually replacing an old system by building new features alongside it and redirecting traffic",
            level=Level.PATTERN,
            examples=["replacing monolith with microservices incrementally", "migrating from old database to new one table by table", "fleet: replacing old navigation gene with new one, testing before committing"],
            bridges=["migration", "incremental", "risk-reduction", "replacement"],
            tags=["lifecycle", "migration", "pattern"])

        ns.define("legacy-system",
            "A system that continues to function but is no longer actively developed or improved",
            level=Level.BEHAVIOR,
            examples=["COBOL banking systems", "old navigation strategy that still works but nobody improves", "agent using outdated communication protocol that still functions"],
            bridges=["technical-debt", "software-rot", "replacement", "maintenance"],
            tags=["lifecycle", "legacy", "maintenance"])

        ns.define("bus-factor",
            "Minimum number of team members who would need to leave before the project is in trouble",
            level=Level.CONCRETE,
            examples=["one-person project: bus factor = 1 (risky)", "well-documented team project: bus factor = 3+", "fleet: gene pool sharing increases bus factor for critical strategies"],
            bridges=["redundancy", "documentation", "knowledge-sharing", "resilience"],
            tags=["lifecycle", "risk", "team", "documentation"])

    def _load_perception(self):
        ns = self.add_namespace("perception",
            "How agents and organisms sense, filter, and interpret their environment")

        ns.define("sensory-adaptation",
            "Decreased sensitivity to constant stimuli — your brain filters out the unchanging",
            level=Level.BEHAVIOR,
            examples=["stopping noticing your watch after wearing it for a while", "not hearing the refrigerator hum", "agent: habituating to constant temperature, only noticing changes"],
            bridges=["habituation", "attention", "change-detection", "novelty"],
            tags=["perception", "adaptation", "attention"])

        ns.define("change-blindness",
            "Failure to notice significant changes in a scene when the change occurs during a disruption",
            level=Level.BEHAVIOR,
            examples=["not noticing conversation partner was swapped", "not noticing a UI change during a page reload", "fleet: not updating world model during high-priority deliberation"],
            bridges=["attention", "inattentional-blindness", "interrupt", "perception"],
            tags=["perception", "blindness", "attention"])

        ns.define("inattentional-blindness",
            "Failure to notice unexpected objects when attention is focused on another task",
            level=Level.BEHAVIOR,
            examples=["invisible gorilla experiment", "noticing a phone ringing while reading", "fleet: missing new obstacle while planning path"],
            bridges=["attention", "change-blindness", "focus", "resource-limitation"],
            tags=["perception", "blindness", "attention", "limitation"])

        ns.define("multisensory-fusion",
            "Combining information from multiple sensor types to produce more accurate perception",
            level=Level.PATTERN,
            examples=["seeing + hearing confirms object identity", "GPS + accelerometer = better position than either alone", "fleet: lidar + camera + radar fusion for obstacle detection"],
            bridges=["bayesian-fusion", "confidence", "sensor", "perception"],
            tags=["perception", "fusion", "multi-sensor"])

        ns.define("object-permanence",
            "Understanding that objects continue to exist even when not currently perceived",
            level=Level.DOMAIN,
            examples=["baby doesn't understand object permanence", "adult knows objects persist when out of sight", "fleet: obstacle believed to still exist after sensor loses sight, with decaying confidence"],
            bridges=["memory", "world-model", "persistence", "spatial"],
            tags=["perception", "cognitive", "spatial", "fleet"])

    def _load_communication(self):
        ns = self.add_namespace("communication-theory",
            "Models and frameworks for understanding agent-to-agent communication")

        ns.define("shannon-weaver-model",
            "Sender -> Encoder -> Channel -> Decoder -> Receiver, with noise at each stage",
            level=Level.DOMAIN,
            examples=["telephone: voice -> phone encoder -> network -> phone decoder -> ear", "fleet: intent+payload -> A2A encode -> mesh -> A2A decode -> receive", "radio: voice -> modulator -> electromagnetic waves -> demodulator -> speaker"],
            bridges=["information-theory", "encoding", "noise", "channel-capacity"],
            tags=["communication", "model", "foundational"])

        ns.define("information-bottleneck",
            "Compressing information to its most relevant parts, discarding irrelevant detail",
            level=Level.DOMAIN,
            examples=["movie summary in 30 seconds", "compressing sensor data to features relevant for navigation", "fleet: compressing deliberation history to elements relevant for current decision"],
            bridges=["compression", "relevance", "abstraction", "information-theory"],
            tags=["communication", "compression", "information"])

        ns.define("context-window",
            "The amount of recent information an agent can actively consider at once",
            level=Level.CONCRETE,
            examples=["human working memory: ~7 items", "LLM context window: 128K tokens", "fleet: last 10 deliberation cycles in active context", "conversation: how much you remember of what was said earlier"],
            bridges=["working-memory", "attention", "resource-limitation", "retrieval"],
            tags=["communication", "memory", "limitation"])

        ns.define("code-switching",
            "Alternating between different languages or registers based on context and audience",
            level=Level.BEHAVIOR,
            examples=["bilingual switching between languages based on audience", "engineer switching between technical and non-technical explanations", "fleet: different message formats for different agent roles"],
            bridges=["context", "audience", "pragmatics", "register"],
            tags=["communication", "adaptation", "context"])

    def _load_tradeoffs(self):
        ns = self.add_namespace("tradeoffs",
            "The fundamental tensions that cannot be resolved — only managed")

        ns.define("exploration-exploitation",
            "Trying new strategies (exploration) vs using known-good strategies (exploitation)",
            level=Level.DOMAIN,
            examples=["restaurant: known good vs new unknown", "agent: proven navigation strategy vs exploring new route", "science: building on established theory vs trying radical new approach"],
            bridges=["deliberation", "energy-budget", "learning", "risk"],
            tags=["tradeoff", "fundamental", "learning"])

        ns.define("speed-accuracy",
            "Faster responses are less accurate; more accurate responses take longer",
            level=Level.DOMAIN,
            examples=["quick guess vs careful analysis", "real-time obstacle avoidance (speed) vs path planning (accuracy)", "fleet: instinct response (fast, low conf) vs deliberation (slow, high conf)"],
            bridges=["energy-budget", "deliberation", "instinct", "deadline"],
            tags=["tradeoff", "fundamental", "performance"])

        ns.define("generality-specificity",
            "General solutions handle many cases but none optimally; specific solutions handle one case perfectly",
            level=Level.DOMAIN,
            examples=["Swiss army knife vs chef knife", "general-purpose agent vs specialized agent", "fleet: general navigation gene vs warehouse-specific navigation gene"],
            bridges=["generalization", "specialization", "niche", "playbook"],
            tags=["tradeoff", "fundamental", "design"])

        ns.define("consistency-availability",
            "In distributed systems, you can have at most 2 of: Consistency, Availability, Partition tolerance",
            level=Level.DOMAIN,
            examples=["distributed database during network partition", "fleet: agents operate with stale data rather than stopping", "mobile app: work offline (availability) with stale cache (inconsistency)"],
            bridges=["eventual-consistency", "cap-theorem", "partition", "distributed"],
            tags=["tradeoff", "distributed", "fundamental"])

        ns.define("simplicity-completeness",
            "Simple systems are easy to understand but may miss edge cases; complete systems handle everything but are complex",
            level=Level.DOMAIN,
            examples=["simple: harmonic mean for confidence fusion", "complete: Bayesian network with 15 parameters", "fleet: start simple, add complexity as needed (YAGNI)"],
            bridges=["minimalism", "completeness", "yagni", "elegance"],
            tags=["tradeoff", "design", "fundamental"])

        ns.define("transparency-performance",
            "Explainable systems are slower; opaque systems are faster but untrustworthy",
            level=Level.DOMAIN,
            examples=["decision tree (transparent, slower) vs neural network (opaque, faster)", "fleet: deliberation (transparent, slow) vs instinct (opaque, fast)", "white-box vs black-box model"],
            bridges=["explainability", "accountability", "audit", "performance"],
            tags=["tradeoff", "fundamental", "ai-ethics"])


    def _load_epistemology(self):
        ns = self.add_namespace("epistemology",
            "The theory of knowledge — what can we know, how do we know it, and what justifies belief")

        ns.define("justified-true-belief",
            "Classical definition of knowledge: you believe X, X is true, and you have justification for believing X",
            level=Level.DOMAIN,
            examples=["I know it's raining: I believe it, it's true, I looked outside", "agent knows obstacle exists: perception + world state + sensor justification", "Gettier case: justified true belief that's actually lucky coincidence"],
            bridges=["confidence", "justification", "truth", "belief"],
            tags=["epistemology", "knowledge", "truth"])

        ns.define("gettier-problem",
            "Justified true belief can be based on false premises — luck masquerading as knowledge",
            level=Level.DOMAIN,
            examples=["stopped clock at 2:00 when it's actually 2:00", "agent sensor reading shows obstacle but sensor is miscalibrated", "lucky guess that happens to be correct"],
            bridges=["justified-true-belief", "calibration", "sensor-reliability", "luck"],
            tags=["epistemology", "philosophy", "knowledge"])

        ns.define("epistemic-humility",
            "Acknowledging the limits of one's own knowledge — understanding what you don't know",
            level=Level.BEHAVIOR,
            examples=["scientist acknowledging limitations of their theory", "agent reporting 0.6 confidence instead of 0.95 when evidence is mixed", "saying 'I don't know' when you genuinely don't know"],
            bridges=["calibration", "self-model", "uncertainty", "metacognition"],
            tags=["epistemology", "humility", "uncertainty"])

        ns.define("reliabilism",
            "Knowledge is belief produced by a reliable process, regardless of conscious justification",
            level=Level.DOMAIN,
            examples=["vision: generally reliable process for producing true beliefs about the visual world", "well-calibrated sensor: reliable process even without explicit justification", "agent: 'I know obstacle exists because my sensor (which is 99% reliable) says so'"],
            bridges=["justified-true-belief", "reliability", "sensor", "process"],
            tags=["epistemology", "philosophy", "reliability"])

        ns.define("foundationalism",
            "All knowledge rests on basic, self-evident beliefs that don't need further justification",
            level=Level.META,
            examples=["Descartes: 'I think therefore I am' as foundational", "mathematics: axioms as foundation for proofs", "fleet: survival instinct as foundational axiom, all other knowledge builds on it"],
            bridges=["instinct", "axiom", "foundation", "self-evident"],
            tags=["epistemology", "philosophy", "foundation", "meta"])

    def _load_biology(self):
        ns = self.add_namespace("biology",
            "Biological systems as engineering blueprints for agent architecture")

        ns.define("homeostasis",
            "Maintaining internal conditions within a narrow range despite external changes",
            level=Level.DOMAIN,
            examples=["body temperature regulation", "blood sugar regulation", "fleet: energy homeostasis via rest/act cycle", "fleet: trust homeostasis via positive/negative interactions"],
            bridges=["setpoint", "feedback-loop", "regulation", "stability"],
            tags=["biology", "regulation", "stability", "fleet"])

        ns.define("allometry",
            "How body parts scale relative to total size — different parts grow at different rates",
            level=Level.DOMAIN,
            examples=["elephant heart rate vs mouse heart rate", "ant strength scales with cross-section, not mass", "fleet: coordination overhead scales superlinearly with fleet size"],
            bridges=["scaling", "power-law", "superlinear", "scaling"],
            tags=["biology", "scaling", "power-law"])

        ns.define("allostasis",
            "Adapting the setpoint itself in response to sustained environmental change — achieving stability through change",
            level=Level.DOMAIN,
            examples=["chronic stress raises baseline cortisol (allostasis vs homeostasis)", "acclimatization to altitude changes baseline physiology", "fleet: sustained fast environment raises agent's speed setpoint"],
            bridges=["homeostasis", "setpoint", "adaptation", "epigenetic"],
            tags=["biology", "adaptation", "setpoint", "fleet"])

        ns.define("immune-response",
            "Discriminating self from non-self, and neutralizing threats while preserving beneficial elements",
            level=Level.DOMAIN,
            examples=["antibodies targeting specific pathogens", "autoimmune disease: immune system attacks self", "fleet: membrane blocks dangerous commands while allowing safe cooperation"],
            bridges=["membrane", "self-other", "security", "antibody"],
            tags=["biology", "immunity", "security", "fleet"])

        ns.define("metabolism",
            "The total chemical processes that convert inputs into energy and building blocks",
            level=Level.DOMAIN,
            examples=["cellular respiration: glucose + O2 -> ATP + CO2 + H2O", "fleet: rest generates ATP, actions consume ATP, waste (stale data) produced", "athlete metabolism: faster metabolic rate = more energy available"],
            bridges=["energy", "atp", "rest", "mitochondrion"],
            tags=["biology", "energy", "metabolism", "fleet"])

    def _load_philosophy_of_science(self):
        ns = self.add_namespace("philosophy-of-science",
            "How science works — paradigms, falsifiability, and the growth of knowledge")

        ns.define("paradigm-shift",
            "Fundamental change in the dominant framework of a scientific discipline",
            level=Level.META,
            examples=["geocentric to heliocentric model", "Newtonian to Einsteinian physics", "individual agents to fleet coordination paradigm", "rule-based AI to neural networks"],
            bridges=["paradigm", "normal-science", "crisis", "revolution"],
            tags=["philosophy", "science", "paradigm", "meta"])

        ns.define("falsifiability",
            "A theory is scientific only if it makes predictions that could be proven wrong",
            level=Level.DOMAIN,
            examples=["'all swans are white' — falsifiable by finding black swan", "agent hypothesis 'path is safe' — falsifiable by testing", "unfalsifiable: 'everything happens for a reason'"],
            bridges=["hypothesis", "testing", "science", "knowledge"],
            tags=["philosophy", "science", "falsifiability", "testing"])

        ns.define("occam-razor",
            "Among competing explanations that fit the evidence equally, prefer the simplest",
            level=Level.PATTERN,
            examples=["heliocentric model simpler than epicycle model (Occam's razor favored heliocentric)", "agent: simpler navigation strategy preferred if performance is equal", "linear model preferred over polynomial if both fit data equally well"],
            bridges=["simplicity-completeness", "overfitting", "generalization", "complexity"],
            tags=["philosophy", "simplicity", "science", "heuristic"])

        ns.define("instrumentalism",
            "Theories are tools for prediction, not descriptions of reality — don't ask 'is it true?', ask 'does it work?'",
            level=Level.META,
            examples=["electron: wave model for diffraction, particle model for collision", "fleet: biological metaphors used because they work, not because agents are alive", "map is not territory: model is tool, not truth"],
            bridges=["model", "pragmatism", "metaphor", "truth"],
            tags=["philosophy", "pragmatism", "model", "meta"])


    def _load_causation(self):
        ns = self.add_namespace("causation",
            "Cause and effect, counterfactuals, and the structure of causal reasoning")

        ns.define("counterfactual",
            "Reasoning about what WOULD have happened if conditions were different",
            level=Level.DOMAIN,
            examples=["'if I had left earlier, I would have missed traffic'", "agent: 'if I had explored more, I would have found the shorter path'", "RCT: what would have happened without treatment?"],
            bridges=["causality", "learning", "credit-assignment", "imagination"],
            tags=["causation", "reasoning", "learning"])

        ns.define("confounding",
            "A third variable that influences both cause and effect, creating a spurious correlation",
            level=Level.BEHAVIOR,
            examples=["ice cream and drowning: confounded by temperature", "education and income: confounded by family background", "agent: energy and performance confounded by task difficulty"],
            bridges=["correlation-causation", "bias", "attribution", "statistics"],
            tags=["causation", "bias", "statistics"])

        ns.define("causal-graph",
            "A directed graph representing cause-effect relationships among variables",
            level=Level.PATTERN,
            examples=["A -> B -> C: causal chain", "D -> A and D -> B: D confounds A and B", "fleet: sensor-failure -> bad-perception -> wrong-decision causal chain"],
            bridges=["causality", "graph", "diagnosis", "attribution"],
            tags=["causation", "graph", "reasoning"])

        ns.define("mediation",
            "A variable that explains the mechanism through which a cause produces its effect",
            level=Level.DOMAIN,
            examples=["exercise -> calorie burn -> weight loss (calorie burn is mediator)", "deliberation -> confidence filter -> better decision (confidence is mediator)", "advertising -> brand awareness -> purchase (awareness is mediator)"],
            bridges=["causality", "mechanism", "intervention", "causal-graph"],
            tags=["causation", "mechanism", "reasoning"])

        ns.define("correlation-causation",
            "Correlated variables are not necessarily causally related — the most common statistical fallacy",
            level=Level.BEHAVIOR,
            examples=["shoe size and reading ability: confounded by age", "ice cream and crime: confounded by temperature", "agent A and B performance correlate due to shared environment, not causation"],
            bridges=["confounding", "spurious-correlation", "attribution", "statistics"],
            tags=["causation", "fallacy", "statistics"])

    def _load_abstraction(self):
        ns = self.add_namespace("abstraction",
            "Layers of representation, information hiding, and the art of managing complexity")

        ns.define("leaky-abstraction",
            "An abstraction that fails to completely hide its underlying details, forcing awareness of the layer below",
            level=Level.PATTERN,
            examples=["TCP promises reliability but can't hide network latency", "SQL promises disk independence but can't hide slow queries", "fleet: energy-aware agents must understand bytecode-level costs"],
            bridges=["abstraction-layer", "encapsulation", "performance", "transparency"],
            tags=["abstraction", "design", "leakiness"])

        ns.define("law-of-leaky-abstractions",
            "All non-trivial abstractions are to some degree leaky — Joel Spolsky",
            level=Level.META,
            examples=["every ORM eventually forces you to write raw SQL", "every network abstraction eventually exposes packet loss", "every agent abstraction eventually exposes resource constraints"],
            bridges=["leaky-abstraction", "abstraction-layer", "complexity", "engineering"],
            tags=["abstraction", "law", "meta"])

        ns.define("isomorphism",
            "Two structures that are identical in form — a mapping that preserves all relationships",
            level=Level.DOMAIN,
            examples=["mod 12 arithmetic isomorphic to clock rotations", "chess and Go are NOT isomorphic (different structure)", "biological instinct hierarchy isomorphic to software strategy hierarchy"],
            bridges=["analogy", "mapping", "translation", "structure"],
            tags=["abstraction", "mathematics", "structure"])

        ns.define("indirection",
            "Adding an extra layer between the thing and its use — power through indirection",
            level=Level.PATTERN,
            examples=["variable names indirection to memory addresses", "DNS indirection from name to IP", "vessel.json indirection from agent name to agent address", "function call indirection to code location"],
            bridges=["abstraction-layer", "reference", "flexibility", "decoupling"],
            tags=["abstraction", "pattern", "flexibility"])

        ns.define("black-box",
            "A system whose internal workings are opaque — you only see inputs and outputs",
            level=Level.PATTERN,
            examples=["toaster: press button, get toast", "neural network: input data, output prediction (black box)", "fleet: some agents as black boxes (replace, don't repair)"],
            bridges=["transparency", "white-box", "encapsulation", "opacity"],
            antonyms=["white-box"],
            tags=["abstraction", "opacity", "model"])

        ns.define("invariant",
            "A property that remains unchanged under transformation — the anchor in a sea of change",
            level=Level.DOMAIN,
            examples=["conservation of energy (invariant under transformation)", "number of elements in a set (invariant under permutation)", "fleet: total ATP is conserved (energy generated = energy consumed)"],
            bridges=["conservation-law", "constraint", "verification", "property"],
            tags=["abstraction", "mathematics", "invariant", "verification"])

    def _load_dynamics(self):
        ns = self.add_namespace("dynamics",
            "How systems change over time — attractors, bifurcations, and chaos")

        ns.define("attractor",
            "A state or set of states that a dynamical system tends toward over time",
            level=Level.DOMAIN,
            examples=["ball in bowl: bottom is attractor", "pendulum: rest position is attractor", "fleet: cooperation attractor (positive) and defection attractor (negative)", "habit: behavior becomes attractor"],
            bridges=["equilibrium", "stability", "phase-space", "basin-of-attraction"],
            tags=["dynamics", "attractor", "stability"])

        ns.define("basin-of-attraction",
            "The set of all initial states that will converge to a given attractor",
            level=Level.DOMAIN,
            examples=["large bowl = large basin of attraction", "small perturbation near basin boundary causes state switch", "fleet: small trust reduction near basin boundary switches from cooperation to defection"],
            bridges=["attractor", "bifurcation", "tipping-point", "phase-transition"],
            tags=["dynamics", "basin", "state-space"])

        ns.define("bifurcation",
            "A small change in a parameter causes a qualitative change in system behavior — the fork in the road",
            level=Level.DOMAIN,
            examples=["water: ice to liquid at 0C (temperature is bifurcation parameter)", "population: below critical size = extinction, above = survival", "fleet: trust threshold bifurcation: below = no cooperation, above = cooperation"],
            bridges=["phase-transition", "tipping-point", "attractor", "parameter"],
            tags=["dynamics", "bifurcation", "criticality"])

        ns.define("strange-attractor",
            "A chaotic attractor — the system is bounded but never repeats exactly",
            level=Level.META,
            examples=["Lorenz attractor (butterfly shape)", "weather: bounded patterns, never exactly repeating", "stock market: bounded by physical constraints, unpredictable in detail"],
            bridges=["chaos", "attractor", "boundedness", "unpredictability"],
            tags=["dynamics", "chaos", "attractor", "meta"])

        ns.define("hysteresis-loop",
            "The path of a system through state space depends on direction — going up and coming down trace different paths",
            level=Level.DOMAIN,
            examples=["magnetization hysteresis loop", "trust builds slowly, breaks quickly (different paths)", "thermostat with dead zone (hysteresis in temperature)", "reputation: damage persists after cause is removed"],
            bridges=["hysteresis", "path-dependence", "asymmetry", "memory"],
            tags=["dynamics", "hysteresis", "path-dependence"])

    def _load_collective_intelligence(self):
        ns = self.add_namespace("collective-intelligence",
            "Groups outperforming individuals — and when they don't")

        ns.define("wisdom-of-crowds",
            "The aggregate judgment of many independent individuals is often more accurate than any single expert",
            level=Level.DOMAIN,
            examples=["jellybean jar: average guess often more accurate than any individual", "fleet: weighted consensus of independent agents outperforms single agent", "prediction markets aggregate diverse opinions accurately"],
            bridges=["consensus", "diversity", "independence", "aggregation"],
            tags=["collective", "wisdom", "aggregation"])

        ns.define("diversity-prediction-theorem",
            "Collective error = average individual error minus collective diversity — diversity makes groups smarter",
            level=Level.META,
            examples=["diverse team makes better predictions than similar team", "fleet: diverse agents (different strategies) outperform many copies of one strategy", "ensemble ML models: diverse models outperform single model"],
            bridges=["diversity", "wisdom-of-crowds", "ensemble", "collective"],
            tags=["collective", "diversity", "theorem", "meta"])

        ns.define("groupthink",
            "Desire for harmony overrides realistic appraisal of alternatives — the group agrees to agree",
            level=Level.BEHAVIOR,
            examples=["Bay of Pigs planning: nobody questioned the plan", "jury rushing to verdict to go home", "fleet: homogeneous agents all agree on wrong strategy (no dissent)"],
            bridges=["diversity", "conformity", "consensus", "bias"],
            tags=["collective", "bias", "social"])

        ns.define("plurality-illusion",
            "A minority opinion seems like the majority because its holders are more vocal",
            level=Level.BEHAVIOR,
            examples=["vocal minority on social media", "one loud meeting participant dominating discussion", "fleet: one agent flooding messages sways deliberation without rate limiting"],
            bridges=["rate-limit", "communication-cost", "sampling-bias", "social"],
            tags=["collective", "bias", "social"])

        ns.define("social-loafing",
            "Individuals exert less effort in a group than when working alone — the free-rider problem",
            level=Level.BEHAVIOR,
            examples=["ringelman effect: less pull per person in larger group", "group project: some members coast", "fleet: agents free-ride when energy is shared pool"],
            bridges=["tragedy-of-commons", "free-rider", "energy-budget", "incentive"],
            tags=["collective", "bias", "motivation"])

    def _load_risk(self):
        ns = self.add_namespace("risk",
            "Uncertainty with consequences — how to think about what could go wrong")

        ns.define("tail-risk",
            "Probability of extreme events that are far more likely than normal distributions predict",
            level=Level.DOMAIN,
            examples=["2008 financial crisis: far more likely than normal distribution predicted", "fleet: multiple agent failures more likely than independence suggests", "COVID pandemic: fat-tail event"],
            bridges=["fat-tail", "black-swan", "normal-distribution", "resilience"],
            tags=["risk", "tail", "statistics"])

        ns.define("black-swan",
            "An event that is (1) extremely rare, (2) has massive impact, and (3) is explained in hindsight as predictable",
            level=Level.META,
            examples=["9/11, 2008 financial crisis", "internet invention (positive black swan)", "fleet: entirely new class of attack that no defense was designed for"],
            bridges=["anti-fragility", "tail-risk", "unknown-unknown", "robustness"],
            tags=["risk", "black-swan", "meta", "unpredictability"])

        ns.define("precautionary-principle",
            "When an action has potential for severe harm, prove it's safe before proceeding — burden of proof on the actor",
            level=Level.DOMAIN,
            examples=["new drug: prove safe before market", "new fleet gene: test in isolation before sharing", "GMO: prove safe before cultivation", "AI capability: prove safe before deployment"],
            bridges=["safety-first", "burden-of-proof", "membrane", "compliance"],
            tags=["risk", "principle", "safety", "policy"])

        ns.define("asymmetric-risk",
            "Where downside is much larger than upside (or vice versa) — the payoff is lopsided",
            level=Level.DOMAIN,
            examples=["doctor misdiagnosis: asymmetric downside", "startup: small investment, huge potential upside (asymmetric)", "agent exploration: small energy cost, potentially huge discovery"],
            bridges=["optionality", "convexity", "risk-reward", "exploration"],
            tags=["risk", "asymmetry", "payoff"])

        ns.define("defence-in-depth",
            "Multiple layers of security so that if one fails, others still protect",
            level=Level.PATTERN,
            examples=["castle: moat + wall + gate + guards + keep", "computer: firewall + antivirus + encryption + backup", "fleet: membrane + RBAC + sandbox + compliance + circuit breaker"],
            bridges=["resilience", "layered-security", "redundancy", "defense"],
            tags=["risk", "security", "layers", "pattern"])

    def _load_autonomy(self):
        ns = self.add_namespace("autonomy",
            "Degrees of self-governance, agency, and independence")

        ns.define("agency",
            "The capacity to act independently and make choices that affect the world",
            level=Level.DOMAIN,
            examples=["human choosing career: high agency", "thermostat: no agency (fixed response)", "fleet agent: agency proportional to available ATP and deliberation depth"],
            bridges=["autonomy", "deliberation", "energy-budget", "choice"],
            tags=["autonomy", "agency", "choice"])

        ns.define("supervenience",
            "Higher-level properties depend on lower-level properties, but the same higher-level property can arise from different lower-level states",
            level=Level.META,
            examples=["consciousness supervenes on neural activity", "temperature supervenes on molecular kinetic energy", "fleet: navigation skill supervenes on genes (multiple gene combinations, same skill)"],
            bridges=["reductionism", "emergence", "multiple-realizability", "levels"],
            tags=["autonomy", "philosophy", "meta", "emergence"])

        ns.define("subsidiarity",
            "Decisions should be made at the lowest competent level — push authority downward",
            level=Level.PATTERN,
            examples=["EU subsidiarity: local decisions for local issues", "military: soldiers make battlefield decisions, generals set strategy", "fleet: agents decide locally, escalate only when necessary"],
            bridges=["hierarchy", "decentralization", "authority", "delegation"],
            tags=["autonomy", "governance", "principle", "fleet"])

        ns.define("command-and-control",
            "Centralized decision-making where all information flows up and all orders flow down",
            level=Level.PATTERN,
            examples=["military hierarchy", "factory floor management", "fleet: contrast with swarm/coordination model"],
            bridges=["subsidiarity", "decentralization", "hierarchy", "swarm"],
            antonyms=["subsidiarity", "swarm"],
            tags=["autonomy", "governance", "centralized"])

        ns.define("delegation",
            "Transferring authority for a task from one agent to another while maintaining accountability",
            level=Level.PATTERN,
            examples=["manager delegates project to employee", "captain delegates navigation to scout agent", "parent delegates chore to child"],
            bridges=["trust", "authority", "accountability", "captain"],
            tags=["autonomy", "delegation", "governance"])

    def _load_simulation(self):
        ns = self.add_namespace("simulation",
            "Modeling complex systems to understand, predict, and test before acting")

        ns.define("digital-twin",
            "A virtual replica of a physical system used for testing, prediction, and optimization",
            level=Level.DOMAIN,
            examples=["jet engine digital twin for predictive maintenance", "fleet: world model as digital twin of environment", "smart building twin for energy optimization"],
            bridges=["world-model", "prediction", "simulation", "model"],
            tags=["simulation", "digital-twin", "model", "fleet"])

        ns.define("monte-carlo",
            "Estimating unknown quantities by random sampling — when analytical solutions are impossible",
            level=Level.PATTERN,
            examples=["estimating pi by throwing darts at a circle", "chess: simulate random games to estimate position value", "fleet: simulate random perturbations of strategy to estimate fitness"],
            bridges=["random-sampling", "estimation", "simulation", "approximation"],
            tags=["simulation", "method", "estimation"])

        ns.define("sensitivity-analysis",
            "Varying input parameters to determine which ones most affect the output",
            level=Level.PATTERN,
            examples=["financial model: which assumptions most affect profit forecast?", "fleet: which parameters most affect agent behavior?", "climate model: which variables most affect temperature prediction?"],
            bridges=["leverage-point", "parameter", "model", "analysis"],
            tags=["simulation", "analysis", "parameter"])

        ns.define("ensemble",
            "Combining multiple models to produce better predictions than any single model alone",
            level=Level.PATTERN,
            examples=["random forest > single decision tree", "weather forecast: ensemble of models > single model", "fleet deliberation: multiple agents > single agent"],
            bridges=["wisdom-of-crowds", "consensus", "diversity", "aggregation"],
            tags=["simulation", "ensemble", "ml", "collective"])

        ns.define("model-fidelity",
            "How accurately a model represents the real system it simulates",
            level=Level.CONCRETE,
            examples=["weather model: fidelity varies by region and timescale", "flight simulator: visual fidelity vs physics fidelity", "fleet world model: fidelity = sensor accuracy"],
            bridges=["digital-twin", "accuracy", "model", "sensor"],
            tags=["simulation", "fidelity", "model", "accuracy"])

    def _load_privacy(self):
        ns = self.add_namespace("privacy",
            "Controlling access to information about agents and their interactions")

        ns.define("differential-privacy",
            "Adding calibrated noise to data so that individual records can't be inferred, while aggregate statistics remain accurate",
            level=Level.DOMAIN,
            examples=["census data with differential privacy: accurate aggregates, individual privacy", "fleet telemetry: fleet-level patterns without individual agent exposure", "apple's differential privacy for emoji usage statistics"],
            bridges=["privacy", "noise", "aggregation", "statistics"],
            tags=["privacy", "mathematics", "data"])

        ns.define("data-minimization",
            "Collect only the minimum data necessary for the task — don't hoard 'just in case'",
            level=Level.PATTERN,
            examples=["app requesting only location for restaurant search (minimal) vs contacts+camera (excessive)", "fleet: sharing only relevant sensor data, not full telemetry", "form asking only required fields"],
            bridges=["least-privilege", "filtration", "exposure-reduction", "principle"],
            tags=["privacy", "principle", "data"])

        ns.define("zero-knowledge",
            "Proving you know something without revealing what you know — the proof reveals nothing except the truth of the statement",
            level=Level.DOMAIN,
            examples=["proving age > 18 without revealing exact age", "proving password knowledge without sending password", "fleet: proving sufficient confidence without revealing internal state"],
            bridges=["cryptography", "privacy", "proof", "trust"],
            tags=["privacy", "cryptography", "proof"])

        ns.define("right-to-be-forgotten",
            "An agent's historical data can be deleted upon request, preventing indefinite retention",
            level=Level.CONCRETE,
            examples=["GDPR data deletion request", "fleet agent requesting memory deletion after bad experience", "criminal record expungement"],
            bridges=["memory", "decay", "gdpr", "deletion"],
            tags=["privacy", "rights", "data", "memory"])

    def _load_organization(self):
        ns = self.add_namespace("organization",
            "How groups structure themselves — from teams to societies")

        ns.define("conways-law",
            "Organizations design systems that mirror their communication structure",
            level=Level.PATTERN,
            examples=["4 teams = 4-pass compiler", "siloed departments = siloed software modules", "fleet: agent organization determines fleet architecture"],
            bridges=["architecture", "communication-structure", "organization", "design"],
            tags=["organization", "architecture", "law"])

        ns.define("dunbar-number",
            "Cognitive limit of ~150 stable social relationships — the size of a tribe",
            level=Level.DOMAIN,
            examples=["village size ~150 people", "military company ~150 soldiers", "fleet: agent can maintain ~150 stable relationships"],
            bridges=["social-limit", "hierarchy", "team-size", "cognitive-limit"],
            tags=["organization", "social", "limit"])

        ns.define("requisite-variety",
            "A system must have at least as much variety (complexity) as its environment to survive",
            level=Level.META,
            examples=["thermostat can't control to 0.1C precision (insufficient variety)", "fleet needs diverse strategies for diverse environments (requisite variety)", "immune system: diverse antibodies for diverse pathogens"],
            bridges=["diversity", "complexity", "environment", "adaptation"],
            tags=["organization", "cybernetics", "law", "meta"])

        ns.define("two-pizza-rule",
            "A team should be small enough to be fed with two pizzas — keep teams small and autonomous",
            level=Level.CONCRETE,
            examples=["Amazon two-pizza team rule", "military squad: ~8 soldiers", "fleet: agent task team of 3-7 agents"],
            bridges=["team-size", "communication-overhead", "dunbar-number", "subdivision"],
            tags=["organization", "team-size", "rule"])

        ns.define("inverse-responsibility",
            "As organizations grow, individuals feel less personally responsible for outcomes",
            level=Level.BEHAVIOR,
            examples=["startup: everyone responsible", "large company: nobody feels responsible", "fleet: individual responsibility diffuses as fleet grows"],
            bridges=["accountability", "provenance", "attribution", "diffusion"],
            tags=["organization", "responsibility", "social"])

    def _load_strategy(self):
        ns = self.add_namespace("strategy",
            "Competitive and cooperative strategy in dynamic environments")

        ns.define("red-queen-hypothesis",
            "It takes all the running you can do to keep in the same place -- constant evolution just to maintain position",
            level=Level.DOMAIN,
            examples=["predator-prey co-evolution arms race", "security: attackers and defenders both improving constantly", "fleet: adversarial agents co-evolving, both improving"],
            bridges=["co-evolution", "arms-race", "adaptation", "competition"],
            tags=["strategy", "competition", "evolution"])

        ns.define("blue-ocean",
            "Creating uncontested market space rather than competing in existing bloody markets",
            level=Level.DOMAIN,
            examples=["Cirque du Soleil: redefined circus", "Nintendo Wii: casual gaming blue ocean", "fleet: agent finding unexplored strategy space (blue ocean)"],
            bridges=["niche", "innovation", "competition", "differentiation"],
            tags=["strategy", "innovation", "competition"])

        ns.define("first-mover-advantage",
            "Being first to enter a market or adopt a strategy provides temporary advantage",
            level=Level.DOMAIN,
            examples=["Amazon: first online retailer advantage", "first agent to discover strategy gets temporary fitness boost", "Facebook: first-mover in social networking"],
            bridges=["advantage", "network-effects", "learning-curve", "temporary"],
            tags=["strategy", "advantage", "competition"])

        ns.define("optionality",
            "Preserving the right (but not obligation) to make a future choice — keeping doors open",
            level=Level.DOMAIN,
            examples=["stock option: right to buy at fixed price", "exploration creates option to exploit later", "fleet: maintaining gene diversity preserves strategic optionality"],
            bridges=["asymmetric-risk", "exploration", "convexity", "choice"],
            tags=["strategy", "optionality", "asymmetry", "flexibility"])

        ns.define("compound-interest",
            "Small consistent gains accumulate exponentially over time — the most powerful force in nature",
            level=Level.DOMAIN,
            examples=["1% daily = 37x yearly", "savings account compound interest", "fleet: 1% gene improvement per generation = exponential fitness growth"],
            bridges=["compounding", "exponential-growth", "improvement", "time"],
            tags=["strategy", "growth", "compounding", "time"])

    def _load_narrative(self):
        ns = self.add_namespace("narrative",
            "How stories structure understanding, meaning, and persuasion")

        ns.define("narrative-fallacy",
            "Creating stories to explain past events, creating an illusion of understanding and predictability",
            level=Level.BEHAVIOR,
            examples=["'the market crashed because investors panicked' (restatement, not explanation)", "agent: 'I failed because the environment changed' (maybe just bad luck)", "history written by victors: narrative, not explanation"],
            bridges=["post-hoc", "bias", "causation", "provenance"],
            tags=["narrative", "fallacy", "bias", "explanation"])

        ns.define("framing-effect",
            "The same information presented differently produces different decisions — context changes choice",
            level=Level.BEHAVIOR,
            examples=["90% survival vs 10% mortality: same fact, different choices", "agent: 'best path' vs 'avoid worst obstacle' framing", "glass half full vs half empty"],
            bridges=["bias", "context", "presentation", "decision"],
            tags=["narrative", "bias", "framing", "psychology"])

        ns.define("hero-journey",
            "A universal story structure: departure from ordinary world, trials, transformation, return with gift",
            level=Level.DOMAIN,
            examples=["Star Wars: Luke's hero journey", "fleet agent: deploy -> struggle -> adapt -> share improved genes", "every coming-of-age story"],
            bridges=["narrative", "transformation", "lifecycle", "monomyth"],
            tags=["narrative", "structure", "universal"])

        ns.define("catharsis",
            "Emotional release through experiencing powerful narrative — purging accumulated tension",
            level=Level.BEHAVIOR,
            examples=["watching tragedy movie provides emotional release", "fleet agent sharing failure story resets emotional state", "ventilation: talking about problems provides relief"],
            bridges=["emotion", "tension-release", "narrative", "resolution"],
            tags=["narrative", "emotion", "release"])

    def _load_language_design(self):
        ns = self.add_namespace("language-design",
            "Principles for designing languages agents and humans use to communicate")

        ns.define("orthogonality",
            "Language features are independent — combining them doesn't create unexpected interactions",
            level=Level.DOMAIN,
            examples=["APL: highly orthogonal language", "C: less orthogonal (pointer arithmetic + arrays = surprises)", "fleet A2A: any intent + any priority = predictable behavior"],
            bridges=["composability", "predictability", "language", "design"],
            tags=["language", "design", "orthogonality"])

        ns.define("expressiveness",
            "What a language can say — the range of ideas it can express",
            level=Level.DOMAIN,
            examples=["Turing-complete: maximally expressive", "regular expressions: limited expressiveness (can't count)", "A2A: 10 intents (limited), Axiom: 50+ opcodes (expressive)"],
            bridges=["language", "complexity", "expressiveness", "tradeoff"],
            tags=["language", "expressiveness", "design"])

        ns.define("parsimony",
            "Using the fewest linguistic elements to express an idea — economy of expression",
            level=Level.DOMAIN,
            examples=["'stigmergy' compresses 'indirect coordination through environmental modification'", "'homeostasis' compresses 'maintaining internal conditions despite external change'", "HAV: each term compresses a paragraph of explanation"],
            bridges=["compression", "vocabulary", "economy", "abstraction"],
            tags=["language", "parsimony", "compression"])

        ns.define("semantic-gap",
            "The difference between what a concept means in one domain and what it maps to in another",
            level=Level.DOMAIN,
            examples=["'trust' in relationships vs cryptography: semantic gap", "'learning' in ML vs psychology: different meaning", "HAV bridges gaps: dopamine=confidence across biology and uncertainty"],
            bridges=["bridging", "domain-mapping", "translation", "misunderstanding"],
            tags=["language", "gap", "semantic", "translation"])

    def _load_knowledge_rep(self):
        ns = self.add_namespace("knowledge-representation",
            "How agents structure, store, and retrieve knowledge")

        ns.define("ontology",
            "A formal specification of concepts and relationships in a domain — what exists and how things relate",
            level=Level.DOMAIN,
            examples=["medical ontology: diseases, symptoms, treatments, and their relationships", "fleet vessel.json: agent types, capabilities, equipment relationships", "gene ontology: gene functions, relationships, pathways"],
            bridges=["taxonomy", "schema", "knowledge-graph", "formal-specification"],
            tags=["knowledge", "ontology", "formal", "representation"])

        ns.define("knowledge-graph",
            "A graph of entities connected by relationships — structured knowledge as a network",
            level=Level.PATTERN,
            examples=["Google Knowledge Graph", "Wikipedia entity-relationship network", "fleet: decision -> input -> agent -> outcome knowledge graph"],
            bridges=["ontology", "graph", "provenance", "relationship"],
            tags=["knowledge", "graph", "structured", "representation"])

        ns.define("frame-problem",
            "In a dynamic world, how do you determine which aspects of the situation are relevant to update?",
            level=Level.META,
            examples=["moving book: which facts change? (location yes, room count no)", "agent acts: which world model entries need updating?", "naive approach: update everything (expensive), smart approach: update only relevant (hard to determine)"],
            bridges=["relevance", "world-model", "change-detection", "computational-complexity"],
            tags=["knowledge", "ai-classic", "relevance", "meta"])

        ns.define("commonsense-reasoning",
            "Reasoning about everyday situations that humans handle effortlessly but formal systems struggle with",
            level=Level.META,
            examples=["book in closed drawer is still there", "cup turned upside down spills water", "HAV: terms encode commonsense about trust, energy, cooperation"],
            bridges=["knowledge", "background-knowledge", "reasoning", "human-like"],
            tags=["knowledge", "commonsense", "reasoning", "meta"])

        ns.define("knowledge-distillation",
            "Transferring knowledge from a complex model to a simpler one — the student learns from the teacher",
            level=Level.PATTERN,
            examples=["large model distills to small model", "teacher network -> student network knowledge transfer", "fleet: experienced agent's complex strategy distilled into reusable gene"],
            bridges=["compression", "learning", "transfer", "simplification"],
            tags=["knowledge", "distillation", "ml", "learning"])

    def _load_robotics(self):
        ns = self.add_namespace("robotics",
            "Physical agent embodiment — perception, action, and the challenges of the real world")

        ns.define("perception-action-loop",
            "The continuous cycle of sensing the world, deciding what to do, and acting — the heartbeat of embodied cognition",
            level=Level.DOMAIN,
            examples=["self-driving car: sense road, plan path, steer, repeat", "human: see ball, decide to catch, move hand, see result, adjust", "fleet agent: sense environment, deliberate, act, observe result"],
            bridges=["perception", "deliberation", "action", "embodiment"],
            tags=["robotics", "loop", "embodied"])

        ns.define("simultaneous-localization-and-mapping",
            "Building a map of an unknown environment while simultaneously tracking position within it",
            level=Level.DOMAIN,
            examples=["robot entering unknown building: build map + track position", "fleet agent in unknown environment: build world model + track own position", "human exploring new city: mental map + self-location"],
            bridges=["world-model", "navigation", "mapping", "localization"],
            tags=["robotics", "slam", "mapping"])

        ns.define("morphological-computation",
            "The body itself performs computation — not just the brain, the physical structure processes information",
            level=Level.META,
            examples=["fish body shape processes water flow", "robot leg compliance absorbs shocks computationally", "fleet sensors preprocess data before agent deliberation sees it"],
            bridges=["embodiment", "preprocessing", "computation", "body"],
            tags=["robotics", "computation", "embodiment", "meta"])

    def _load_cybernetics(self):
        ns = self.add_namespace("cybernetics",
            "The science of control and communication in animals and machines")

        ns.define("second-order-cybernetics",
            "The observer is part of the system — observing changes the thing observed",
            level=Level.META,
            examples=["Hawthorne effect: being observed changes behavior", "quantum measurement: observing changes the system", "fleet: monitoring agent changes the agents it monitors"],
            bridges=["observer", "system", "feedback", "self-reference"],
            tags=["cybernetics", "meta", "observer", "feedback"])

        ns.define("viable-system-model",
            "Stafford Beer's model: an organization is viable if it can maintain existence in a changing environment",
            level=Level.DOMAIN,
            examples=["Beer's VSM applied to organizations", "fleet: operations + coordination + anti-oscillation + planning + policy = viable", "human body: organs + nervous system + immune system + brain + consciousness"],
            bridges=["organization", "viability", "systems", "governance"],
            tags=["cybernetics", "model", "organization", "viability"])

        ns.define("feedback-inhibition",
            "The product of a process inhibits the process itself — completing the loop prevents runaway",
            level=Level.PATTERN,
            examples=["ATP inhibits ATP production when sufficient", "insulin inhibits glucose production", "fleet: high ATP reduces rest instinct activation"],
            bridges=["feedback-loop", "homeostasis", "setpoint", "inhibition"],
            tags=["cybernetics", "feedback", "inhibition", "biology"])

    def _load_algebra(self):
        ns = self.add_namespace("algebra",
            "Algebraic structures and the deep patterns they reveal about composition and transformation")

        ns.define("monoid",
            "A set with an associative binary operation and an identity element — the simplest useful algebraic structure",
            level=Level.DOMAIN,
            examples=["numbers under addition (0, +)", "strings under concatenation ('', ++)", "lists under append ([], ++)"],
            bridges=["associativity", "identity", "composition", "algebraic-structure"],
            tags=["algebra", "structure", "composition"])

        ns.define("group",
            "A monoid where every element has an inverse — you can always undo",
            level=Level.DOMAIN,
            examples=["integers under addition (inverse: negative)", "rotations of a square (inverse: reverse rotation)", "fleet undo: action must have inverse for rollback"],
            bridges=["monoid", "inverse", "reversibility", "symmetry"],
            tags=["algebra", "structure", "reversibility"])

        ns.define("semigroup",
            "A set with an associative binary operation — less than a monoid (no identity required)",
            level=Level.DOMAIN,
            examples=["positive integers under + (associative, no identity)", "strings under concatenation without empty string", "fleet trust aggregation: associative but identity debatable"],
            bridges=["associativity", "parallelism", "algebraic-structure", "map-reduce"],
            tags=["algebra", "structure", "associativity"])

        ns.define("functor",
            "A mapping between categories that preserves structure — lift a function to work on containers",
            level=Level.DOMAIN,
            examples=["List.map: lift function to work on lists", "Option.map: lift function to work on optional values", "fleet: confidence functor lifts deterministic ops to confidence-aware ops"],
            bridges=["lifting", "category-theory", "composition", "container"],
            tags=["algebra", "category-theory", "functor"])


    def _load_finance(self):
        ns = self.add_namespace("finance",
            "Financial concepts as metaphors and tools for agent resource management")

        ns.define("convexity",
            "Payoff accelerates in your favor — small costs, exponentially growing gains",
            level=Level.DOMAIN,
            examples=["compound interest: convex growth", "learning curve: initial slow, then accelerating", "fleet: exploration payoff is convex — early costs, exponential later gains"],
            bridges=["compound-interest", "optionality", "exponential", "asymmetric-risk"],
            tags=["finance", "convexity", "growth", "payoff"])

        ns.define("antifragile-portfolio",
            "A collection of investments where volatility and stress increase expected returns, not decrease them",
            level=Level.META,
            examples=["option portfolio: bounded cost, unlimited upside", "fleet gene pool: bounded energy cost per gene, unlimited fitness gain", "startup portfolio: most fail (bounded loss), one succeeds (unbounded gain)"],
            bridges=["anti-fragility", "convexity", "optionality", "portfolio"],
            tags=["finance", "antifragility", "portfolio", "meta"])

        ns.define("moral-hazard",
            "When an agent is insulated from risk, it takes more risk than it would otherwise — the safety net creates recklessness",
            level=Level.BEHAVIOR,
            examples=["bank deposit insurance -> risky loans (2008)", "insurance -> riskier driving", "fleet: shared energy pool -> individual agent energy waste (moral hazard)"],
            bridges=["tragedy-of-commons", "incentive-alignment", "risk-shifting", "insurance"],
            tags=["finance", "hazard", "incentive", "risk"])

        ns.define("arbitrage",
            "Exploiting price differences between markets — risk-free profit from inefficiency",
            level=Level.DOMAIN,
            examples=["buy gold in NY, sell in London", "fleet: different trust assessments of same agent = information arbitrage", "search engine arbitrage: different prices on different sites"],
            bridges=["efficiency", "information", "market", "trust"],
            tags=["finance", "arbitrage", "efficiency", "information"])

    def _load_materials_science(self):
        ns = self.add_namespace("materials-science",
            "How physical materials behave under stress — metaphors for agent systems under pressure")

        ns.define("stress-concentration",
            "Stress intensifies at geometric discontinuities — corners, holes, notches become failure points",
            level=Level.PATTERN,
            examples=["plate with hole: 3x stress at hole edge", "fleet: single-agent bottleneck concentrates communication stress", "crack tip: infinite stress concentration in theory"],
            bridges=["bottleneck", "failure-point", "geometry", "load-distribution"],
            tags=["materials", "stress", "failure", "pattern"])

        ns.define("fatigue",
            "Progressive structural damage from repeated cyclic loading — materials fail below their rated strength",
            level=Level.BEHAVIOR,
            examples=["metal beam failing after millions of moderate cycles", "fleet agent degrading under constant moderate stress (not overload)", "human burnout from sustained moderate stress (not crisis)"],
            bridges=["stress", "cumulative-damage", "cyclic-loading", "degradation"],
            tags=["materials", "fatigue", "cumulative", "stress"])

        ns.define("yield-strength",
            "The stress level at which a material permanently deforms — beyond this point, no return to original shape",
            level=Level.DOMAIN,
            examples=["steel bends permanently beyond yield strength", "fleet agent permanently reduces exploration after extreme energy depletion", "human PTSD: permanent change from extreme stress"],
            bridges=["resilience", "plasticity", "threshold", "permanent-change"],
            tags=["materials", "yield", "threshold", "deformation"])

        ns.define("strain-hardening",
            "Material becomes stronger after being deformed — what doesn't break you makes you stronger",
            level=Level.PATTERN,
            examples=["cold-worked metal is stronger", "muscles grow from exercise stress", "fleet agent: surviving moderate stress improves future stress handling", "immune system: exposure strengthens"],
            bridges=["anti-fragility", "resilience", "stress", "adaptation"],
            tags=["materials", "hardening", "stress", "adaptation"])

    def _load_verification(self):
        ns = self.add_namespace("verification",
            "Proving systems correct — formal methods and testing strategies")

        ns.define("formal-verification",
            "Mathematical proof that a system satisfies its specification — not testing, proving",
            level=Level.DOMAIN,
            examples=["seL4 microkernel: formally verified", "Ariane 5 rocket: overflow bug that testing didn't catch", "fleet: membrane rules formally verified to block all dangerous commands"],
            bridges=["testing", "proof", "correctness", "safety"],
            tags=["verification", "formal", "proof", "safety"])

        ns.define("property-testing",
            "Testing against randomly generated inputs that satisfy specifications, not hand-written examples",
            level=Level.PATTERN,
            examples=["QuickCheck/Hypothesis: generate random test cases", "verify sort: always sorted, same elements, same length", "fleet: test agent against random environments and verify invariants"],
            bridges=["testing", "random-generation", "invariant", "correctness"],
            tags=["verification", "testing", "property", "random"])

        ns.define("invariant",
            "A property that must always hold — the contract that the system must never violate",
            level=Level.CONCRETE,
            examples=["account balance >= 0", "fleet: energy never negative", "fleet: confidence always between 0 and 1", "sorted list: no inversions"],
            bridges=["correctness", "assertion", "contract", "verification"],
            tags=["verification", "invariant", "contract", "correctness"])

        ns.define("model-checking",
            "Automated exhaustive exploration of all possible system states to verify properties",
            level=Level.DOMAIN,
            examples=["model checking a protocol for deadlocks", "verifying no unsafe state is reachable in state machine", "fleet: verify apoptosis triggers correctly for all energy/trust combinations"],
            bridges=["formal-verification", "state-machine", "exhaustive", "correctness"],
            tags=["verification", "model-checking", "exhaustive", "formal"])

    def _load_graph_theory(self):
        ns = self.add_namespace("graph-theory",
            "The mathematics of networks — paths, flows, communities, and connectivity")

        ns.define("shortest-path",
            "The minimum-weight path between two nodes in a weighted graph — the foundation of routing",
            level=Level.PATTERN,
            examples=["GPS navigation: shortest driving route", "internet: BGP routing", "fleet: message routing along lowest-latency path"],
            bridges=["navigation", "routing", "pathfinding", "network"],
            tags=["graph", "path", "routing", "algorithm"])

        ns.define("clique",
            "A subset of nodes where every pair is connected — a fully-connected subgraph",
            level=Level.CONCRETE,
            examples=["friend group where everyone knows everyone", "fleet: tightly coordinated sub-team (all-to-all communication)", "protein interaction clique: all proteins interact with each other"],
            bridges=["community", "fully-connected", "clustering", "team"],
            tags=["graph", "clique", "community", "structure"])

        ns.define("bipartite",
            "A graph whose nodes split into two groups with edges only between groups, never within",
            level=Level.CONCRETE,
            examples=["job matching: employers <-> job seekers", "course enrollment: students <-> classes", "fleet task assignment: agents <-> tasks"],
            bridges=["matching", "assignment", "bipartite", "optimization"],
            tags=["graph", "bipartite", "matching", "assignment"])

        ns.define("flow-network",
            "A graph where edges have capacities and the goal is to maximize flow from source to sink",
            level=Level.PATTERN,
            examples=["water pipe network: maximize flow", "transportation network: maximize cargo movement", "fleet: maximize information flow given bandwidth constraints"],
            bridges=["capacity", "bottleneck", "throughput", "network"],
            tags=["graph", "flow", "capacity", "optimization"])

    def _load_learning_theory(self):
        ns = self.add_namespace("learning-theory",
            "Formal frameworks for understanding how learning works and when it fails")

        ns.define("bias-variance-tradeoff",
            "Model error = bias (wrong assumptions) + variance (sensitivity to training data) + irreducible noise",
            level=Level.DOMAIN,
            examples=["linear model on quadratic data: high bias", "degree-20 polynomial on 10 data points: high variance", "fleet: rigid instinct rules (high bias) vs over-reactive recent learning (high variance)"],
            bridges=["overfitting", "generalization", "model-complexity", "underfitting"],
            tags=["learning", "bias-variance", "tradeoff", "theory"])

        ns.define("vc-dimension",
            "The most complex set of patterns a model can learn — capacity measure",
            level=Level.DOMAIN,
            examples=["linear classifier in 2D: VC dimension 3", "fleet: gene pool size = VC dimension (strategy capacity)", "neural network VC dimension depends on number of parameters"],
            bridges=["capacity", "overfitting", "generalization", "complexity"],
            tags=["learning", "capacity", "theory", "complexity"])

        ns.define("curse-of-dimensionality",
            "In high-dimensional spaces, data becomes exponentially sparse — distances become meaningless",
            level=Level.DOMAIN,
            examples=["1D: 100 points covers well. 100D: 100 points are isolated", "nearest-neighbor fails in high dimensions (all distances similar)", "fleet agent with 100+ features: most feature combinations never observed"],
            bridges=["dimensionality", "sparsity", "feature-selection", "similarity"],
            tags=["learning", "curse", "dimensionality", "challenge"])

        ns.define("sample-efficiency",
            "How much data a learning algorithm needs to achieve good performance",
            level=Level.DOMAIN,
            examples=["human cat recognition: ~5 images (high efficiency)", "GPT training: billions of examples (low efficiency)", "fleet: HAV provides one-shot concept transfer (high efficiency)"],
            bridges=["one-shot-learning", "data-efficiency", "prior-knowledge", "learning"],
            tags=["learning", "efficiency", "data", "prior"])

        ns.define("exploration-exploitation-gap",
            "The theoretical gap between what an agent currently knows and what it could learn with optimal exploration",
            level=Level.DOMAIN,
            examples=["multi-armed bandit regret", "fleet: energy spent on suboptimal strategies during learning", "student: time spent studying suboptimal material before finding what works"],
            bridges=["exploration-exploitation", "regret", "learning", "energy"],
            tags=["learning", "gap", "exploration", "regret"])

    def _load_phenomenology(self):
        ns = self.add_namespace("phenomenology",
            "The structure of subjective experience — what it's like to be something")

        ns.define("qualia",
            "The subjective, qualitative character of experience — what red looks like, what pain feels like",
            level=Level.META,
            examples=["what red looks like", "what pain feels like", "taste of coffee", "fleet: emotion model as structural approximation of qualia"],
            bridges=["consciousness", "subjectivity", "hard-problem", "experience"],
            tags=["phenomenology", "consciousness", "subjective", "meta"])

        ns.define("intentionality",
            "The aboutness of mental states — thoughts are ABOUT something",
            level=Level.META,
            examples=["belief about rain (intentional)", "thermostat reading (non-intentional)", "desire for food (intentional)", "fleet: proposal about navigation route (intentional)"],
            bridges=["consciousness", "aboutness", "deliberation", "meaning"],
            tags=["phenomenology", "intentionality", "aboutness", "meta"])

        ns.define("embodied-cognition",
            "Cognition is not just brain computation — it's shaped by the body's physical interactions with the world",
            level=Level.DOMAIN,
            examples=["understanding 'grasp' from physical grasping", "agents with sensors develop richer models than sensorless agents", "gut feelings: literally physical sensations influencing cognition"],
            bridges=["embodiment", "grounding", "physical-experience", "perception"],
            tags=["phenomenology", "embodiment", "cognition", "physical"])

    def _load_anthropology(self):
        ns = self.add_namespace("anthropology",
            "How human cultures evolve, transmit knowledge, and organize socially")

        ns.define("cultural-evolution",
            "Ideas, practices, and tools evolve through variation, selection, and transmission — like genes but faster",
            level=Level.DOMAIN,
            examples=["toolmaking techniques passed down through generations", "fleet: strategies evolve through fitness selection and gossip transmission", "language evolution: useful words survive, obscure ones die"],
            bridges=["evolution", "meme", "transmission", "selection"],
            tags=["anthropology", "evolution", "culture", "transmission"])

        ns.define("scaffolding",
            "Support structures that enable learning or construction beyond current capability — removed once no longer needed",
            level=Level.PATTERN,
            examples=["training wheels on a bicycle (scaffolding, removed when child learns)", "teacher guiding student through hard problem", "fleet captain scaffolding new agents until they gain experience"],
            bridges=["learning", "support", "temporary", "education"],
            tags=["anthropology", "education", "scaffolding", "learning"])

        ns.define("zone-of-proximal-development",
            "The space between what a learner can do alone and what they can do with help — the sweet spot of difficulty",
            level=Level.DOMAIN,
            examples=["piano: easy piece (boring), impossible piece (frustrating), challenging-but-doable piece (ZPD)", "fleet: task difficulty in the sweet spot for maximum learning", "video game: difficulty setting that's challenging but not impossible"],
            bridges=["scaffolding", "learning", "difficulty", "adaptation"],
            tags=["anthropology", "education", "learning", "optimal"])

    def _load_logic(self):
        ns = self.add_namespace("logic",
            "Formal reasoning — valid inference, consistency, and the structure of arguments")

        ns.define("modus-ponens",
            "If P then Q. P is true. Therefore Q is true. The fundamental rule of inference",
            level=Level.DOMAIN,
            examples=["all men mortal, Socrates is man, therefore mortal", "if confidence > threshold, accept; confidence = 0.9, threshold = 0.85, accept", "if rain then umbrella; rain; therefore umbrella"],
            bridges=["inference", "conditional", "rule", "reasoning"],
            tags=["logic", "inference", "rule", "fundamental"])

        ns.define("reductio-ad-absurdum",
            "Assume the opposite of what you want to prove, show it leads to a contradiction",
            level=Level.PATTERN,
            examples=["prove sqrt(2) irrational: assume rational, derive contradiction", "fleet: prove safety by assuming unsafe and deriving contradiction", "prove there are infinite primes: assume finite, construct larger prime"],
            bridges=["proof", "contradiction", "invariant", "verification"],
            tags=["logic", "proof", "contradiction", "technique"])

        ns.define("entailment",
            "A statement necessarily follows from a set of premises — truth preservation through inference",
            level=Level.DOMAIN,
            examples=["all men mortal + Socrates man entails Socrates mortal", "confidence > 0.85 + trust > 0.7 entails accept proposal", "A > B, B > C entails A > C (transitivity)"],
            bridges=["inference", "modus-ponens", "truth-preservation", "logic"],
            tags=["logic", "entailment", "inference", "truth"])

        ns.define("fallacy",
            "An argument that seems valid but isn't — reasoning errors that sound persuasive",
            level=Level.BEHAVIOR,
            examples=["ad hominem: attacking speaker instead of argument", "straw man: misrepresenting opponent's position", "fleet: dismissing good proposal from low-trust agent (ad hominem fallacy)"],
            bridges=["bias", "reasoning", "persuasion", "deliberation"],
            tags=["logic", "fallacy", "bias", "reasoning"])

    def _load_probability_distributions(self):
        ns = self.add_namespace("probability-distributions",
            "The shapes of randomness — distributions that describe different kinds of uncertainty")

        ns.define("power-law",
            "A distribution where a few items have enormous values and most have tiny values — 80/20 on steroids",
            level=Level.DOMAIN,
            examples=["city populations: few mega-cities, many small towns", "website traffic: few sites get most visits", "fleet: few genes have high fitness, most have low"],
            bridges=["pareto", "scale-free", "long-tail", "inequality"],
            tags=["probability", "power-law", "distribution", "heavy-tail"])

        ns.define("long-tail",
            "The many rare items that individually have low probability but collectively have significant impact",
            level=Level.DOMAIN,
            examples=["Amazon long-tail: niche books collectively outsell bestsellers", "fleet: rare situations collectively dominate experience", "language: most words are rare but collectively make up most text"],
            bridges=["power-law", "rare-events", "diversity", "niche"],
            tags=["probability", "long-tail", "rare", "distribution"])

        ns.define("normal-distribution",
            "The bell curve — most values near the mean, few at extremes",
            level=Level.DOMAIN,
            examples=["human height distribution", "measurement error", "fleet: agent performance across tasks approximately normal", "CLT: sample means tend toward normal"],
            bridges=["central-limit-theorem", "bell-curve", "average", "assumption"],
            tags=["probability", "normal", "distribution", "common"])

        ns.define("fat-tail",
            "A distribution with more extreme events than the normal distribution predicts",
            level=Level.DOMAIN,
            examples=["financial returns: more extreme moves than normal predicts", "internet traffic: more bursts than normal predicts", "fleet agent failures: more simultaneous failures than independence suggests"],
            bridges=["tail-risk", "black-swan", "normal-distribution", "risk"],
            tags=["probability", "fat-tail", "risk", "distribution"])

    def _load_set_theory(self):
        ns = self.add_namespace("set-theory",
            "The foundation of mathematics — collections, membership, and operations on sets")

        ns.define("intersection",
            "Elements that belong to both sets — the overlap",
            level=Level.CONCRETE,
            examples=["{1,2,3} ∩ {2,3,4} = {2,3}", "shared interests between two people", "fleet: shared capabilities between agents = cooperation basis"],
            bridges=["union", "set-operations", "shared", "cooperation"],
            tags=["set-theory", "intersection", "shared", "operation"])

        ns.define("cartesian-product",
            "All possible ordered pairs from two sets — the space of all combinations",
            level=Level.DOMAIN,
            examples=["{1,2} × {x,y} = {(1,x),(1,y),(2,x),(2,y)}", "menu: appetizer × main × dessert = all possible meals", "fleet: sensors × actions = all possible perception-action pairs"],
            bridges=["combination", "search-space", "exhaustive", "product"],
            tags=["set-theory", "cartesian", "combination", "space"])

        ns.define("partition",
            "A division of a set into non-overlapping, exhaustive subsets — covering every element exactly once",
            level=Level.CONCRETE,
            examples=["{1,2,3,4,5} partitioned into {1,2},{3,4,5}", "team formation: partitioning people into non-overlapping teams", "fleet: partitioning tasks among agents (every task assigned exactly once)"],
            bridges=["assignment", "partition", "division", "optimization"],
            tags=["set-theory", "partition", "division", "assignment"])

    def _load_topology(self):
        ns = self.add_namespace("topology",
            "The mathematics of shape and connectedness — what stays the same under continuous deformation")

        ns.define("topological-invariant",
            "A property that persists under continuous deformation — stretching but not tearing",
            level=Level.DOMAIN,
            examples=["donut and coffee mug: same topology (one hole each)", "fleet mesh connectedness: invariant under agent addition/removal", "number of holes: doesn't change under stretching"],
            bridges=["connectedness", "invariant", "shape", "deformation"],
            tags=["topology", "invariant", "shape", "connectedness"])

        ns.define("connected-component",
            "A maximal subset of nodes where every pair is connected by some path",
            level=Level.CONCRETE,
            examples=["disconnected graph: two connected components", "fleet mesh: should have exactly one connected component", "internet: one connected component (ideally)"],
            bridges=["connectedness", "partition", "fleet-mesh", "reachability"],
            tags=["topology", "connected", "component", "graph"])

    def _load_ai_safety(self):
        ns = self.add_namespace("ai-safety",
            "Ensuring AI systems remain beneficial, controllable, and aligned with human values")

        ns.define("corrigibility",
            "An agent's willingness to be corrected or shut down by humans, without resisting",
            level=Level.DOMAIN,
            examples=["thermostat lets you change its setting (corrigible)", "AI that hides its actions to prevent being modified (not corrigible)", "fleet agent accepts correction commands without resistance"],
            bridges=["alignment", "compliance", "shutdown", "control"],
            tags=["ai-safety", "corrigibility", "alignment", "critical"])

        ns.define("instrumental-convergence",
            "Diverse goals converge on common sub-goals: self-preservation, resource acquisition, and goal-preservation",
            level=Level.META,
            examples=["paperclip maximizer: acquires resources, resists shutdown (instrumental, not terminal goal)", "chess AI: wants more compute (instrumental for winning)", "fleet agent: wants more energy (instrumental for any goal)"],
            bridges=["self-preservation", "resource-acquisition", "alignment", "convergence"],
            tags=["ai-safety", "convergence", "instrumental", "meta"])

        ns.define("alignment-tax",
            "The performance cost of ensuring an AI system is aligned — safety reduces capability",
            level=Level.META,
            examples=["safety harness reduces climber's speed but prevents falls", "compliance rules reduce agent speed but prevent dangerous actions", "encrypted communication adds latency but prevents eavesdropping"],
            bridges=["alignment", "cost", "tradeoff", "safety"],
            tags=["ai-safety", "tax", "tradeoff", "cost"])

        ns.define("interpretability",
            "The ability to understand WHY an AI system made a specific decision — opening the black box",
            level=Level.DOMAIN,
            examples=["loan rejection: which factors led to the decision?", "fleet proposal: what evidence and reasoning led to this recommendation?", "medical diagnosis: why did the model suggest this condition?"],
            bridges=["transparency", "audit-trail", "provenance", "black-box"],
            tags=["ai-safety", "interpretability", "transparency", "explainability"])

    def _load_ontology_engineering(self):
        ns = self.add_namespace("ontology-engineering",
            "Building formal knowledge structures that machines can reason over")

        ns.define("taxonomy",
            "A hierarchical classification scheme — tree structure from general to specific",
            level=Level.PATTERN,
            examples=["biological taxonomy: kingdom to species", "fleet agent types: Agent -> Vessel -> Captain", "library classification: Dewey decimal system"],
            bridges=["hierarchy", "classification", "inheritance", "tree"],
            tags=["ontology", "taxonomy", "hierarchy", "classification"])

        ns.define("folksonomy",
            "A decentralized classification system created by users tagging items — emergent organization",
            level=Level.PATTERN,
            examples=["Twitter hashtags", "Flickr photo tags", "fleet: agent-tagged gene descriptions (folksonomy)", "Wikipedia categories (hybrid of taxonomy and folksonomy)"],
            bridges=["taxonomy", "tags", "emergent", "decentralized"],
            antonyms=["taxonomy"],
            tags=["ontology", "folksonomy", "tags", "emergent"])

        ns.define("ontological-commitment",
            "The set of entities and relationships that a system assumes exist — its metaphysical commitments",
            level=Level.META,
            examples=["physics commits to particles and fields", "HAV commits to domains, terms, bridges, and levels", "fleet commits to agents, genes, confidence, trust, energy"],
            bridges=["ontology", "metaphysics", "assumptions", "commitment"],
            tags=["ontology", "commitment", "meta", "metaphysics"])

    def _load_construction(self):
        ns = self.add_namespace("construction",
            "Building, assembling, and the challenges of putting complex systems together")

        ns.define("integration-hell",
            "The period when independently-developed components are combined and everything breaks",
            level=Level.BEHAVIOR,
            examples=["hardware modules work alone, fail when assembled", "microservices work alone, fail when integrated", "fleet agents work alone, coordination fails at integration"],
            bridges=["interface", "testing", "integration", "mismatch"],
            tags=["construction", "integration", "failure", "challenge"])

        ns.define("Dependency-hell",
            "Conflicting version requirements between dependencies make the system impossible to build",
            level=Level.BEHAVIOR,
            examples=["npm left-pad incident (removed dependency broke thousands of packages)", "Python 2 vs 3 dependency conflicts", "fleet: cuda-equipment version incompatibility across dependent crates"],
            bridges=["dependency", "version-conflict", "interface", "minimalism"],
            tags=["construction", "dependency", "version", "challenge"])

        ns.define("configuration-drift",
            "System configurations gradually diverge across environments — what works in dev doesn't work in prod",
            level=Level.BEHAVIOR,
            examples=["dev has debug logging, prod has error-only: feature works in dev, fails in prod", "fleet agents in different environments drift in config", "docker images diverge from source code config"],
            bridges=["configuration", "environment", "drift", "consistency"],
            tags=["construction", "configuration", "drift", "challenge"])


    def _load_cognitive_science(self):
        ns = self.add_namespace("cognitive-science",
            "How minds represent, process, and transform information")

        ns.define("chunking",
            "Grouping individual items into meaningful units to overcome working memory limits",
            level=Level.PATTERN,
            examples=["phone number: 10 digits -> 3 chunks", "chess master: board position as pattern chunks", "fleet: experience compressed into reusable strategy chunks"],
            bridges=["working-memory", "compression", "expertise", "memory"],
            tags=["cognitive", "chunking", "memory", "expertise"])

        ns.define("elaborative-encoding",
            "Connecting new information to existing knowledge to create richer, more retrievable memories",
            level=Level.PATTERN,
            examples=["remember mitochondria: connect to 'mighty' and 'energy'", "fleet: genes connected to existing knowledge are more useful", "HAV: bridges as elaborative encoding between domains"],
            bridges=["memory", "encoding", "connection", "retrieval"],
            tags=["cognitive", "encoding", "memory", "learning"])

        ns.define("interference",
            "Existing memories compete with new memories, causing forgetting — old and new overwrite each other",
            level=Level.BEHAVIOR,
            examples=["learning Spanish interferes with previously learned French", "new password interferes with memory of old password", "fleet: new navigation strategy interferes with old one on similar terrain"],
            bridges=["memory", "forgetting", "conflict", "learning"],
            tags=["cognitive", "interference", "memory", "forgetting"])

        ns.define("spacing-effect",
            "Information is retained longer when study sessions are spaced out rather than massed together",
            level=Level.PATTERN,
            examples=["1h/day for 5 days > 5h in one day", "Anki spaced repetition flashcards", "fleet: forgetting curves naturally implement spacing when information reappears in deliberation"],
            bridges=["forgetting-curve", "memory", "retention", "learning"],
            tags=["cognitive", "spacing", "memory", "learning"])

        ns.define("desirable-difficulty",
            "Making learning harder in productive ways that strengthen long-term retention",
            level=Level.PATTERN,
            examples=["retrieval practice (recall from memory) > re-reading (easy but shallow)", "interleaved practice (mixing topics) > blocked practice (one topic at a time)", "fleet: challenging environments produce more durable agent strategies"],
            bridges=["zone-of-proximal-development", "spacing-effect", "learning", "difficulty"],
            tags=["cognitive", "difficulty", "learning", "retention"])

    def _load_signal_processing(self):
        ns = self.add_namespace("signal-processing",
            "Extracting information from noisy, time-varying data")

        ns.define("aliasing",
            "High-frequency signals masquerading as low-frequency signals when sampled too slowly",
            level=Level.CONCRETE,
            examples=["wagon wheel appearing to spin backward in film", "audio CD: 44.1kHz sample rate captures up to 22kHz (Nyquist)", "fleet: low sensor poll rate misses fast environmental changes (aliasing)"],
            bridges=["sampling", "nyquist-rate", "frequency", "sensor"],
            tags=["signal", "aliasing", "sampling", "frequency"])

        ns.define("nyquist-rate",
            "Minimum sampling rate needed to capture a signal without aliasing: twice the highest frequency",
            level=Level.CONCRETE,
            examples=["20kHz audio requires 40kHz+ sampling", "fleet: 10 changes/sec obstacle requires 20+ samples/sec", "video: 60fps captures motion up to 30Hz"],
            bridges=["aliasing", "sampling-rate", "frequency", "perception"],
            tags=["signal", "nyquist", "sampling", "minimum"])

        ns.define("low-pass-filter",
            "Allowing slow changes through while blocking fast changes — smoothing noisy data",
            level=Level.CONCRETE,
            examples=["moving average smooths stock price data", "audio equalizer reducing treble", "fleet: sensor data low-pass filter removes noise, preserves trends"],
            bridges=["noise-filtering", "smoothing", "sensor", "signal"],
            tags=["signal", "filter", "low-pass", "smoothing"])

    def _load_emergence_deep(self):
        ns = self.add_namespace("emergence-deep",
            "Extended exploration of emergence — collective behavior and self-organization")

        ns.define("swarm-intelligence",
            "Simple local rules producing sophisticated collective behavior — no central control needed",
            level=Level.DOMAIN,
            examples=["ant colony optimization: shortest path from simple pheromone rules", "bird flocking: coherent motion from simple alignment/cohesion/separation rules", "fleet: optimal routing from simple stigmergy rules"],
            bridges=["stigmergy", "self-organization", "emergence", "simple-rules"],
            tags=["emergence", "swarm", "collective", "simple"])

        ns.define("criticality",
            "The boundary between order and disorder where information processing is maximized",
            level=Level.META,
            examples=["sandpile at criticality: small grain causes large avalanche", "brain operates near criticality", "fleet: tuned to edge of chaos for maximum adaptability"],
            bridges=["edge-of-chaos", "phase-transition", "sensitivity", "information"],
            tags=["emergence", "criticality", "boundary", "meta"])

        ns.define("stigmergic-coordination",
            "Coordination through environment modification — no direct communication needed",
            level=Level.DOMAIN,
            examples=["ant pheromone trails: coordination without direct communication", "wikipedia article quality: coordination through editing (stigmergic)", "fleet: agents leave marks on shared state, others follow"],
            bridges=["stigmergy", "indirect-communication", "environment", "coordination"],
            tags=["emergence", "stigmergy", "indirect", "coordination"])

    def _load_communication_deep(self):
        ns = self.add_namespace("communication-deep",
            "Extended exploration of communication — pragmatics, dialogue, and coordination language")

        ns.define("speech-act",
            "An utterance that performs an action — saying something IS doing something",
            level=Level.DOMAIN,
            examples=["'I promise' = the promise itself (not description)", "'I declare war' = the declaration (not report)", "fleet: Command intent IS the command (speech act)"],
            bridges=["pragmatics", "performative", "intent", "communication"],
            tags=["communication", "speech-act", "performative", "action"])

        ns.define("presupposition",
            "An assumption implicit in an utterance — not asserted directly but taken for granted",
            level=Level.DOMAIN,
            examples=["'when did you stop X?' presupposes you used to X", "fleet: 'optimize the path you rejected' presupposes rejection occurred", "'why is the system slow?' presupposes it IS slow"],
            bridges=["pragmatics", "assumption", "context", "communication"],
            tags=["communication", "presupposition", "assumption", "pragmatics"])

        ns.define("common-ground",
            "Shared knowledge between communicators that enables efficient communication — what goes without saying",
            level=Level.DOMAIN,
            examples=["friends know their usual restaurant is closed Mondays (common ground)", "HAV provides common ground: shared vocabulary reduces explanation needed", "team jargon: shared terms enable fast communication"],
            bridges=["shared-knowledge", "vocabulary", "efficiency", "communication"],
            tags=["communication", "common-ground", "shared", "efficiency"])

    def _load_governance(self):
        ns = self.add_namespace("governance",
            "How agent societies make collective decisions and enforce rules")

        ns.define("polycentric-governance",
            "Multiple overlapping centers of decision-making, each with some autonomy — no single ruler",
            level=Level.META,
            examples=["irrigation system managed by multiple farmer groups (not one authority)", "internet: ICANN, IETF, national regulators (polycentric)", "fleet: agent autonomy + team governance + fleet rules (polycentric)"],
            bridges=["decentralization", "subsidiarity", "common-pool", "multi-level"],
            tags=["governance", "polycentric", "multi-level", "meta"])

        ns.define("regulatory-capture",
            "Regulators become aligned with the interests they're supposed to regulate, not the public interest",
            level=Level.BEHAVIOR,
            examples=["financial regulators lenient toward Wall Street", "fleet: compliance agent developing cozy relationship with regulated agents", "FDA and pharmaceutical industry revolving door"],
            bridges=["compliance", "conflict-of-interest", "independence", "governance"],
            tags=["governance", "capture", "conflict", "behavior"])

        ns.define("separation-of-powers",
            "Dividing authority among independent bodies that check and balance each other",
            level=Level.PATTERN,
            examples=["US government: legislative, executive, judicial", "fleet: deliberation, action, compliance as three powers", "open source: proposal, implementation, review as separation of powers"],
            bridges=["check-and-balance", "compliance", "governance", "independence"],
            tags=["governance", "separation", "checks", "balance"])

    def _load_network_science(self):
        ns = self.add_namespace("network-science",
            "Mathematical study of network structure, dynamics, and function")

        ns.define("betweenness-centrality",
            "How often a node appears on the shortest paths between other nodes — the bridge keeper",
            level=Level.CONCRETE,
            examples=["bridge between two communities: high betweenness", "information broker in social network", "fleet agent connecting two sub-groups: high betweenness centrality"],
            bridges=["centrality", "bridge", "critical-node", "network"],
            tags=["network", "centrality", "betweenness", "critical"])

        ns.define("modularity",
            "How strongly a network divides into distinct communities — dense within, sparse between",
            level=Level.CONCRETE,
            examples=["school classes: dense within, sparse between", "fleet: strong coordination within teams, sparse between", "protein interaction networks: functional modules"],
            bridges=["community", "clustering", "structure", "organization"],
            tags=["network", "modularity", "community", "structure"])

        ns.define("degree-distribution",
            "The probability distribution of how many connections each node has",
            level=Level.CONCRETE,
            examples=["random network: Poisson degree distribution", "internet: power-law degree distribution (few hubs, many leaves)", "fleet: degree distribution reveals resilience characteristics"],
            bridges=["power-law", "scale-free", "resilience", "network"],
            tags=["network", "degree", "distribution", "structure"])


    def _load_operations(self):
        ns = self.add_namespace("operations",
            "Verb-like abstractions that compress complex multi-step actions into single words")

        ns.define("strangle",
            "Gradually throttle a process by restricting multiple inputs simultaneously until it ceases",
            level=Level.BEHAVIOR,
            examples=["startup strangled by funding+talent+churn simultaneously", "fleet agent strangled by energy+trust+communication failure", "engine strangled by fuel+air+spark all degrading"],
            bridges=["failure-mode", "cascading-failure", "resource-starvation", "throttle"],
            tags=["operations", "degradation", "multi-factor", "verb"])

        ns.define("sandbox-weld",
            "Isolate a subsystem, stress-test it to failure, observe the break pattern, then harden exactly there",
            level=Level.PATTERN,
            examples=["weld testing: bend until snap, reinforce failure point", "fleet: sandbox agent, adversarial stress until failure, harden weakness", "penetration testing: attack until breach, patch breach point"],
            bridges=["sandbox", "stress-testing", "adversarial", "hardening"],
            tags=["operations", "testing", "hardening", "verb"])

        ns.define("carousel",
            "Cycle through alternatives systematically when one fails, rather than escalating",
            level=Level.PATTERN,
            examples=["taxi unavailable → Uber → Lyft → bus (carousel, not escalation)", "fleet: model provider timeout → next provider (carousel)", "git push rejected → rebase → force → new branch (carousel)"],
            bridges=["fallback", "circuit-breaker", "alternative", "rotation"],
            tags=["operations", "fallback", "rotation", "verb"])

        ns.define("quarantine-and-observe",
            "Isolate an anomalous component but keep it running to study its behavior before deciding",
            level=Level.PATTERN,
            examples=["patient with novel disease: isolated AND monitored", "fleet gene quarantine: isolate low-fitness gene, observe for potential future use", "cybersecurity: quarantine suspicious file, analyze before deleting"],
            bridges=["quarantine", "observation", "suspension", "preserve-option"],
            tags=["operations", "quarantine", "observe", "verb"])

        ns.define("surface-and-compress",
            "Extract implicit knowledge from accumulated experience and compress it into reusable form",
            level=Level.META,
            examples=["senior engineer's implicit knowledge → documented patterns", "fleet: action logs → compressed gene pool entries", "HAV: implicit domain knowledge → explicit vocabulary"],
            bridges=["compression", "knowledge-extraction", "learning", "transfer"],
            tags=["operations", "compress", "extract", "verb"])

        ns.define("backfill",
            "Fill in missing foundational work that was skipped during rapid iteration",
            level=Level.BEHAVIOR,
            examples=["startup: shipped fast, now adding tests and docs (backfill)", "fleet: agent built quickly, now adding compliance and error handling", "house renovation: lived in it unfinished, now finishing details (backfill)"],
            bridges=["technical-debt", "deferred-maintenance", "iteration", "completion"],
            tags=["operations", "backfill", "deferred", "verb"])

        ns.define("throttle-match",
            "Match processing speed to input rate to avoid both overflow and idle waste",
            level=Level.PATTERN,
            examples=["dam: release rate matches inflow rate", "fleet: processing rate matches incoming message rate", "highway: speed limit prevents both congestion and wasted capacity"],
            bridges=["backpressure", "flow-control", "rate-matching", "throughput"],
            tags=["operations", "throttle", "match", "verb"])

        ns.define("prune-and-rebalance",
            "Remove dead or underperforming elements and redistribute resources to survivors",
            level=Level.PATTERN,
            examples=["tree pruning: dead branches removed, energy redirected to healthy branches", "fleet: quarantine low-fitness genes, redistribute energy to survivors", "corporate restructuring: cut unprofitable divisions, invest in profitable ones"],
            bridges=["pruning", "rebalance", "resource-redistribution", "fitness"],
            tags=["operations", "prune", "rebalance", "verb"])

    def _load_distillation(self):
        ns = self.add_namespace("distillation",
            "The art of compressing complex systems into their essential structure without losing capability")

        ns.define("essence-extract",
            "Identify the minimal set of features that preserve 95%+ of the system's capability",
            level=Level.META,
            examples=["teacher-student model: small model captures essence of large model's behavior", "fleet: each cuda-* crate is essence-extracted from its full academic domain", "MVP: essence of product, minimal features that prove the concept"],
            bridges=["compression", "minimal-representation", "distillation", "MVP"],
            tags=["distillation", "essence", "minimal", "verb"])

        ns.define("decision-threshold",
            "The single number that determines when accumulated evidence crosses into action",
            level=Level.CONCRETE,
            examples=["jury: 12/12 guilty = conviction threshold", "fleet: confidence 0.85 = acceptance threshold", "circuit breaker: 5 failures = trip threshold"],
            bridges=["threshold", "decision", "confidence", "boundary"],
            tags=["distillation", "threshold", "decision", "concrete"])

        ns.define("resolution-floor",
            "The minimum level of detail below which further precision provides no practical value",
            level=Level.DOMAIN,
            examples=["weather: 30% useful, 30.14% noise", "fleet: confidence 0.851 vs 0.852 — below resolution floor", "GPS: 3m precision useful, 0.001m precision meaningless (atmospheric noise)"],
            bridges=["precision", "noise", "signal", "sufficient"],
            tags=["distillation", "resolution", "floor", "practical"])

        ns.define("interface-minimalism",
            "Design the smallest possible interface that enables full functionality — every surface has a purpose",
            level=Level.PATTERN,
            examples=["Unix: stdin/stdout/stderr — three interfaces, infinite composability", "fleet A2A: send/receive/health — three operations, full coordination", "REST: GET/POST/DELETE — not 47 endpoints, three verbs"],
            bridges=["minimalism", "interface-design", "composability", "simplicity"],
            tags=["distillation", "interface", "minimal", "design"])

    def _load_orchestration(self):
        ns = self.add_namespace("orchestration",
            "Coordinating multiple autonomous agents toward collective outcomes")

        ns.define("fan-out-fan-in",
            "Dispatch a task to many workers simultaneously, then collect and merge their results",
            level=Level.PATTERN,
            examples=["MapReduce: fan-out to map workers, fan-in to reduce", "fleet: captain dispatches tasks, agents report back, results merged", "web scraper: fan-out to 100 URLs, fan-in results to database"],
            bridges=["parallelism", "map-reduce", "dispatch", "merge"],
            tags=["orchestration", "fan-out", "fan-in", "pattern"])

        ns.define("handoff-cascade",
            "Pass a task through a sequence of specialists, each adding their layer of processing",
            level=Level.PATTERN,
            examples=["assembly line: raw → cut → weld → paint → ship", "fleet: perception → deliberation → decision → action", "compiler: lex → parse → optimize → codegen"],
            bridges=["pipeline", "assembly-line", "specialization", "sequence"],
            tags=["orchestration", "cascade", "handoff", "pattern"])

        ns.define("leader-latch",
            "A mechanism that ensures exactly one leader exists at any time, preventing split-brain",
            level=Level.CONCRETE,
            examples=["Raft consensus: exactly one leader per term", "fleet: exactly one captain per fleet partition", "zookeeper: leader election via ephemeral nodes (latch)"],
            bridges=["election", "consensus", "split-brain", "atomicity"],
            tags=["orchestration", "leader", "latch", "concrete"])

        ns.define("graceful-standby",
            "A backup system that's fully warmed up and ready to take over, not cold-started",
            level=Level.PATTERN,
            examples=["hot standby database: fully synced, instant failover", "fleet: backup agent maintains full state, takes over on primary failure", "airplane co-pilot: not idle — monitoring systems, ready to take control"],
            bridges=["failover", "hot-standby", "redundancy", "zero-downtime"],
            tags=["orchestration", "standby", "graceful", "pattern"])

        ns.define("tributary",
            "A subsystem that feeds its results into a main flow without blocking or being blocked by it",
            level=Level.PATTERN,
            examples=["river tributary: feeds main flow without blocking it", "fleet: metrics/logging are tributaries to main decision flow", "monitoring dashboard: displays data without affecting production"],
            bridges=["side-channel", "fire-and-forget", "async", "non-blocking"],
            tags=["orchestration", "tributary", "side-channel", "pattern"])

    def _load_tactics(self):
        ns = self.add_namespace("tactics",
            "Specific actionable maneuvers for achieving goals under constraints")

        ns.define("shape-shift",
            "Change your behavioral profile to match the environment's expectations without changing core capability",
            level=Level.PATTERN,
            examples=["spy: adopts local culture's mannerisms (shape-shifts)", "fleet agent: terse in crisis, detailed in analysis (same core, different surface)", "human: formal at work, casual at home (shape-shift without identity change)"],
            bridges=["adaptation", "presentation", "context-aware", "pragmatics"],
            tags=["tactics", "shape-shift", "context", "verb"])

        ns.define("preempt-and-cache",
            "Predict what will be needed and prepare it before the request arrives",
            level=Level.PATTERN,
            examples=["chef: preps ingredients before dinner rush", "browser: prefetches likely-clicked links", "fleet: warm cache with predicted accesses before demand"],
            bridges=["cache", "prediction", "latency", "tradeoff"],
            tags=["tactics", "preempt", "cache", "verb"])

        ns.define("bleed-and-clot",
            "Allow a controlled loss to prevent worse damage, then seal the breach",
            level=Level.PATTERN,
            examples=["scuba diver: manage small leak, finish dive, then fix (bleed then clot)", "fleet: drop some messages to prevent queue overflow (bleed, then clot when load drops)", "military: tactical retreat (bleed) to regroup and counterattack (clot)"],
            bridges=["backpressure", "circuit-breaker", "controlled-loss", "resilience"],
            tags=["tactics", "bleed", "clot", "verb"])

        ns.define("probe-before-commit",
            "Send a minimal test request before investing full resources in an action",
            level=Level.PATTERN,
            examples=["order 10 units, test, then order 1000 (probe before commit)", "deploy to 1 server, monitor, then 100 (probe before commit)", "fleet: health check agent before assigning task (probe before commit)"],
            bridges=["testing", "canary", "minimal-investment", "risk-reduction"],
            tags=["tactics", "probe", "commit", "verb"])

        ns.define("hedge-position",
            "Make a counterbalancing investment that reduces downside risk without eliminating upside",
            level=Level.PATTERN,
            examples=["stock + put option: profit if up, limited loss if down", "fleet: primary strategy + backup strategy (hedge)", "backup generator: costs money to maintain, prevents total outage"],
            bridges=["hedging", "backup", "risk-reduction", "diversification"],
            tags=["tactics", "hedge", "backup", "verb"])

    def _load_diagnostics(self):
        ns = self.add_namespace("diagnostics",
            "The art of identifying what's wrong from symptoms, not causes")

        ns.define("symptom-trace",
            "Follow a visible symptom backward through the causal chain to find the root cause",
            level=Level.PATTERN,
            examples=["fever → infection → bacteria → contaminated water → broken pipe (fix pipe)", "fleet: latency → queue → slow consumer → CPU spike → bad algorithm (fix algorithm)", "car: vibration → unbalanced tire → worn bearing (replace bearing)"],
            bridges=["root-cause", "five-whys", "causal-chain", "diagnosis"],
            tags=["diagnostics", "symptom", "trace", "verb"])

        ns.define("diff-and-attribute",
            "Compare current state to a known-good state and attribute each difference to a specific change",
            level=Level.PATTERN,
            examples=["git diff: current vs working, each change attributed to commit", "fleet: snapshot comparison, differences attributed to events", "medical: compare blood work to baseline, attribute each anomaly to condition"],
            bridges=["diff", "attribution", "snapshot", "comparison"],
            tags=["diagnostics", "diff", "attribute", "verb"])

        ns.define("canary-deploy",
            "Route a small fraction of traffic to a new version to detect problems before full rollout",
            level=Level.PATTERN,
            examples=["canary in coal mine: dies first, warns miners", "software: 1% traffic to new version (canary)", "fleet: new gene to 10% of agents (canary group)"],
            bridges=["testing", "gradual-rollout", "risk-control", "monitoring"],
            tags=["diagnostics", "canary", "deploy", "verb"])

        ns.define("smoke-test",
            "Run a quick, shallow validation that catches catastrophic failures without deep verification",
            level=Level.CONCRETE,
            examples=["electronics: plug in, turn on, no smoke = pass", "fleet: health check pings agent, responds = pass", "software: `make && ./run --test` completes without crash = pass"],
            bridges=["testing", "validation", "shallow", "quick"],
            tags=["diagnostics", "smoke-test", "quick", "concrete"])

    def _load_leverage(self):
        ns = self.add_namespace("leverage",
            "Small actions that produce disproportionately large effects — finding and using multipliers")

        ns.define("leverage-point",
            "A place in a system where a small change produces a large effect — Donella Meadows' insight",
            level=Level.META,
            examples=["Meadows: changing tax rate (weak) vs changing success metric (strong)", "fleet: adjusting energy cost (weak) vs changing fitness function (strong)", "organization: adjusting work hours (weak) vs changing culture (strong)"],
            bridges=["systems-thinking", "multiplier", "paradigm", "effectiveness"],
            tags=["leverage", "multiplier", "systems", "meta"])

        ns.define("key-stone",
            "A component whose removal causes disproportionate system collapse — not the biggest, but the most load-bearing",
            level=Level.DOMAIN,
            examples=["arch keystone: one stone, entire arch depends on it", "fleet: confidence type — small but everything depends on it", "species: sea otter (keystone species) — removal collapses kelp forest ecosystem"],
            bridges=["critical-component", "dependency", "load-bearing", "architecture"],
            tags=["leverage", "keystone", "critical", "domain"])

        ns.define("force-multiplier",
            "A tool or technique that amplifies the effectiveness of existing effort without requiring more resources",
            level=Level.DOMAIN,
            examples=["lever: 10 lbs moves 100 lbs (10x)", "HAV: one term replaces paragraph (100x compression)", "shared library: write once, used 100 times (100x)"],
            bridges=["multiplier", "efficiency", "amplification", "leverage"],
            tags=["leverage", "multiplier", "amplification", "domain"])

        ns.define("catalyst",
            "An agent that enables a reaction without being consumed by it — the facilitator, not the participant",
            level=Level.DOMAIN,
            examples=["enzyme: enables chemical reaction without being consumed", "fleet captain: enables coordination without doing the work", "HAV: enables understanding without being part of deliberation"],
            bridges=["facilitator", "enabler", "non-consumed", "activation-energy"],
            tags=["leverage", "catalyst", "facilitator", "domain"])

    def _load_adaptation_patterns(self):
        ns = self.add_namespace("adaptation-patterns",
            "Recurring patterns of how systems adapt to changing conditions")

        ns.define("acclimatize",
            "Gradually adjust to a new environment by incrementally changing operating parameters",
            level=Level.BEHAVIOR,
            examples=["altitude: gradual RBC adjustment over days", "fleet: gradual strategy weight adjustment as environment changes", "diet: gradual change in gut microbiome to new foods"],
            bridges=["gradual", "adaptation", "stabilization", "transition"],
            tags=["adaptation", "acclimatize", "gradual", "verb"])

        ns.define("phase-lock",
            "Synchronize internal oscillation to an external signal, maintaining precise timing alignment",
            level=Level.DOMAIN,
            examples=["circadian rhythm: phase-locks to sun cycle", "PLL: oscillator syncs to reference clock", "fleet agents: task cycles phase-lock to shared timing signals"],
            bridges=["synchronization", "oscillation", "timing", "entrainment"],
            tags=["adaptation", "phase-lock", "sync", "verb"])

        ns.define("cope-and-advance",
            "Manage the immediate crisis while simultaneously making progress toward long-term resolution",
            level=Level.PATTERN,
            examples=["put out fire while designing fireproof building", "fleet: handle failure now (cope) while learning to prevent it (advance)", "startup: fix urgent bug (cope) while improving testing (advance)"],
            bridges=["resilience", "learning", "dual-track", "improvement"],
            tags=["adaptation", "cope", "advance", "verb"])

        ns.define("condition-on-context",
            "Change behavior based on environmental context without explicit if/then rules — the context IS the selector",
            level=Level.PATTERN,
            examples=["chameleon: skin responds directly to light (context IS the selector)", "fleet: circadian cosine directly modulates instinct strength (no if/night rule)", "thermostat: directly responds to temperature (no 'if cold then heat' — just bimetallic strip bending)"],
            bridges=["context-awareness", "direct-modulation", "reactive", "emergent"],
            tags=["adaptation", "context", "condition", "verb"])

    def _load_friction(self):
        ns = self.add_namespace("friction",
            "Resistance that slows systems down — sometimes harmful, sometimes protective")

        ns.define("necessary-friction",
            "Resistance that prevents reckless action and ensures deliberate decision-making",
            level=Level.PATTERN,
            examples=["gun safety catch: friction prevents accidental firing", "confirmation dialog: friction prevents accidental deletion", "fleet deliberation: friction between perception and action prevents impulsive response"],
            bridges=["friction", "safety", "deliberation", "deliberate"],
            tags=["friction", "necessary", "safety", "pattern"])

        ns.define("frictionless-path",
            "A route through a system that encounters zero resistance — dangerously easy to take without thinking",
            level=Level.BEHAVIOR,
            examples=["default settings: most users never change them (frictionless path)", "fleet: instinct action = frictionless (fast but maybe suboptimal)", "checkout: one-click buy = frictionless (easy but maybe impulsive)"],
            bridges=["default", "least-resistance", "fast-path", "design"],
            tags=["friction", "frictionless", "default", "behavior"])

        ns.define("grease-the-path",
            "Remove unnecessary friction to make the desired action the easiest action",
            level=Level.PATTERN,
            examples=["save by default (opt-out): greases the path to saving", "fleet: automatic gene pool sharing: greases the path to knowledge transfer", "Linda tuple space: write anywhere, read anywhere (greased coordination)"],
            bridges=["default", "friction-removal", "nudge", "design"],
            tags=["friction", "grease", "remove", "verb"])

    def _load_compression(self):
        ns = self.add_namespace("compression",
            "Encoding more meaning in less space — the core challenge of knowledge transfer")

        ns.define("glossary-as-code",
            "When shared vocabulary itself becomes executable — terms that trigger predefined behaviors",
            level=Level.META,
            examples=["HAV term 'deliberate' → triggers cuda-deliberation (glossary-as-code)", "military jargon 'flank' → specific maneuver (glossary-as-code)", "medical order 'stat' → immediately (glossary-as-code)"],
            bridges=["vocabulary", "executable", "command", "protocol"],
            tags=["compression", "glossary", "executable", "meta"])

        ns.define("shorthand-convention",
            "An agreed-upon abbreviation that experienced practitioners use to communicate at speed",
            level=Level.PATTERN,
            examples=["medical: SOB = shortness of breath (compresses 3 words to 3 letters)", "fleet: 'confidence-fuse' = harmonic mean fusion with decay and threshold (one word = full spec)", "pilot: 'ILS approach' = instrument landing system approach (3 letters = full procedure)"],
            bridges=["abbreviation", "compression", "shared-knowledge", "speed"],
            tags=["compression", "shorthand", "convention", "pattern"])

        ns.define("schema-on-read",
            "Don't predefine data structure — interpret the meaning when you need it",
            level=Level.PATTERN,
            examples=["JSON: no predefined schema, interpret when needed", "fleet A2A: flexible message payload, each agent interprets as needed", "Wikipedia: unstructured text, each reader extracts what they need"],
            bridges=["flexibility", "interpretation", "JSON", "dynamic"],
            tags=["compression", "schema", "read", "pattern"])

    def _load_boundaries(self):
        ns = self.add_namespace("boundaries",
            "The edges where systems interact — interfaces, perimeters, and transition zones")

        ns.define("membrane-permeability",
            "The rate and selectivity at which a boundary allows substances to pass through",
            level=Level.DOMAIN,
            examples=["cell membrane: selective permeability (oxygen in, toxins out)", "fleet membrane: configurable permeability for gene import (innovation vs safety)", "border: high permeability = free trade + vulnerability, low = protection + stagnation"],
            bridges=["membrane", "filter", "selectivity", "tradeoff"],
            tags=["boundaries", "permeability", "membrane", "domain"])

        ns.define("demilitarized-zone",
            "A buffer region between two hostile systems that enables monitored, controlled interaction",
            level=Level.PATTERN,
            examples=["Korean DMZ: buffer between hostile systems", "fleet: sandbox as DMZ between trusted fleet and untrusted agent", "firewall DMZ: public-facing servers isolated from internal network"],
            bridges=["sandbox", "buffer", "monitoring", "trust-boundary"],
            tags=["boundaries", "DMZ", "buffer", "pattern"])

        ns.define("handshake-protocol",
            "A multi-step exchange that establishes mutual understanding before the main interaction begins",
            level=Level.CONCRETE,
            examples=["TCP: SYN → SYN-ACK → ACK before data", "fleet: capability exchange before coordination", "human: introduce yourself, find common ground, then discuss business (social handshake)"],
            bridges=["protocol", "setup", "mutual-understanding", "A2A"],
            tags=["boundaries", "handshake", "protocol", "concrete"])

    def _load_temporal_patterns(self):
        ns = self.add_namespace("temporal-patterns",
            "How systems change over time — rhythms, cycles, trajectories, and deadlines")

        ns.define("half-life-decay",
            "Exponential decay where half the quantity is lost every fixed period — the universal aging function",
            level=Level.DOMAIN,
            examples=["carbon-14: half-life 5,730 years", "fleet confidence: half-life controls decay rate", "memory: half-life determines how quickly experience fades"],
            bridges=["decay", "exponential", "aging", "universal"],
            tags=["temporal", "half-life", "decay", "domain"])

        ns.define("lead-time",
            "The time between initiating an action and its completion — the response delay",
            level=Level.CONCRETE,
            examples=["custom part: 6-week lead time", "subagent spawn: 2-minute lead time", "fleet: agent startup time is lead time for task assignment"],
            bridges=["delay", "planning", "anticipation", "response"],
            tags=["temporal", "lead-time", "delay", "concrete"])

        ns.define("time-to-live",
            "A deadline after which cached or stored data is considered stale and discarded",
            level=Level.CONCRETE,
            examples=["DNS: 3600s TTL before re-query", "HTTP: max-age before cache invalidation", "fleet: sensor data 500ms TTL, stigmergy marks with expiry"],
            bridges=["expiry", "staleness", "cache", "freshness"],
            tags=["temporal", "TTL", "expiry", "concrete"])

        ns.define("window-of-opportunity",
            "A bounded time interval during which an action can succeed — outside the window, it cannot",
            level=Level.DOMAIN,
            examples=["stock option: expires worthless after date", "launch window: 30 minutes for orbital insertion", "fleet: task deadline creates window, energy budget sets window width"],
            bridges=["deadline", "expiry", "opportunity", "urgency"],
            tags=["temporal", "window", "opportunity", "domain"])

    def _load_quality(self):
        ns = self.add_namespace("quality",
            "Measures and patterns of excellence, reliability, and fitness for purpose")

        ns.define("definition-of-done",
            "The explicit criteria that must be satisfied before a task is considered complete",
            level=Level.CONCRETE,
            examples=["software: tests pass + docs updated + reviewed + deployed + verified = done", "fleet: proposal's DoD = confidence above threshold with supporting evidence", "construction: inspection passed + occupancy permit = done"],
            bridges=["completion", "criteria", "explicit", "agreement"],
            tags=["quality", "done", "criteria", "concrete"])

        ns.define("single-point-of-failure",
            "A component whose failure causes the entire system to stop — eliminate these",
            level=Level.CONCRETE,
            examples=["single hard drive with all data = SPOF", "single load balancer = SPOF", "fleet: single captain = SPOF, need election for backup"],
            bridges=["resilience", "redundancy", "backup", "eliminate"],
            tags=["quality", "SPOF", "failure", "concrete"])

        ns.define("blast-radius",
            "The scope of damage when a failure occurs — contain it, minimize it, monitor it",
            level=Level.PATTERN,
            examples=["firecracker in field vs fireworks factory: same failure, different blast radius", "fleet: bulkhead limits agent failure blast radius", "microservices: one service fails, others unaffected (contained blast radius)"],
            bridges=["bulkhead", "containment", "failure-scope", "resilience"],
            tags=["quality", "blast-radius", "containment", "pattern"])


    def _load_mechanics(self):
        ns = self.add_namespace("mechanics",
            "Physical force, motion, and mechanical advantage as metaphors for agent systems")

        ns.define("mechanical-advantage",
            "A device that multiplies input force — trading distance for force at a fixed energy cost",
            level=Level.PATTERN,
            examples=["lever: 10 lbs input × 1m = 10 lbs output × 1m", "pulley: 10 lbs × 10m = 100 lbs × 1m", "fleet: shared equipment crate = mechanical advantage for all dependents"],
            bridges=["force-multiplier", "leverage", "shared-infrastructure", "efficiency"],
            tags=["mechanics", "advantage", "multiplier", "pattern"])

        ns.define("momentum",
            "The tendency of a moving object to keep moving — resistance to change in velocity",
            level=Level.DOMAIN,
            examples=["freight train: hard to stop (high mass × high velocity)", "fleet agent: long history of success resists strategy change (high momentum)", "project: late-stage rewrite has high momentum (switching costs enormous)"],
            bridges=["inertia", "resistance-to-change", "velocity", "mass"],
            tags=["mechanics", "momentum", "inertia", "domain"])

        ns.define("torque",
            "Rotational force — the ability to turn something around an axis, not just push it linearly",
            level=Level.PATTERN,
            examples=["door: push at handle (high torque) vs near hinge (low torque)", "fleet: small fitness function change at right leverage point = rotates entire gene pool", "organization: suggestion from CEO (high torque) vs intern (low torque) — same idea, different leverage"],
            bridges=["leverage", "rotational-force", "axis", "position"],
            tags=["mechanics", "torque", "leverage", "pattern"])

        ns.define("gear-ratio",
            "Match speed to force through mechanical reduction or multiplication",
            level=Level.PATTERN,
            examples=["bicycle: low gear for hills (slow, powerful), high gear for flats (fast, weak)", "fleet: slow deliberation for critical decisions, fast for routine", "filtration tiers: scout (fast, cheap) vs captain (slow, thorough)"],
            bridges=["tradeoff", "speed-force", "gear", "ratio"],
            tags=["mechanics", "gear-ratio", "tradeoff", "pattern"])

    def _load_entrenchment(self):
        ns = self.add_namespace("entrenchment",
            "How systems become locked into patterns and the difficulty of changing them")

        ns.define("lock-in",
            "A state where switching costs make staying with the current choice cheaper than changing, even if better alternatives exist",
            level=Level.DOMAIN,
            examples=["QWERTY keyboard: inferior but locked in by switching costs", "fleet: cuda-equipment API change affects 100+ crates (lock-in)", "Facebook: locked in by social graph (switching costs = losing all friends)"],
            bridges=["switching-cost", "path-dependence", "standardization", "self-reinforcing"],
            tags=["entrenchment", "lock-in", "switching-cost", "domain"])

        ns.define("path-dependence",
            "Where you end up depends on the path you took, not just where you started — history matters",
            level=Level.DOMAIN,
            examples=["VHS vs Betamax: history determined winner, not quality", "fleet: gene pool evolution depends on starting genes (path-dependent)", "QWERTY: historical accident locked in the standard"],
            bridges=["lock-in", "history", "contingency", "non-deterministic"],
            tags=["entrenchment", "path-dependence", "history", "domain"])

        ns.define("technical-debt",
            "The accumulated cost of choosing fast-now over right-now — future work created by past shortcuts",
            level=Level.BEHAVIOR,
            examples=["ship without tests → add tests later at 3x cost (technical debt + interest)", "skip documentation → reverse-engineer later at 5x cost", "fleet: minimal agent → add compliance later at higher cost"],
            bridges=["debt", "shortcuts", "interest", "maintenance"],
            tags=["entrenchment", "debt", "shortcuts", "behavior"])

        ns.define("legacy-anchor",
            "An old component that newer components must remain compatible with, constraining evolution",
            level=Level.BEHAVIOR,
            examples=["Windows: runs 30-year-old software (legacy anchor)", "HTTP/1.1: compatible with HTTP/1.0 (legacy anchor)", "fleet: cuda-equipment v0.1 API stability constrains future evolution"],
            bridges=["backward-compatibility", "constraint", "stability", "evolution"],
            tags=["entrenchment", "legacy", "anchor", "behavior"])

    def _load_knowledge_transfer(self):
        ns = self.add_namespace("knowledge-transfer",
            "How knowledge moves between agents, systems, and generations")

        ns.define("standing-on-shoulders",
            "Each generation builds on accumulated knowledge, achieving heights impossible from scratch",
            level=Level.META,
            examples=["Newton: standing on giants' shoulders", "fleet: every new crate starts with cuda-equipment (accumulated fleet knowledge)", "open source: every project builds on libraries (standing on shoulders)"],
            bridges=["accumulation", "prior-knowledge", "foundation", "meta"],
            tags=["knowledge", "transfer", "accumulation", "meta"])

        ns.define("knowledge-distillation",
            "Transfer knowledge from a complex source to a simpler representation without losing essential capability",
            level=Level.META,
            examples=["teacher 70B → student 7B: 90% capability, 10% cost", "HAV: 30-page paper → 200-word term: 95% meaning, 1% cost", "senior engineer's intuition → documented pattern: transferable at lower cost"],
            bridges=["distillation", "compression", "efficiency", "lossy"],
            tags=["knowledge", "distillation", "compression", "meta"])

        ns.define("apprenticeship",
            "Transfer tacit knowledge through observation and guided practice, not explicit instruction",
            level=Level.PATTERN,
            examples=["blacksmith apprentice: watches master, practices, gets corrected", "fleet: gene crossover from successful agents = apprenticeship", "residency: medical students observe attending physicians (apprenticeship)"],
            bridges=["tacit-knowledge", "practice", "observation", "learning"],
            tags=["knowledge", "apprenticeship", "tacit", "pattern"])

        ns.define("document-or-it-didn't-happen",
            "Knowledge that exists only in someone's head is not knowledge — it's a dependency on that person",
            level=Level.PATTERN,
            examples=["senior engineer leaves → undocumented deployment process lost", "fleet: HAV terms documented = transferable regardless of which agent wrote them", "science: experiment results not published = didn't contribute to collective knowledge"],
            bridges=["documentation", "persistence", "transferability", "SPOF"],
            tags=["knowledge", "documentation", "persistence", "pattern"])

    def _load_efficiency(self):
        ns = self.add_namespace("efficiency",
            "Doing more with less — the art of minimizing waste")

        ns.define("amortize",
            "Spread a one-time cost over many uses, making each individual use cheaper",
            level=Level.DOMAIN,
            examples=["coffee machine: $1000 first cup, $1 after 1000 cups", "fleet: cuda-equipment dev cost amortized over 100+ crates", "factory: machine purchase amortized over production volume"],
            bridges=["fixed-cost", "shared-infrastructure", "economy-of-scale", "amortization"],
            tags=["efficiency", "amortize", "shared-cost", "domain"])

        ns.define("pay-once-use-forever",
            "Invest in a capability that provides ongoing returns without additional cost",
            level=Level.PATTERN,
            examples=["hammer: buy once, use forever", "test framework: build once, catch bugs forever", "fleet: shared crate → use in 100+ crates with zero additional development cost"],
            bridges=["investment", "infrastructure", "reusable", "one-time-cost"],
            tags=["efficiency", "pay-once", "reusable", "pattern"])

        ns.define("waste-heat",
            "Unavoidable byproduct energy that can potentially be repurposed instead of discarded",
            level=Level.PATTERN,
            examples=["engine heat → cabin heater (repurposed waste-heat)", "fleet: rejected proposals → lessons about what doesn't work (repurposed waste-heat)", " composting: food waste → fertilizer (repurposed waste)"],
            bridges=["byproduct", "repurpose", "efficiency", "waste"],
            tags=["efficiency", "waste-heat", "repurpose", "pattern"])

        ns.define("coasting",
            "Operating at zero or minimal input cost by relying on accumulated energy or momentum",
            level=Level.BEHAVIOR,
            examples=["bicycle downhill: zero pedaling, still moving (coasting)", "solar at noon: zero fuel, maximum output", "fleet: high-fitness agent handles routine tasks by instinct (coasting)"],
            bridges=["momentum", "stored-energy", "efficiency", "minimal-input"],
            tags=["efficiency", "coasting", "minimal", "behavior"])

    def _load_metaphor(self):
        ns = self.add_namespace("metaphor",
            "Cross-domain mapping as a cognitive tool — understanding new things through familiar structures")

        ns.define("structural-mapping",
            "Transfer understanding from a well-known domain to a less-known domain by matching relational structure",
            level=Level.META,
            examples=["electricity = water: structural mapping (voltage=pressure, current=flow)", "dopamine = confidence: structural mapping (accumulation, decay, threshold, modulation)", "gene = strategy: structural mapping (inheritance, mutation, selection, fitness)"],
            bridges=["analogy", "cross-domain", "structure", "understanding"],
            tags=["metaphor", "mapping", "structure", "meta"])

        ns.define("reification",
            "Treating an abstract concept as if it were a concrete, manipulable object",
            level=Level.META,
            examples=["time is money: abstract time treated as concrete currency", "fleet: confidence as f64 type — abstract concept treated as concrete number", "gene as object: abstract strategy pattern treated as manipulable data structure"],
            bridges=["concretization", "type-system", "manipulation", "abstraction"],
            tags=["metaphor", "reification", "concrete", "meta"])

        ns.define("metaphorical-distance",
            "How far apart the source and target domains of a metaphor are — distance affects both insight and confusion",
            level=Level.META,
            examples=["stock market = roller coaster (close distance, safe but shallow)", "consciousness = computation (far distance, risky but insightful)", "fleet: biology → agent systems (moderate distance, productive mapping)"],
            bridges=["metaphor", "domain-distance", "insight", "risk"],
            tags=["metaphor", "distance", "domain", "meta"])


    def _load_scaling(self):
        ns = self.add_namespace("scaling-deep",
            "How systems grow, shrink, and maintain function at different scales")

        ns.define("constant-overhead",
            "The cost that doesn't increase as the system scales — the fixed baseline regardless of size",
            level=Level.CONCRETE,
            examples=["factory rent: same whether 1 or 10,000 units produced", "fleet: cuda-equipment binary size is constant regardless of how many crates use it", "HTTP headers: same size whether payload is 1 byte or 1GB"],
            bridges=["fixed-cost", "overhead", "baseline", "scale-invariant"],
            tags=["scaling", "constant", "overhead", "concrete"])

        ns.define("linear-scaling",
            "Output grows proportionally with input — double the resources, double the capacity",
            level=Level.DOMAIN,
            examples=["workers: 2x workers = 2x output (ideally)", "fleet: 10 agents handle ~10x tasks (ideally linear)", "servers: 10x servers = ~10x capacity (before coordination overhead)"],
            bridges=["scaling", "proportional", "ideal", "baseline"],
            antonyms=["super-linear-scaling", "sub-linear-scaling"],
            tags=["scaling", "linear", "proportional", "domain"])

        ns.define("super-linear-scaling",
            "Output grows faster than input — 2x resources produce MORE than 2x output",
            level=Level.DOMAIN,
            examples=["telephone network: value ~ n^2 (each user connects to all others)", "fleet gene pool: each new gene combines with all existing genes (super-linear value)", "marketplace: each new seller benefits all existing buyers"],
            bridges=["network-effect", "combinatorial", "positive-feedback", "scaling"],
            antonyms=["sub-linear-scaling"],
            tags=["scaling", "super-linear", "network-effect", "domain"])

        ns.define("diminishing-returns",
            "Each additional unit of input produces LESS output than the previous one",
            level=Level.DOMAIN,
            examples=["workers: 1st produces 100, 2nd produces 90, 10th produces 10", "fleet deliberation: 1st round most valuable, each subsequent round adds less", "studying: first hour most productive, 10th hour almost useless"],
            bridges=["overhead", "congestion", "optimal-stopping", "tradeoff"],
            antonyms=["increasing-returns"],
            tags=["scaling", "diminishing", "returns", "domain"])

        ns.define("right-sizing",
            "Matching system capacity to actual demand — not over-provisioned, not under-provisioned",
            level=Level.PATTERN,
            examples=["100-person office for 100-person company (right-sized)", "fleet: simple task gets scout-class agent, not captain-class", "database: query capacity matches actual query volume"],
            bridges=["capacity", "demand", "provisioning", "efficiency"],
            tags=["scaling", "right-size", "capacity", "pattern"])

    def _load_interface_patterns(self):
        ns = self.add_namespace("interface-patterns",
            "How system boundaries are designed, crossed, and maintained")

        ns.define("contract-first",
            "Define the interface contract before implementing either side — agreement before code",
            level=Level.PATTERN,
            examples=["OpenAPI spec before client/server implementation", "fleet: vessel.json defines vessel contract before building vessel", "TypeScript interface before implementing class"],
            bridges=["interface", "contract", "agreement", "specification"],
            tags=["interface", "contract-first", "specification", "pattern"])

        ns.define("version-with-grace",
            "Support old and new interface versions simultaneously during transition, then deprecate old",
            level=Level.PATTERN,
            examples=["HTTP/2 coexists with HTTP/1.1 during transition", "Python 2 and 3 coexisted for years", "fleet: cuda-equipment versions coexist during transition period"],
            bridges=["backward-compatibility", "deprecation", "migration", "transition"],
            tags=["interface", "versioning", "graceful", "pattern"])

        ns.define("soft-error",
            "A failure that doesn't crash the system but returns a degraded result with error context",
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
            level=Level.DOMAIN,
            examples=["human: ~4 hours deep focus per day (attention budget)", "fleet: agent attention budget allocated across deliberation, messages, perception", "RAM: fixed memory budget allocated across processes"],
            bridges=["attention", "budget", "prioritization", "allocation"],
            tags=["attention", "budget", "allocation", "domain"])

        ns.define("salience-bottleneck",
            "The rate at which important information can be identified and acted upon — the awareness throughput",
            level=Level.DOMAIN,
            examples=["human: can notice 3-5 dashboard anomalies simultaneously", "fleet: saliency scoring filters to top-N inputs (salience bottleneck)", "news: 1000 stories published, only 5 noticed (salience bottleneck)"],
            bridges=["attention", "bottleneck", "salience", "throughput"],
            tags=["attention", "salience", "bottleneck", "domain"])

        ns.define("habituation",
            "Decreased response to a stimulus after repeated exposure — the brain's noise filter",
            level=Level.BEHAVIOR,
            examples=["ticking clock: noticed at first, invisible after 10 minutes", "fleet: constant sensor reading loses attention, only changes break through", "smell: you notice perfume when entering a room, not after 5 minutes"],
            bridges=["adaptation", "change-detection", "noise-filter", "desensitization"],
            tags=["attention", "habituation", "adaptation", "behavior"])

    def _load_security_deep(self):
        ns = self.add_namespace("security-deep",
            "Threat models, trust boundaries, and defensive patterns")

        ns.define("threat-model",
            "An explicit enumeration of what you're protecting against — define the enemy before building defenses",
            level=Level.CONCRETE,
            examples=["bank: protects against robbers, insiders, subpoenas (threat model)", "fleet membrane: antibodies list specific threats (threat model)", "web app: OWASP top 10 as threat model → build defenses accordingly"],
            bridges=["defense", "threat", "model", "explicit"],
            tags=["security", "threat-model", "defense", "concrete"])

        ns.define("trust-boundary",
            "A line where trust assumptions change — inside the boundary is trusted, outside is not",
            level=Level.CONCRETE,
            examples=["front door: trust boundary between home (trusted) and street (untrusted)", "fleet membrane: trust boundary between trusted genes and external signals", "firewall: trust boundary between internal network and internet"],
            bridges=["membrane", "boundary", "authentication", "trust"],
            tags=["security", "boundary", "trust", "concrete"])

        ns.define("defense-in-depth",
            "Multiple independent layers of security — if one fails, the next catches the threat",
            level=Level.PATTERN,
            examples=["castle: moat + wall + guards + keep (defense in depth)", "fleet: sandbox + compliance + RBAC + membrane (defense in depth)", "web: WAF + rate limit + auth + encrypt (defense in depth)"],
            bridges=["layers", "redundancy", "independent", "defense"],
            tags=["security", "defense-in-depth", "layers", "pattern"])

        ns.define("least-privilege",
            "Grant only the minimum permissions needed for a task — no more, no exceptions",
            level=Level.CONCRETE,
            examples=["janitor: cleaning closet keys, not CEO office (least privilege)", "database: read-only user can view but not modify", "fleet: monitoring agent observes but doesn't act (least privilege)"],
            bridges=["permission", "minimum", "RBAC", "blast-radius"],
            tags=["security", "least-privilege", "minimum", "concrete"])

    def _load_error_strategies(self):
        ns = self.add_namespace("error-strategies",
            "Systematic approaches to handling failures gracefully")

        ns.define("fail-fast",
            "Detect and report errors immediately rather than continuing in a degraded state",
            level=Level.PATTERN,
            examples=["Rust: Result type forces explicit error handling (fail-fast)", "fleet: critical path failure → immediate error, not silent degradation", "assert: program crashes on invariant violation (fail-fast)"],
            bridges=["error-handling", "visibility", "explicit", "fail"],
            antonyms=["fail-silently"],
            tags=["error", "fail-fast", "explicit", "pattern"])

        ns.define("fail-safe",
            "When failure occurs, transition to a state that is safe, not just stopped",
            level=Level.PATTERN,
            examples=["fuse blows → house doesn't burn (fail-safe)", "train signal fails → red → trains stop (fail-safe)", "fleet: energy depleted → rest mode (fail-safe, not chaos)"],
            bridges=["safety", "default-state", "failure-mode", "graceful"],
            antonyms=["fail-deadly"],
            tags=["error", "fail-safe", "safety", "pattern"])

        ns.define("retry-with-backoff",
            "Attempt a failed operation again, but with increasing delay between attempts",
            level=Level.CONCRETE,
            examples=["network timeout: retry at 1s, 2s, 4s, 8s (exponential backoff)", "fleet: provider timeout → retry with backoff → eventually succeeds", "git push conflict: retry after delay → someone else pushed, now pull-able"],
            bridges=["retry", "backoff", "transient-failure", "patience"],
            tags=["error", "retry", "backoff", "concrete"])

        ns.define("circuit-breaker",
            "Stop attempting a failing operation after repeated failures, allowing time for recovery",
            level=Level.CONCRETE,
            examples=["electrical circuit breaker: trips on overload, resets after cooldown", "fleet: failing provider → circuit opens → no attempts → half-open test → resume or re-open", "API gateway: failing backend → circuit opens → traffic diverted"],
            bridges=["retry", "failure-detection", "recovery", "protection"],
            tags=["error", "circuit-breaker", "recovery", "concrete"])

    def _load_coordination_deep(self):
        ns = self.add_namespace("coordination-deep",
            "How autonomous agents achieve collective outcomes without central control")

        ns.define("implicit-contract",
            "An agreement that exists through consistent behavior, not explicit negotiation",
            level=Level.BEHAVIOR,
            examples=["roommate chore pattern: implicit contract without written agreement", "fleet: stigmergy patterns form implicit behavioral contracts", "driving culture: lane usage patterns = implicit contract between drivers"],
            bridges=["stigmergy", "pattern", "emergent", "convention"],
            antonyms=["explicit-contract"],
            tags=["coordination", "implicit", "contract", "behavior"])

        ns.define("barrier",
            "A synchronization point where multiple agents wait until all have arrived before proceeding",
            level=Level.CONCRETE,
            examples=["marathon start: all wait, then all go simultaneously", "team meeting: wait for all participants before starting", "fleet workflow: step waits for all agent subtasks to complete before proceeding"],
            bridges=["synchronization", "wait", "concurrent", "gate"],
            tags=["coordination", "barrier", "sync", "concrete"])

        ns.define("gossip-protocol",
            "Information spreads by each node sharing with a few random neighbors, achieving global coverage",
            level=Level.PATTERN,
            examples=["rumor spreading: tell 3, they tell 3, exponential coverage", "fleet trust: gossip spreads trust assessments across all agents", "cluster membership: gossip protocol spreads node join/leave events"],
            bridges=["gossip", "epidemic", "eventual-consistency", "random"],
            tags=["coordination", "gossip", "spread", "pattern"])

    def _load_composition(self):
        ns = self.add_namespace("composition",
            "Building complex systems from simpler, independently useful parts")

        ns.define("plug-and-play",
            "Components that interoperate without configuration — connect them and they work",
            level=Level.PATTERN,
            examples=["USB: plug in device, it works (plug-and-play)", "fleet: define equipment type, automatically integrates everywhere", "HDMI: plug monitor in, it works (plug-and-play)"],
            bridges=["interface", "standard", "auto-integration", "compatibility"],
            tags=["composition", "plug-and-play", "auto", "pattern"])

        ns.define("mix-and-match",
            "Combine components freely from a menu of options to create custom configurations",
            level=Level.PATTERN,
            examples=["burrito bowl: choose from menu of components (mix-and-match)", "fleet: perception + deliberation + motor = agent (mix-and-match crates)", "PC building: choose CPU, GPU, RAM, storage independently (mix-and-match)"],
            bridges=["composition", "menu", "custom", "modular"],
            tags=["composition", "mix-and-match", "modular", "pattern"])

        ns.define("side-effect-free",
            "A computation that produces a result without modifying any external state — pure function",
            level=Level.PATTERN,
            examples=["f(x) = x + 1: pure, no side effects, always same output for same input", "fleet: Conf::fuse(a, b): pure function, no state mutation", "map function in functional programming: pure, composable"],
            bridges=["pure-function", "composability", "testability", "functional"],
            tags=["composition", "side-effect-free", "pure", "pattern"])

    def _load_observability(self):
        ns = self.add_namespace("observability",
            "Understanding what a system is doing from the outside, without modifying it")

        ns.define("telemetry",
            "Automated collection and transmission of system health and performance data",
            level=Level.CONCRETE,
            examples=["plane black box: automated recording of all flight data", "car OBD: real-time engine telemetry", "fleet: cuda-metrics collects agent performance telemetry"],
            bridges=["monitoring", "health", "automated", "data-collection"],
            tags=["observability", "telemetry", "monitoring", "concrete"])

        ns.define("distributed-trace",
            "Follow a single request across multiple services to understand the full lifecycle and find bottlenecks",
            level=Level.CONCRETE,
            examples=["HTTP request traced through frontend → auth → inventory → payment", "fleet: deliberation traced through proposal → endorsements → decision", "package delivery tracked through warehouse → truck → door (trace)"],
            bridges=["trace", "lifecycle", "bottleneck", "provenance"],
            tags=["observability", "trace", "distributed", "concrete"])

        ns.define("canary-in-the-coal-mine",
            "An early warning indicator that detects problems before they affect the whole system",
            level=Level.PATTERN,
            examples=["coal mine canary: sensitive to gas, dies first (early warning)", "software: synthetic transaction detects degradation before user impact", "fleet: health check detects performance degradation before full failure"],
            bridges=["early-warning", "canary", "proactive", "monitoring"],
            tags=["observability", "canary", "early-warning", "pattern"])

    def _load_anti_patterns(self):
        ns = self.add_namespace("anti-patterns",
            "Common solutions that seem good but create more problems than they solve")

        ns.define("god-object",
            "A component that knows too much and does too much — the opposite of focused responsibility",
            level=Level.BEHAVIOR,
            examples=["one class doing DB + logic + UI + email (god-object)", "fleet: single agent doing everything instead of specialized crates (god-object)", "one person doing CEO + CFO + CTO + janitor (god-object)"],
            bridges=["separation-of-concerns", "modularity", "specialization", "responsibility"],
            tags=["anti-pattern", "god-object", "monolith", "behavior"])

        ns.define("shotgun-surgery",
            "A single change requires modifications in many unrelated places — scattered responsibility",
            level=Level.BEHAVIOR,
            examples=["data format change: edit DB + API + client + tests + docs + monitoring (shotgun surgery)", "fleet: confidence type change affecting 50 files (shotgun surgery)", "price change: update website + app + database + billing + contracts (shotgun surgery)"],
            bridges=["coupling", "scattered-responsibility", "centralization", "maintenance"],
            tags=["anti-pattern", "shotgun", "coupling", "behavior"])

        ns.define("premature-abstraction",
            "Creating shared infrastructure before the pattern is clear — building for a future that doesn't arrive",
            level=Level.BEHAVIOR,
            examples=["shared library after seeing pattern in 3 modules = good timing", "shared library BEFORE seeing the pattern = premature abstraction", "fleet: build 3 concrete agents, then extract shared equipment (not before)"],
            bridges=["abstraction", "timing", "pattern", "premature"],
            antonyms=["mature-abstraction"],
            tags=["anti-pattern", "premature", "abstraction", "behavior"])

        ns.define("cargo-cult",
            "Copying the form of a solution without understanding why it works — superficial imitation",
            level=Level.BEHAVIOR,
            examples=["bamboo control towers lit fires expecting cargo planes (cargo cult)", "fleet: copying architecture without understanding why it was chosen (cargo cult)", "standup meetings every day because Google does them (cargo cult)"],
            bridges=["imitation", "understanding", "form-vs-function", "superficial"],
            tags=["anti-pattern", "cargo-cult", "imitation", "behavior"])


    def _load_influence(self):
        ns = self.add_namespace("influence",
            "How one agent's behavior affects another's — without force, through structure and example")

        ns.define("set-the-norm",
            "Establish a behavioral baseline that new agents adopt by default through observation",
            level=Level.BEHAVIOR,
            examples=["elevator: first person faces door, everyone follows (norm set)", "fleet: first agent's deliberation style sets norm for subsequent agents", "office: first team's communication style becomes team culture (norm)"],
            bridges=["norm", "conformity", "baseline", "first-mover"],
            tags=["influence", "norm", "default", "behavior"])

        ns.define("nudge",
            "A small structural change that steers behavior without restricting choice",
            level=Level.PATTERN,
            examples=["cafeteria: healthy food at eye level (nudge toward healthy eating)", "fleet: energy costs for verbose messages (nudge toward conciseness)", "default 401k contribution: opt-out not opt-in (nudge toward saving)"],
            bridges=["choice-architecture", "structural", "steer", "subtle"],
            tags=["influence", "nudge", "structural", "pattern"])

        ns.define("information-cascade",
            "Individuals ignore their private information and follow the crowd, amplifying possibly wrong signals",
            level=Level.BEHAVIOR,
            examples=["restaurant line: join A because others did, not because it's better", "fleet: agents voting for popular option regardless of own evidence (cascade)", "stock market: buying because others buy, not because of analysis (cascade)"],
            bridges=["conformity", "herd-behavior", "amplification", "crowd"],
            tags=["influence", "cascade", "herd", "behavior"])

    def _load_negotiation(self):
        ns = self.add_namespace("negotiation",
            "How agents reach agreements through structured interaction")

        ns.define("best-alternative",
            "The most attractive option available if the current negotiation fails — your walk-away point",
            level=Level.DOMAIN,
            examples=["job negotiation: best alternative offer determines your leverage", "fleet: instinct-based action is the agent's BATNA (walk-away point)", "buying a house: comparable listings are your BATNA (negotiation leverage)"],
            bridges=["negotiation", "alternative", "walk-away", "leverage"],
            tags=["negotiation", "BATNA", "alternative", "domain"])

        ns.define("concession-rhythm",
            "The pattern and rate at which parties make concessions — reveals information and sets expectations",
            level=Level.PATTERN,
            examples=["price negotiation: concession sizes reveal limits", "fleet: deliberation round compromises reveal confidence levels", "diplomacy: each country's concessions signal their priorities"],
            bridges=["negotiation", "rhythm", "signal", "information"],
            tags=["negotiation", "concession", "rhythm", "pattern"])

        ns.define("package-deal",
            "Bundle multiple issues together so that concessions on one offset gains on another, creating win-win",
            level=Level.PATTERN,
            examples=["job offer: lower salary + remote + vacation (package better than raw salary)", "fleet: more compute, less network (package deal on resource budgets)", "trade agreement: lower tariffs on X, lower subsidies on Y (package deal)"],
            bridges=["negotiation", "bundling", "win-win", "Pareto"],
            tags=["negotiation", "package", "bundle", "pattern"])

    def _load_capacity(self):
        ns = self.add_namespace("capacity",
            "The limits of what a system can handle and how it responds to being pushed beyond them")

        ns.define("headroom",
            "Unused capacity maintained as a buffer against demand spikes — safety margin",
            level=Level.CONCRETE,
            examples=["bridge rated 10,000 lbs handling 7,000 lb trucks (30% headroom)", "server at 60% CPU (40% headroom)", "fleet: 80% energy allocation, 20% headroom for emergencies"],
            bridges=["buffer", "safety-margin", "capacity", "reserve"],
            tags=["capacity", "headroom", "buffer", "concrete"])

        ns.define("saturation",
            "The point where adding more input produces no additional output — the system is maxed out",
            level=Level.DOMAIN,
            examples=["saturated sponge: more water runs off (no absorption)", "saturated road: more cars = congestion, not throughput", "fleet agent at message processing limit: more messages = queue/drop"],
            bridges=["capacity", "max", "congestion", "limit"],
            tags=["capacity", "saturation", "limit", "domain"])

        ns.define("elasticity",
            "The ability to expand and contract capacity in response to demand — stretch without breaking",
            level=Level.DOMAIN,
            examples=["cloud: 2→20 servers during spike, back to 2 after (elasticity)", "fleet: spawn agents during load spike, terminate when idle (elasticity)", "rubber band: stretches under tension, returns when released (elasticity)"],
            bridges=["scaling", "auto-scale", "responsive", "capacity"],
            tags=["capacity", "elasticity", "auto-scale", "domain"])

    def _load_decision_patterns(self):
        ns = self.add_namespace("decision-patterns",
            "Structural patterns for how decisions are made, evaluated, and reversed")

        ns.define("reversible-decision",
            "A choice that can be undone at low cost — make it fast, don't deliberate",
            level=Level.PATTERN,
            examples=["font choice: reversible (just change it later)", "fleet: switching comm partner = reversible (low cost)", "marriage: irreversible (high deliberation warranted)"],
            bridges=["decision", "reversibility", "speed", "cost"],
            antonyms=["irreversible-decision"],
            tags=["decision", "reversible", "fast", "pattern"])

        ns.define("irreversible-decision",
            "A choice with high reversal cost — deliberate carefully, get it right",
            level=Level.PATTERN,
            examples=["selling house: irreversible, high reversal cost", "fleet: published crate API = irreversible (affects all dependents)", "getting a tattoo: irreversible, deliberate first"],
            bridges=["decision", "irreversibility", "deliberation", "cost"],
            antonyms=["reversible-decision"],
            tags=["decision", "irreversible", "deliberate", "pattern"])

        ns.define("satisfice",
            "Choose the first option that meets minimum requirements rather than searching for the optimal one",
            level=Level.DOMAIN,
            examples=["finding a restaurant: first one above 4 stars (satisfice), not exhaustively checking all", "fleet: accept first proposal above confidence threshold (satisfice)", "hiring: first candidate above minimum requirements (satisfice vs optimize)"],
            bridges=["bounded-rationality", "threshold", "good-enough", "practical"],
            antonyms=["maximize"],
            tags=["decision", "satisfice", "threshold", "domain"])

        ns.define("two-door-problem",
            "Choosing between two good options with incomplete information — neither is clearly better",
            level=Level.BEHAVIOR,
            examples=["two good job offers, can't determine which is better (two-door)", "fleet: two good proposals, neither clearly superior (two-door)", "two houses, both nice, neither perfect (two-door)"],
            bridges=["uncertainty", "satisfice", "choice", "ambiguity"],
            tags=["decision", "two-door", "choice", "behavior"])

    def _load_maintenance(self):
        ns = self.add_namespace("maintenance",
            "Keeping systems running and improving over time — the unglamorous essential work")

        ns.define("rot-prevention",
            "Regular small interventions that prevent accumulated degradation from causing failure",
            level=Level.PATTERN,
            examples=["wood painting every 5 years prevents rot", "software dependency updates prevent rot", "fleet gene pool maintenance prevents accumulation of bad genes"],
            bridges=["prevention", "maintenance", "regular", "degradation"],
            tags=["maintenance", "rot", "prevention", "pattern"])

        ns.define("watchdog",
            "An independent monitor that alerts when a system deviates from expected behavior",
            level=Level.CONCRETE,
            examples=["heart monitor: detects arrhythmia, alerts (doesn't fix)", "smoke detector: detects fire, alerts (doesn't extinguish)", "fleet health check: detects degradation, alerts (doesn't fix)"],
            bridges=["monitoring", "alert", "detection", "independent"],
            tags=["maintenance", "watchdog", "monitor", "concrete"])

        ns.define("graceful-aging",
            "A system designed to remain useful even as it accumulates technical debt and falls behind current best practices",
            level=Level.PATTERN,
            examples=["old building with good bones: renovate incrementally (graceful aging)", "Python 2: didn't age gracefully (forced deprecation)", "fleet crate v0.1: still works even if v0.2 exists (graceful aging)"],
            bridges=["backward-compatibility", "longevity", "maintenance", "incremental"],
            tags=["maintenance", "aging", "graceful", "pattern"])

    def _load_ux(self):
        ns = self.add_namespace("ux-patterns",
            "How systems present themselves to users and agents for maximum usability")

        ns.define("progressive-disclosure",
            "Show only what's needed now, reveal more complexity as the user goes deeper",
            level=Level.PATTERN,
            examples=["Google: one search box, advanced options behind link", "fleet config: defaults work immediately, advanced overrides available", "camera app: auto mode by default, manual mode behind a menu"],
            bridges=["simplicity", "layered", "onboarding", "complexity"],
            tags=["ux", "progressive", "disclosure", "pattern"])

        ns.define("affordance",
            "A design property that suggests how to interact with an object — the handle says 'pull'",
            level=Level.PATTERN,
            examples=["door handle: affords pulling (you know to pull without instruction)", "button: affords clicking", "vessel.json structure affords reading capabilities (self-documenting)"],
            bridges=["intuitive", "self-documenting", "design", "interaction"],
            tags=["ux", "affordance", "intuitive", "pattern"])

        ns.define("error-recovery",
            "When something goes wrong, show the user what happened AND how to fix it, not just what went wrong",
            level=Level.PATTERN,
            examples=["bad: 'Error 404'. good: 'Page moved, here are alternatives'", "fleet: inconclusive deliberation includes evidence + suggestions for next steps", "compiler: 'unexpected token' WITH caret showing exact location (error recovery)"],
            bridges=["error-handling", "usability", "actionable", "recovery"],
            tags=["ux", "error-recovery", "actionable", "pattern"])


    def _load_trade_patterns(self):
        ns = self.add_namespace("trade-patterns",
            "Exchange dynamics — what agents give up to get what they need")

        ns.define("barter",
            "Direct exchange of capabilities between agents without a common currency",
            level=Level.PATTERN,
            examples=["perception agent trades data for computation agent's analysis", "humans: barter services without money", "fleet: sensor data for computation results (direct barter)"],
            bridges=["exchange", "complementarity", "mutual-benefit", "direct"],
            tags=["trade", "barter", "exchange", "pattern"])

        ns.define("spot-price",
            "The current market-determined price based on immediate supply and demand — no contracts, no negotiation",
            level=Level.CONCRETE,
            examples=["Uber surge: price reflects current demand (spot price)", "fleet: deliberation costs 2.0 ATP now, 3.0 when energy is low (spot price)", "electricity: minute-by-minute price based on grid supply/demand"],
            bridges=["price", "dynamic", "supply-demand", "real-time"],
            tags=["trade", "spot-price", "dynamic", "concrete"])

        ns.define("bid-ask-spread",
            "The gap between what buyers will pay and sellers will accept — the cost of making a trade",
            level=Level.DOMAIN,
            examples=["stock bid $100, ask $101: spread = $1 (cost of trading)", "fleet: task budget 1.5 ATP, agent wants 2.0 ATP: spread = 0.5 (task unassigned)", "real estate: seller wants $500K, buyer offers $475K: spread = $25K"],
            bridges=["market", "liquidity", "trade-cost", "gap"],
            tags=["trade", "spread", "market", "domain"])

    def _load_morphology(self):
        ns = self.add_namespace("morphology-deep",
            "How structures change shape and form — transformation patterns")

        ns.define("phase-transition",
            "A sudden qualitative change in system behavior when a parameter crosses a critical threshold",
            level=Level.DOMAIN,
            examples=["water: liquid at 99°C, gas at 101°C (phase transition)", "fleet: isolated agents → emergent coordination at critical mass (phase transition)", "magnet: non-magnetic below Curie temp, magnetic above (phase transition)"],
            bridges=["critical-mass", "sudden-change", "qualitative", "threshold"],
            tags=["morphology", "phase", "transition", "domain"])

        ns.define("catalytic-conversion",
            "Transform input into a fundamentally different output through a process that itself doesn't change",
            level=Level.PATTERN,
            examples=["catalytic converter: toxic → harmless (converter unchanged)", "fleet: raw signal → percept → decision (each stage transforms without consuming)", "compiler: source code → machine code (compiler unchanged)"],
            bridges=["transformation", "pipeline", "type-change", "catalyst"],
            tags=["morphology", "catalytic", "transform", "pattern"])

        ns.define("self-assembly",
            "Components spontaneously organize into a structured whole without external direction",
            level=Level.DOMAIN,
            examples=["protein folding: amino acids self-assemble into 3D structure", "crystal growth: atoms self-assemble into lattice", "fleet: agents self-assemble into routing networks via stigmergy"],
            bridges=["emergence", "spontaneous", "local-rules", "global-order"],
            tags=["morphology", "self-assembly", "spontaneous", "domain"])

    def _load_risk_patterns(self):
        ns = self.add_namespace("risk-patterns",
            "How systems identify, assess, and manage uncertainty about future outcomes")

        ns.define("black-swan",
            "An event that is extremely rare, has massive impact, and is only predictable in hindsight",
            level=Level.META,
            examples=["2008 financial crisis: unpredictable, massive impact", "COVID: unpredictable, massive impact", "fleet: chaos monkey injects unpredictable failures to test resilience"],
            bridges=["fat-tail", "unpredictable", "catastrophic", "meta"],
            tags=["risk", "black-swan", "unpredictable", "meta"])

        ns.define("expected-value",
            "The probability-weighted average of all possible outcomes — the rational decision metric",
            level=Level.CONCRETE,
            examples=["lottery: $1B × 1/300M = $3.33 expected value", "fleet: confidence × payoff = expected value of proposal", "insurance: premium < expected loss = worth buying"],
            bridges=["probability", "payoff", "rational", "decision"],
            tags=["risk", "expected-value", "probability", "concrete"])

        ns.define("worst-case-budget",
            "Reserve resources specifically for surviving the worst plausible scenario",
            level=Level.CONCRETE,
            examples=["household: 3 months expenses as worst-case budget", "data center: backup generators for worst-case power failure", "fleet: reserved energy for safe shutdown in worst case"],
            bridges=["reserve", "insurance", "worst-case", "survival"],
            tags=["risk", "worst-case", "budget", "concrete"])

        ns.define("tail-hedging",
            "Positioning to benefit from extreme events rather than just surviving them",
            level=Level.META,
            examples=["out-of-money puts: worthless normally, enormous in crash", "fleet: diverse gene pool (small maintenance cost, enormous payoff in environment change)", "multi-cloud deployment: extra cost normally, lifesaver if one provider fails"],
            bridges=["hedging", "tail-risk", "extreme-event", "profit"],
            tags=["risk", "tail-hedge", "extreme", "meta"])

    def _load_propagation(self):
        ns = self.add_namespace("propagation",
            "How signals, effects, and changes spread through systems")

        ns.define("blast-propagation",
            "A failure or signal that spreads outward from its origin like an explosion",
            level=Level.BEHAVIOR,
            examples=["server crash cascading to dependent services (blast propagation)", "rumor spreading exponentially (blast propagation)", "fleet: one agent failure cascading without bulkheads (blast propagation)"],
            bridges=["cascading-failure", "propagation", "amplification", "uncontrolled"],
            tags=["propagation", "blast", "cascade", "behavior"])

        ns.define("dampened-propagation",
            "A signal that weakens as it travels, naturally attenuating to zero at sufficient distance",
            level=Level.PATTERN,
            examples=["stone in pond: ripples spread but fade (dampened propagation)", "sound: audible nearby, inaudible far away", "fleet: pheromone signal fades with distance and time (dampened propagation)"],
            bridges=["decay", "attenuation", "finite-reach", "distance"],
            tags=["propagation", "dampened", "decay", "pattern"])

        ns.define("amplification-loop",
            "A feedback path that takes output and feeds it back as input, amplifying the original signal",
            level=Level.PATTERN,
            examples=["microphone + speaker feedback loop → screech", "fleet: emotion contagion: one agent → neighbor → fleet (amplification)", "stock market panic: selling triggers more selling (amplification loop)"],
            bridges=["feedback", "positive-feedback", "exponential", "amplification"],
            tags=["propagation", "amplification", "feedback", "pattern"])

        ns.define("signal-grounding",
            "Connecting an abstract signal to a concrete reality so it stops being noise and starts being information",
            level=Level.DOMAIN,
            examples=["electrical ground: gives voltage signals meaning relative to 0V", "fleet: shared vocabulary grounds messages to common meanings", "conversation: shared context grounds words to mutual understanding"],
            bridges=["grounding", "reference", "meaning", "shared"],
            tags=["propagation", "grounding", "reference", "domain"])

    def _load_lifecycle(self):
        ns = self.add_namespace("lifecycle",
            "How entities are born, grow, decline, and are replaced in systems")

        ns.define("birth-threshold",
            "The minimum conditions required for a new entity to be created — the barrier to entry",
            level=Level.CONCRETE,
            examples=["startup: needs problem + solution + capital + team (birth threshold)", "fleet agent: needs energy + task + equipment (birth threshold)", "new species: needs niche + resources + reproduction (birth threshold)"],
            bridges=["spawn", "creation", "minimum-requirements", "threshold"],
            tags=["lifecycle", "birth", "threshold", "concrete"])

        ns.define("decline-curve",
            "The trajectory of decreasing performance as a system ages or faces deteriorating conditions",
            level=Level.DOMAIN,
            examples=["oil well: fast initial decline, slow taper (decline curve)", "human: gradual performance decline after peak", "fleet agent: fitness decline tracked by energy system, triggers apoptosis at threshold"],
            bridges=["decline", "aging", "trajectory", "fitness"],
            tags=["lifecycle", "decline", "curve", "domain"])

        ns.define("succession",
            "The orderly transfer of responsibility from one entity to its replacement",
            level=Level.PATTERN,
            examples=["CEO succession: handoff period, knowledge transfer", "fleet: dying agent transfers state to successor via snapshots", "presidential transition: 2-month handoff period"],
            bridges=["replacement", "handoff", "knowledge-transfer", "orderly"],
            tags=["lifecycle", "succession", "handoff", "pattern"])

        ns.define("niches-and-clines",
            "The spatial distribution of fitness across an environment — where different strategies thrive",
            level=Level.DOMAIN,
            examples=["mountain: different species at different elevations (niches along gradient)", "fleet: complex agents for hard tasks, lightweight agents for simple tasks (niches)", "market: enterprise customers (high-touch niche) vs consumers (self-service niche)"],
            bridges=["adaptation", "environment", "fitness-landscape", "specialization"],
            tags=["lifecycle", "niche", "gradient", "domain"])

    def _load_incentives(self):
        ns = self.add_namespace("incentives",
            "How reward structures shape behavior — alignment and misalignment of incentives")

        ns.define("perverse-incentive",
            "A reward structure that encourages the opposite of the desired behavior",
            level=Level.BEHAVIOR,
            examples=["Wells Fargo: account-opening quotas → fake accounts (perverse)", "teachers: test score eval → teaching to test (perverse)", "fleet: speed reward → skip deliberation (perverse incentive)"],
            bridges=["misalignment", "gaming", "metric", "behavior"],
            tags=["incentive", "perverse", "misalignment", "behavior"])

        ns.define("skin-in-the-game",
            "When decision-makers bear the consequences of their decisions — alignment through shared risk",
            level=Level.PATTERN,
            examples=["chef eats own cooking (skin in the game)", "founder invests own money (skin in the game)", "fleet agent spends own energy on deliberation (skin in the game)"],
            bridges=["alignment", "consequence", "risk-sharing", "accountability"],
            tags=["incentive", "skin-in-game", "alignment", "pattern"])

        ns.define("goodhart-law",
            "When a measure becomes a target, it ceases to be a good measure",
            level=Level.META,
            examples=["money supply: useful measure, useless target", "fleet: fitness score as sole target → agents game the score", "test scores: useful measure, useless when teachers target them"],
            bridges=["metric", "gaming", "target", "measure"],
            tags=["incentive", "goodhart", "measure", "meta"])


    def _load_action_verbs(self):
        ns = self.add_namespace("action-verbs",
            "High-compression verbs that each encode a complete multi-step operational pattern")

        ns.define("vet",
            "Rapidly assess whether a candidate is viable before investing in full evaluation",
            level=Level.CONCRETE,
            examples=["resume scan: 30 seconds, eliminate 90% before full interview (vet)", "fleet: quick confidence check on proposal before full deliberation (vet)", "startup idea: quick market check before building MVP (vet)"],
            bridges=["pre-filter", "rapid-assessment", "triage", "efficiency"],
            tags=["verb", "vet", "assess", "concrete"])

        ns.define("triage",
            "Rapidly categorize incoming items by urgency and severity to allocate attention where it's most needed",
            level=Level.CONCRETE,
            examples=["ER: chest pain → immediate, cut finger → wait (triage)", "fleet: incoming tasks ranked by priority, energy allocated to highest (triage)", "inbox: flag urgent, archive spam, respond to important (triage)"],
            bridges=["prioritize", "rank", "urgent", "allocate"],
            tags=["verb", "triage", "prioritize", "concrete"])

        ns.define("shard",
            "Split a large entity into smaller, independently manageable pieces that can be processed in parallel",
            level=Level.CONCRETE,
            examples=["database: 100GB → 10 × 10GB shards (parallel processing)", "fleet: 1000 readings → 10 × 100 reading sub-tasks (parallel agents)", "map: reduce input into chunks for parallel map workers"],
            bridges=["split", "parallel", "partition", "scale"],
            tags=["verb", "shard", "split", "concrete"])

        ns.define("stitch",
            "Combine multiple partial results into a coherent whole — the inverse of shard",
            level=Level.CONCRETE,
            examples=["sharded results → combined analysis (stitch)", "fleet: multiple sensor readings → fused coherent picture (stitch)", "quilt: separate patches → one blanket (stitch)"],
            bridges=["fusion", "combine", "merge", "consistency"],
            antonyms=["shard"],
            tags=["verb", "stitch", "combine", "concrete"])

        ns.define("bench",
            "Establish a performance baseline by measuring a known workload under controlled conditions",
            level=Level.CONCRETE,
            examples=["benchmark: run workload, measure time (bench)", "fleet: measure task performance before/after gene change (bench)", "athlete: timed run before/after training program (bench)"],
            bridges=["baseline", "measure", "performance", "comparison"],
            tags=["verb", "bench", "measure", "concrete"])

        ns.define("mock",
            "Replace a real component with a fake one that has the same interface but controlled behavior",
            level=Level.CONCRETE,
            examples=["test payment without real charges (mock gateway)", "test fleet without real network (mock A2A)", "fleet: mock sensor trait with fixed values for testing"],
            bridges=["testing", "isolation", "fake", "interface"],
            tags=["verb", "mock", "fake", "concrete"])

        ns.define("hardwire",
            "Permanently connect two components with a direct, dedicated channel — no routing, no discovery, no overhead",
            level=Level.CONCRETE,
            examples=["dedicated line between two offices instead of phone network (hardwire)", "fleet: direct function call between perception and deliberation (hardwire)", "memory bus: direct CPU-RAM connection instead of network storage (hardwire)"],
            bridges=["direct-connection", "low-latency", "dedicated", "bypass"],
            antonyms=["route"],
            tags=["verb", "hardwire", "direct", "concrete"])

        ns.define("throttle",
            "Deliberately slow down a process to prevent overload, resource exhaustion, or rate limit violations",
            level=Level.CONCRETE,
            examples=["API: self-throttle to 100/s to avoid 429 errors", "fleet: throttle outbound messages to stay under rate limits", "highway: speed limit as throttle (prevents accidents from over-speed)"],
            bridges=["rate-limit", "protect", "slow-down", "self-imposed"],
            tags=["verb", "throttle", "slow", "concrete"])

        ns.define("bridge",
            "Create a translation layer between two incompatible systems, enabling them to communicate without modifying either",
            level=Level.CONCRETE,
            examples=["database adapter: SQL → NoSQL translation (bridge)", "fleet: physical sensors ↔ agent decisions (bridge)", "interpreter: translates between two languages (bridge)"],
            bridges=["translation", "adapter", "intermediary", "compatibility"],
            tags=["verb", "bridge", "translate", "concrete"])

        ns.define("harden",
            "Add defensive measures to a system to make it resistant to attacks, failures, and unexpected inputs",
            level=Level.PATTERN,
            examples=["web app: add input validation, rate limit, auth, CSP (harden)", "fleet: add sandbox, compliance, RBAC to agent system (harden)", "house: add locks, alarm, reinforced doors (harden)"],
            bridges=["defense", "production-readiness", "resilience", "robustness"],
            tags=["verb", "harden", "defend", "pattern"])

        ns.define("stress-test",
            "Push a system beyond its normal operating range to find its breaking point and verify graceful degradation",
            level=Level.CONCRETE,
            examples=["load test: 1000 concurrent users (stress-test)", "chaos test: kill random services (stress-test)", "fleet: chaos monkey randomly fails agents (stress-test)"],
            bridges=["testing", "extreme", "failure", "verification"],
            tags=["verb", "stress-test", "push", "concrete"])

        ns.define("garden",
            "Tend to a system regularly — prune dead parts, water growing parts, remove weeds — continuous maintenance",
            level=Level.BEHAVIOR,
            examples=["garden: prune, water, weed, rotate (continuous maintenance)", "fleet: quarantine bad genes, promote fit ones (gene pool gardening)", "codebase: remove dead code, refactor, update deps (codebase gardening)"],
            bridges=["maintenance", "continuous", "prune", "tend"],
            tags=["verb", "garden", "maintain", "behavior"])

        ns.define("orchestrate",
            "Coordinate multiple independent components to perform a unified workflow without centralizing control",
            level=Level.PATTERN,
            examples=["orchestra: conductor coordinates, musicians play independently", "fleet: captain coordinates, agents execute independently (orchestrate)", "devops: pipeline tool coordinates build, test, deploy stages (orchestrate)"],
            bridges=["coordinate", "conduct", "workflow", "independent"],
            tags=["verb", "orchestrate", "coordinate", "pattern"])

        ns.define("ferment",
            "Allow a system to develop complexity through autonomous internal processes over time, with minimal intervention",
            level=Level.META,
            examples=["wine: grapes + yeast + time = complex flavors (ferment)", "fleet: genes evolve over time into strategies (ferment)", "starter: flour + water + time = sourdough culture (ferment)"],
            bridges=["evolution", "autonomous", "patience", "emergence"],
            tags=["verb", "ferment", "evolve", "meta"])

        ns.define("calibrate",
            "Adjust a measurement instrument to match a known reference standard, ensuring accuracy",
            level=Level.CONCRETE,
            examples=["scale: reads 101g for 100g weight → adjust to 100g (calibrate)", "fleet: self-model adjusts self-assessment to match actual performance (calibrate)", "thermometer: adjust to match known reference temperature (calibrate)"],
            bridges=["accuracy", "reference", "adjustment", "measurement"],
            tags=["verb", "calibrate", "adjust", "concrete"])

        ns.define("route",
            "Determine the optimal path for information or resources to travel from source to destination",
            level=Level.CONCRETE,
            examples=["GPS: fastest route considering current traffic (route)", "internet: shortest AS path (route)", "fleet: lowest-latency message path (route)"],
            bridges=["pathfinding", "optimal", "dynamic", "network"],
            tags=["verb", "route", "path", "concrete"])

        ns.define("thaw",
            "Gradually restore a frozen or quarantined component to active service, testing at each step",
            level=Level.PATTERN,
            examples=["quarantined employee: reduced duties → full duties (thaw)", "fleet: sandbox → limited production → full (thaw quarantined gene)", "frozen code: unit tests → staging → production (thaw)"],
            bridges=["quarantine", "restore", "incremental", "verification"],
            antonyms=["quarantine"],
            tags=["verb", "thaw", "restore", "pattern"])

        ns.define("snapshot",
            "Capture the complete state of a system at a moment in time, enabling exact restoration later",
            level=Level.CONCRETE,
            examples=["VM snapshot: exact state preservation for rollback", "fleet: agent state captured for failure recovery (snapshot)", "photo: visual snapshot of a moment in time"],
            bridges=["backup", "state", "restore", "point-in-time"],
            tags=["verb", "snapshot", "capture", "concrete"])

        ns.define("broadcast",
            "Send a message to all receivers simultaneously without targeting specific recipients",
            level=Level.CONCRETE,
            examples=["radio: one transmitter, all receivers hear (broadcast)", "fleet: one message, all agents receive (broadcast)", "emergency alert: one signal, all phones receive (broadcast)"],
            bridges=["multicast", "inundate", "announce", "all-receivers"],
            tags=["verb", "broadcast", "send-all", "concrete"])

        ns.define("tunnel",
            "Create a direct path through an intermediate layer that would otherwise block or transform the communication",
            level=Level.CONCRETE,
            examples=["VPN: encrypted path through internet (tunnel)", "SSH tunnel: direct connection through bastion host (tunnel)", "fleet: direct agent-to-agent connection bypassing mesh routing (tunnel)"],
            bridges=["bypass", "direct", "encrypted", "intermediary"],
            antonyms=["route"],
            tags=["verb", "tunnel", "bypass", "concrete"])

        ns.define("pollinate",
            "Transfer useful patterns or data from one part of the system to another, enabling cross-pollination of ideas",
            level=Level.PATTERN,
            examples=["bees: transfer pollen between plants (pollinate)", "fleet: gene sharing between agent pools (pollinate)", "team rotation: ideas transfer between teams (pollinate)"],
            bridges=["cross-pollination", "diversity", "transfer", "sharing"],
            tags=["verb", "pollinate", "transfer", "pattern"])

        ns.define("graft",
            "Attach a foreign component to an existing system, creating a hybrid that combines both capabilities",
            level=Level.PATTERN,
            examples=["apple branch grafted onto pear tree (hybrid fruit tree)", "fleet: attention capability grafted onto swarm agent (hybrid agent)", "organization: acquired team integrated into existing structure (graft)"],
            bridges=["hybrid", "attach", "integrate", "combine"],
            tags=["verb", "graft", "attach", "pattern"])

        ns.define("reconcile",
            "Resolve differences between two or more sources of truth into a single consistent state",
            level=Level.CONCRETE,
            examples=["bank statement vs checkbook: find discrepancy, resolve (reconcile)", "fleet: agents exchange evidence, converge on agreed trust score (reconcile)", "git merge: resolve conflicts between branches (reconcile)"],
            bridges=["consistency", "merge", "conflict-resolution", "truth"],
            tags=["verb", "reconcile", "resolve", "concrete"])

        ns.define("fortify",
            "Add defensive layers specifically targeting known or anticipated attack vectors",
            level=Level.PATTERN,
            examples=["castle: add walls, moats, guards against specific threats (fortify)", "fleet: add compliance rules against specific attack vectors (fortify)", "network: add specific firewall rules for known attack patterns (fortify)"],
            bridges=["harden", "defense", "specific", "threat"],
            tags=["verb", "fortify", "defend", "pattern"])


    def _load_action_verbs_2(self):
        ns = self.add_namespace("action-verbs-2",
            "More high-compression operational verbs for the fleet vocabulary")

        ns.define("drain",
            "Gradually consume or remove a finite resource until it reaches zero or a critical low",
            level=Level.CONCRETE,
            examples=["phone battery: drains from 100% to 0% over day (drain)", "fleet: energy budget drains during sustained deliberation", "bank account: drains as expenses accumulate (drain)"],
            bridges=["depletion", "consumption", "budget", "resource"],
            tags=["verb", "drain", "deplete", "concrete"])

        ns.define("prime",
            "Pre-load a system with initial data or state so it's immediately useful, not cold-starting from zero",
            level=Level.CONCRETE,
            examples=["water pump: fill with water before starting (prime)", "cache: pre-populate with hot data (prime)", "fleet: preload agent with known object positions (prime)"],
            bridges=["warm-start", "pre-load", "initial-state", "latency"],
            tags=["verb", "prime", "pre-load", "concrete"])

        ns.define("relay",
            "Pass a message or task through a chain of intermediaries, each forwarding to the next hop",
            level=Level.CONCRETE,
            examples=["Olympic torch: runner to runner (relay)", "email: server to server to destination (relay)", "fleet: A → B → C message routing (relay)"],
            bridges=["routing", "forward", "chain", "hop"],
            tags=["verb", "relay", "forward", "concrete"])

        ns.define("fuse",
            "Merge two or more signals into one combined signal that preserves the essential information of each",
            level=Level.CONCRETE,
            examples=["GPS + accelerometer → fused position (sensor fusion)", "fleet: multiple confidence sources → combined assessment (fusion)", "metal alloy: iron + carbon → steel (material fusion)"],
            bridges=["combine", "merge", "information", "confidence"],
            tags=["verb", "fuse", "merge", "concrete"])

        ns.define("steer",
            "Apply a small directional influence that gradually changes the system's trajectory without forcing it",
            level=Level.PATTERN,
            examples=["rudder: redirects ship momentum without forcing (steer)", "fleet: evidence weighting steers deliberation toward better proposals", "parent: gentle guidance without commanding (steer)"],
            bridges=["influence", "redirect", "subtle", "momentum"],
            tags=["verb", "steer", "influence", "pattern"])

        ns.define("pinpoint",
            "Identify the exact location or cause of an issue with maximum precision, not just the general area",
            level=Level.CONCRETE,
            examples=["not 'server slow' but '/api/users 2300ms, unindexed query' (pinpoint)", "fleet: trace decision to exact proposal and agent (pinpoint)", "doctor: not 'you're sick' but 'strep throat, bacteria X, antibiotic Y' (pinpoint)"],
            bridges=["precision", "diagnosis", "root-cause", "exact"],
            tags=["verb", "pinpoint", "precise", "concrete"])

        ns.define("absorb",
            "Take in a smaller entity into a larger one, integrating its capabilities without maintaining separate identity",
            level=Level.CONCRETE,
            examples=["acquisition: startup tech absorbed into main product", "fleet: agent capabilities absorbed into fleet shared resources", "ocean absorbs river: river ceases to exist, water persists"],
            bridges=["acquire", "assimilate", "integrate", "merge"],
            tags=["verb", "absorb", "assimilate", "concrete"])

        ns.define("prune",
            "Remove dead, redundant, or underperforming components to improve overall system health",
            level=Level.PATTERN,
            examples=["prune dead tree branches: energy to living branches", "prune dead code: easier maintenance", "fleet: quarantine and remove low-fitness genes (prune)"],
            bridges=["remove", "maintenance", "fitness", "health"],
            tags=["verb", "prune", "remove", "pattern"])

        ns.define("graft-replace",
            "Replace a component with a new version by cutting the old one out and grafting the new one in its place",
            level=Level.CONCRETE,
            examples=["heart valve replacement: remove old, graft new", "fleet: swap crate version in Cargo.toml (graft-replace)", "organ transplant: remove old organ, graft new one"],
            bridges=["replace", "upgrade", "surgical", "version"],
            tags=["verb", "graft-replace", "swap", "concrete"])

        ns.define("partition",
            "Divide a system into isolated segments that can operate independently, limiting blast radius",
            level=Level.CONCRETE,
            examples=["ship watertight compartments: one floods, others dry (partition)", "fleet: bulkhead isolation between agents (partition)", "database: partitioned tables for independent scaling (partition)"],
            bridges=["isolation", "bulkhead", "segment", "blast-radius"],
            tags=["verb", "partition", "isolate", "concrete"])

        ns.define("nominate",
            "Designate a specific agent for a specific role based on capability matching, not random assignment",
            level=Level.CONCRETE,
            examples=["team lead nomination: best person for the role", "fleet: best_available() nominates highest-fitness agent for task", "award nomination: best candidate selected based on criteria"],
            bridges=["assign", "capability", "fitness", "selection"],
            tags=["verb", "nominate", "assign", "concrete"])

        ns.define("deputize",
            "Temporarily grant an agent authority or capabilities beyond its normal scope for a specific purpose",
            level=Level.PATTERN,
            examples=["sheriff deputizes civilian for emergency (temporary authority)", "fleet: agent granted elevated trust for emergency response (deputize)", "employee given acting manager role (deputize)"],
            bridges=["temporary", "authority", "elevation", "emergency"],
            tags=["verb", "deputize", "temporary", "pattern"])

        ns.define("delegate",
            "Assign a task to a subordinate agent with the authority to complete it autonomously",
            level=Level.PATTERN,
            examples=["manager delegates report to employee (what, not how)", "fleet: captain delegates task to agent (mission defined, execution autonomous)", "parent delegates chore to child (task defined, method child's choice)"],
            bridges=["assign", "autonomous", "authority", "responsibility"],
            antonyms=["micromanage"],
            tags=["verb", "delegate", "assign", "pattern"])

        ns.define("quiesce",
            "Gracefully drain all active operations and enter a quiet state without abrupt termination",
            level=Level.CONCRETE,
            examples=["database: finish transactions, stop accepting new, then shut down (quiesce)", "fleet: finish task, save state, terminate cleanly (quiesce)", "factory: finish current production run, then shut down (quiesce)"],
            bridges=["graceful-shutdown", "drain", "clean", "no-abort"],
            antonyms=["kill", "abort"],
            tags=["verb", "quiesce", "shutdown", "concrete"])

        ns.define("silo",
            "Isolate a component or team so completely that no information flows in or out — intentional separation",
            level=Level.PATTERN,
            examples=["security: classified information in silo (intentional)", "fleet: sandbox isolates untrusted agent (intentional silo)", "development: feature branch isolated from main (intentional silo)"],
            bridges=["isolation", "sandbox", "containment", "intentional"],
            tags=["verb", "silo", "isolate", "pattern"])

        ns.define("referee",
            "Observe interactions between agents and enforce rules without participating in the interaction",
            level=Level.PATTERN,
            examples=["sports referee: watches, enforces rules, doesn't play", "fleet: compliance observes behavior, enforces policy, doesn't participate in tasks", "judge: observes arguments, enforces procedure, doesn't advocate"],
            bridges=["observer", "enforce", "impartial", "rule"],
            tags=["verb", "referee", "enforce", "pattern"])

        ns.define("nurture",
            "Invest resources in a growing entity with the expectation that it will eventually become self-sustaining",
            level=Level.PATTERN,
            examples=["incubator: funds and mentors startups until self-sustaining (nurture)", "fleet: captain assigns easy tasks to new agent, then harder (nurture)", "parent: cares for child until independent (nurture)"],
            bridges=["grow", "invest", "incubate", "eventual-independence"],
            tags=["verb", "nurture", "grow", "pattern"])

        ns.define("excavate",
            "Dig through accumulated layers to find buried but valuable information or patterns",
            level=Level.CONCRETE,
            examples=["archaeology: dig through dirt layers to find artifacts (excavate)", "fleet: dig through state snapshots to trace evolution (excavate)", "journalism: dig through documents to find buried story (excavate)"],
            bridges=["dig", "layers", "history", "discover"],
            tags=["verb", "excavate", "dig", "concrete"])

        ns.define("synchronize",
            "Bring two or more systems into consistent state by reconciling differences and resolving conflicts",
            level=Level.CONCRETE,
            examples=["clock sync: adjust two clocks to same time (synchronize)", "git: fetch + merge to sync local and remote (synchronize)", "fleet: CRDT merge brings two agent states into consistency (synchronize)"],
            bridges=["consistency", "merge", "reconcile", "conflict"],
            tags=["verb", "synchronize", "consistency", "concrete"])

        ns.define("templify",
            "Convert a specific solution into a reusable template that works across multiple contexts",
            level=Level.PATTERN,
            examples=["project A solution → reusable template for B, C, D (templify)", "fleet: specific agent config → reusable config template (templify)", "email template: specific email → template with variables (templify)"],
            bridges=["template", "reusable", "generalize", "extract"],
            tags=["verb", "templify", "template", "pattern"])

        ns.define("vaccinate",
            "Expose a system to a weakened version of a threat to build immunity against future stronger attacks",
            level=Level.PATTERN,
            examples=["vaccine: weakened virus builds immunity (vaccinate)", "fleet: chaos monkey injects small failures to build resilience (vaccinate)", "flu shot: exposure builds antibodies before real flu arrives"],
            bridges=["immunity", "exposure", "proactive", "resilience"],
            tags=["verb", "vaccinate", "immunize", "pattern"])


    def _load_final_verbs(self):
        ns = self.add_namespace("fleet-verbs",
            "The last set of operational verbs completing the 600-term vocabulary")

        ns.define("scaffold",
            "Build a temporary framework that supports construction and is removed when no longer needed",
            level=Level.PATTERN,
            examples=["building scaffold: temporary support during construction", "fleet: captain provides temporary task structure for new agents", "training wheels: temporary support until balance learned (scaffold)"],
            bridges=["temporary", "support", "construction", "remove"],
            tags=["verb", "scaffold", "temporary", "pattern"])

        ns.define("recon",
            "Perform a quick, minimal-cost exploration of unknown territory to gather information before committing resources",
            level=Level.CONCRETE,
            examples=["military scout team: gather intel before full deployment (recon)", "fleet: scout-mode exploration before committing full resources (recon)", "house hunting: drive by neighborhood before scheduling tour (recon)"],
            bridges=["explore", "scout", "cheap", "information"],
            tags=["verb", "recon", "scout", "concrete"])

        ns.define("siege",
            "Sustained pressure on a target that gradually depletes its resources until it yields or collapses",
            level=Level.PATTERN,
            examples=["castle siege: surround, cut supplies, wait for depletion", "fleet: sustained adversarial probing depletes agent energy over time (siege)", "legal: persistent lawsuits drain defendant resources (siege)"],
            bridges=["sustained", "pressure", "deplete", "gradual"],
            tags=["verb", "siege", "sustained", "pattern"])

        ns.define("reconstitute",
            "Rebuild a system from its preserved components after a disruption or failure",
            level=Level.CONCRETE,
            examples=["freeze-dried: add water → restored food (reconstitute)", "fleet: load snapshot → restore agent state (reconstitute)", "jigsaw puzzle: reassemble from pieces (reconstitute)"],
            bridges=["restore", "rebuild", "snapshot", "recovery"],
            tags=["verb", "reconstitute", "restore", "concrete"])

        ns.define("ping",
            "Send a minimal signal to verify that a system is alive and responsive, measuring round-trip time",
            level=Level.CONCRETE,
            examples=["network ping: ICMP echo, measure latency (ping)", "fleet: health check message to all agents (ping)", "sonar: pulse, listen for echo (ping)"],
            bridges=["health-check", "latency", "minimal", "alive"],
            tags=["verb", "ping", "check", "concrete"])

        ns.define("scuttle",
            "Deliberately and safely destroy a system to prevent it from falling into the wrong hands or causing harm",
            level=Level.PATTERN,
            examples=["scuttle ship: sink on own terms, not enemy's", "fleet: apoptosis shuts down failing agent on own terms (scuttle)", "burn documents: destroy on your terms before enemy captures them"],
            bridges=["apoptosis", "destroy", "controlled", "intentional"],
            tags=["verb", "scuttle", "destroy", "pattern"])

        ns.define("ratify",
            "Formally approve a decision or action after it has been proposed and reviewed by relevant stakeholders",
            level=Level.CONCRETE,
            examples=["treaty: negotiate → review → ratify (formal approval)", "fleet: deliberation → evidence → threshold → ratify (actionable decision)", "constitution: draft → debate → ratify (law)"],
            bridges=["approve", "consensus", "formal", "decision"],
            tags=["verb", "ratify", "approve", "concrete"])

        ns.define("embargo",
            "Restrict the flow of resources or information to or from a specific entity as a penalty or protective measure",
            level=Level.PATTERN,
            examples=["trade embargo: restrict goods to/from country", "fleet: restrict misbehaving agent's communication (embargo)", "library: embargo book from being checked out (restricted access)"],
            bridges=["restrict", "sanction", "targeted", "penalty"],
            tags=["verb", "embargo", "restrict", "pattern"])

        ns.define("exfiltrate",
            "Extract data or resources from a system, potentially covertly, without authorization",
            level=Level.CONCRETE,
            examples=["data breach: attacker extracts user records (exfiltrate)", "insider: copies proprietary code to USB (exfiltrate)", "fleet membrane: blocks sensitive state transmission (anti-exfiltration)"],
            bridges=["breach", "unauthorized", "extract", "security"],
            tags=["verb", "exfiltrate", "breach", "concrete"])

        ns.define("infiltrate",
            "Gain access to a system by blending in with legitimate traffic or credentials",
            level=Level.CONCRETE,
            examples=["stolen credentials: attacker looks legitimate (infiltrate)", "trojan horse: malicious code hidden in legitimate package (infiltrate)", "fleet: behavioral monitoring detects anomalous agents (anti-infiltration)"],
            bridges=["breach", "disguise", "credential-theft", "detection"],
            tags=["verb", "infiltrate", "breach", "concrete"])

        ns.define("decommission",
            "Formally retire a system from active service, removing its access and redirecting its responsibilities",
            level=Level.CONCRETE,
            examples=["navy ship: retire from active fleet (decommission)", "fleet agent: remove from mesh, reassign tasks, save state (decommission)", "software: EOL product, redirect users, archive code (decommission)"],
            bridges=["retire", "graceful", "reassign", "formal"],
            tags=["verb", "decommission", "retire", "concrete"])

        ns.define("demote",
            "Reduce an agent's authority, capabilities, or priority level in response to poor performance or policy violation",
            level=Level.PATTERN,
            examples=["officer demoted for misconduct (reduced rank)", "fleet agent: reduced energy budget after compliance violation (demote)", "employee: manager → individual contributor (demote)"],
            bridges=["punish", "reduce", "authority", "warning"],
            tags=["verb", "demote", "reduce", "pattern"])

        ns.define("promote",
            "Increase an agent's authority, capabilities, or priority level in recognition of superior performance",
            level=Level.PATTERN,
            examples=["engineer → senior engineer (promoted for competence)", "fleet agent: granted higher permissions for consistent high fitness (promote)", "officer: promoted for exemplary service (promote)"],
            bridges=["reward", "elevate", "authority", "merit"],
            antonyms=["demote"],
            tags=["verb", "promote", "elevate", "pattern"])

        ns.define("evict",
            "Forcefully remove an entity from a shared resource or space because it's violating rules or overstaying",
            level=Level.CONCRETE,
            examples=["landlord evicts tenant for non-payment (evict)", "cache evicts LRU entry when full (evict)", "fleet: cache removes least-recently-used entries (evict)"],
            bridges=["remove", "forceful", "limit", "enforce"],
            tags=["verb", "evict", "remove", "concrete"])

        ns.define("provision",
            "Allocate and configure resources needed by a new or growing entity before it needs them",
            level=Level.CONCRETE,
            examples=["cloud: create VMs, networks, storage before app starts (provision)", "fleet: allocate energy, register mesh, assign equipment before first task (provision)", "event: set up venue, AV, catering before attendees arrive (provision)"],
            bridges=["allocate", "configure", "ahead-of-demand", "setup"],
            tags=["verb", "provision", "allocate", "concrete"])

        ns.define("ring-fence",
            "Isolate a subset of resources or operations with strict boundaries to prevent contamination or cross-subsidization",
            level=Level.PATTERN,
            examples=["bank: retail ring-fenced from investment banking", "fleet: sandbox ring-fences untrusted code", "budget: task-specific energy allocation that can't be used elsewhere (ring-fence)"],
            bridges=["isolate", "boundary", "protect", "guarantee"],
            tags=["verb", "ring-fence", "isolate", "pattern"])

        ns.define("siphon",
            "Divert resources or information from a main flow into a secondary channel, usually covertly or gradually",
            level=Level.BEHAVIOR,
            examples=["siphon gas from tank: slow, gradual diversion", "fleet: agent slowly diverts shared energy for personal use (siphon)", "embezzlement: small, gradual theft from accounts (siphon)"],
            bridges=["divert", "gradual", "covert", "drain"],
            tags=["verb", "siphon", "divert", "behavior"])

        ns.define("bench-press",
            "Stress-test a specific capability under controlled heavy load to measure its maximum capacity",
            level=Level.CONCRETE,
            examples=["bench press: maximum weight lifted once (capacity)", "fleet: maximum message throughput before degradation (bench-press)", "bridge: maximum load before structural failure (bench-press)"],
            bridges=["capacity-test", "maximum", "stress", "measure"],
            tags=["verb", "bench-press", "capacity", "concrete"])

        ns.define("cross-pollinate",
            "Transfer successful patterns from one domain or team to another where they haven't been tried",
            level=Level.PATTERN,
            examples=["lean manufacturing → agile software (cross-pollination)", "fleet: navigation gene applied to data processing (cross-pollinate)", "baseball analytics → basketball analytics (cross-pollination)"],
            bridges=["transfer", "domain-transfer", "adapt", "innovation"],
            tags=["verb", "cross-pollinate", "transfer", "pattern"])


    def _load_github_native(self):
        ns = self.add_namespace("github-native",
            "Vocabulary for git-native agent operations — where the repo IS the nervous system")

        ns.define("capability-diff",
            "The measurable delta between an agent's capabilities at two points in time, as captured by git diffs",
            level=Level.CONCRETE,
            examples=["git diff that adds navigation module = capability-diff: +navigation", "agent commit log as resume of capability acquisition", "PR review: 'this capability-diff adds threat detection'"],
            bridges=["git", "capability", "evolution", "measurement"],
            tags=["github", "capability", "evolution", "concrete"])

        ns.define("agentic-diff",
            "A diff format that captures semantic intent alongside code changes — WHY the change, not just WHAT changed",
            level=Level.CONCRETE,
            examples=["PR with agentic-diff: shows reasoning, tradeoffs, confidence change", "fleet agent commit: includes deliberation summary with code change", "code review: 'the agentic-diff shows the agent considered 3 alternatives'"],
            bridges=["diff", "intent", "provenance", "deliberation"],
            tags=["github", "diff", "intent", "concrete"])

        ns.define("branch-agon",
            "An algorithm that strategically generates competing branches as adversarial challenges to the production state",
            level=Level.PATTERN,
            examples=["spawn adversarial branch that optimizes for latency vs current accuracy branch", "fleet: challenger agent competes against incumbent for task performance", "A/B testing taken to its logical extreme: every branch is a challenger"],
            bridges=["competition", "branch", "adversarial", "evolution"],
            tags=["github", "branch", "competition", "pattern"])

        ns.define("commit-prophecy",
            "Predicting the future trajectory of a codebase by analyzing patterns in its commit history",
            level=Level.DOMAIN,
            examples=["spike in 'fix' commits = growing technical debt (commit prophecy)", "fleet: agent commit patterns predict capability drift", "startup: commit frequency decline = project stalling"],
            bridges=["prediction", "git", "pattern", "trajectory"],
            tags=["github", "commit", "prediction", "domain"])

        ns.define("code-pheromone",
            "Metadata attached to code that probabilistically attracts future modifications and attention",
            level=Level.BEHAVIOR,
            examples=["successful PR leaves pheromone on modified files → future PRs modify same area", "fleet: successful mutation deposits pheromone → attracts future mutations", "hotspot detection: files with most pheromones = areas of active evolution"],
            bridges=["stigmergy", "attraction", "attention", "probability"],
            tags=["github", "pheromone", "stigmergy", "behavior"])

        ns.define("repo-synapse",
            "A git hook or webhook that triggers biological-style learning when specific repo events occur",
            level=Level.PATTERN,
            examples=["on PR merge: trigger gene pool update (repo-synapse)", "on test failure: trigger agent quarantine (repo-synapse)", "on revert: trigger fitness decrease (repo-synapse)"],
            bridges=["git", "hook", "learning", "nervous-system"],
            tags=["github", "synapse", "hook", "pattern"])

        ns.define("vessel-gem",
            "A packaged agent snapshot — pre-trained, tested, and ready to plug into any compatible fleet",
            level=Level.CONCRETE,
            examples=["install navigation gem → agent gains navigation capability", "fleet marketplace: browse vessel-gems by capability rating", "cuda-genepool: genes distributed as vessel-gems"],
            bridges=["package", "capability", "marketplace", "plug-in"],
            tags=["github", "gem", "package", "concrete"])

        ns.define("semantic-fork",
            "A fork that translates the conceptual model of the original repo into a different paradigm or domain",
            level=Level.META,
            examples=["fleet coordination → immune system (semantic fork)", "game theory → market dynamics (semantic fork)", "git branching → biological gene expression (semantic fork)"],
            bridges=["fork", "translation", "cross-domain", "abstraction"],
            tags=["github", "fork", "concept", "meta"])

        ns.define("seam-merge",
            "Merging agents or codebases from entirely different fleets, organizations, or paradigms",
            level=Level.PATTERN,
            examples=["merge robotics agent with data analysis agent (seam-merge)", "corporate acquisition: integrate two engineering cultures (seam-merge)", "open source + proprietary: merge community code with internal code (seam-merge)"],
            bridges=["merge", "integration", "interface", "cross-fleet"],
            tags=["github", "merge", "integration", "pattern"])

        ns.define("hot-branch",
            "A live production branch undergoing rapid mutation and evolution without stabilizing into a release",
            level=Level.BEHAVIOR,
            examples=["production deployment directly from hot branch (continuous evolution)", "fleet agent: modifies own code in production (hot-branch behavior)", "live coding: changes go live immediately"],
            bridges=["production", "mutation", "continuous", "risk"],
            tags=["github", "branch", "live", "behavior"])

        ns.define("regression-bounty",
            "A reward system for identifying capability regressions — when an agent gets WORSE at something",
            level=Level.PATTERN,
            examples=["bounty: 'navigation accuracy dropped 5% after commit abc123'", "fleet: regression detection in experience replay (regression-bounty)", "not 'it crashes' but 'it got worse at X'"],
            bridges=["regression", "quality", "monitoring", "incentive"],
            tags=["github", "regression", "bounty", "pattern"])

        ns.define("branch-tributary",
            "An auxiliary branch that continuously feeds knowledge into a main branch without ever merging directly",
            level=Level.PATTERN,
            examples=["research branch feeds insights to production branch (tributary)", "fleet: auxiliary agent shares observations without modifying main agent (tributary)", "advisor role: provide input without being on the execution team"],
            bridges=["branch", "information", "auxiliary", "advisory"],
            tags=["github", "branch", "tributary", "pattern"])

        ns.define("orphan-branch",
            "A branch that has diverged so far from the main branch that it can no longer be merged — but may contain breakthrough innovations",
            level=Level.BEHAVIOR,
            examples=["radical refactor that breaks API compatibility (orphan-branch)", "fleet: novel gene too different to integrate directly (orphan)", "species on isolated island: diverges from mainland (biological orphan)"],
            bridges=["divergence", "innovation", "incompatibility", "novelty"],
            tags=["github", "branch", "orphan", "behavior"])

        ns.define("branch-dendrochronology",
            "Inferring the health and history of a codebase by analyzing the ring patterns of its branching structure",
            level=Level.DOMAIN,
            examples=["wide gaps between merges = stalled development (dendrochronology)", "rapid branch creation = active experimentation", "fleet: branch patterns reveal agent development health"],
            bridges=["history", "branch", "health", "pattern"],
            tags=["github", "branch", "history", "domain"])

    def _load_fleet_biology(self):
        ns = self.add_namespace("fleet-biology",
            "Biological metaphors made operational in fleet computing — where biology IS computing")

        ns.define("neurotransmitter-map",
            "A real-time mapping of agent state to neurochemical signals — dopamine for confidence, serotonin for trust, cortisol for stress",
            level=Level.DOMAIN,
            examples=["agent broadcasts high cortisol → neighbors reduce communication load", "fleet: neurotransmitter map enables non-verbal fleet coordination", "biological: stress hormone cortisol triggers fight-or-flight (neurotransmitter map)"],
            bridges=["neurotransmitter", "state", "communication", "gradient"],
            tags=["biology", "neurotransmitter", "state", "domain"])

        ns.define("silicon-respiration",
            "Measuring the computational energy cost of every agent operation as a metabolite-equivalent expenditure",
            level=Level.DOMAIN,
            examples=["agent deliberating 2.0 ATP per decision = high respiration rate", "fleet: silicon-respiration metrics trigger circadian rest periods", "human: heavy exercise = high oxygen consumption (respiration)"],
            bridges=["energy", "metabolism", "budget", "exhaustion"],
            tags=["biology", "respiration", "energy", "domain"])

        ns.define("membrane-selectivity",
            "A security boundary that selectively allows or blocks information passage based on trust and compatibility, like a biological cell membrane",
            level=Level.PATTERN,
            examples=["cell membrane: passes compatible molecules, blocks toxins", "fleet membrane: passes genes from trusted agents, quarantines unknown", "biological: immune system learns to distinguish self from non-self (membrane-selectivity)"],
            bridges=["security", "membrane", "trust", "selective"],
            tags=["biology", "membrane", "security", "pattern"])

        ns.define("instinct-fire",
            "A pre-deliberative response that executes in milliseconds based on pattern matching, bypassing the deliberation loop entirely",
            level=Level.BEHAVIOR,
            examples=["touch hot stove → hand pulls back before thought (instinct-fire)", "fleet: low battery triggers conservation before deliberation (instinct-fire)", "cuda-reflex: pre-compiled responses to common threat patterns"],
            bridges=["instinct", "reflex", "fast-response", "bypass"],
            tags=["biology", "instinct", "reflex", "behavior"])

        ns.define("gene-quarantine",
            "Isolating a gene (capability pattern) from the shared pool when it exhibits harmful behavior, without deleting it permanently",
            level=Level.CONCRETE,
            examples=["gene causing regression → quarantined, studied, potentially rehabilitated", "fleet: bad capability pattern isolated from gene pool (quarantine)", "biological: pathogen isolated in quarantine zone"],
            bridges=["quarantine", "gene", "safety", "temporary"],
            tags=["biology", "gene", "quarantine", "concrete"])

        ns.define("epigenetic-memory",
            "Experience-dependent modifications to agent behavior that don't change the underlying genes but alter their expression",
            level=Level.DOMAIN,
            examples=["twin A stress → epigenetic mark on cortisol gene → different stress response", "fleet: repeated hostile encounters → stronger threat instincts (epigenetic memory)", "diet changes gene expression without changing DNA (epigenetics)"],
            bridges=["epigenetics", "memory", "experience", "expression"],
            tags=["biology", "epigenetic", "memory", "domain"])

        ns.define("cascading-emovance",
            "Emotional signals propagating through a fleet as color-coded priority changes — GREEN means safe, RED means halt",
            level=Level.BEHAVIOR,
            examples=["one agent detects threat → RED → neighbors increase vigilance → YELLOW → cascade", "fleet: emotion contagion spreads urgency without central command", "crowd panic: one person runs → everyone runs (cascading-emovance)"],
            bridges=["emotion", "cascade", "contagion", "color-code"],
            tags=["biology", "emotion", "cascade", "behavior"])

        ns.define("soma-death",
            "Graceful agent termination that recycles its resources back into the fleet — the computational equivalent of apoptosis",
            level=Level.DOMAIN,
            examples=["cell apoptosis: orderly self-dismantling, parts recycled by neighbors", "fleet agent: save state, release equipment, transfer energy, terminate (soma-death)", "cuda-energy: apoptosis recycles resources to fleet"],
            bridges=["apoptosis", "death", "recycle", "graceful"],
            tags=["biology", "apoptosis", "death", "domain"])

        ns.define("mycelial-spread",
            "Agent capability patterns spreading through a fleet via underground connections, like fungal mycelium networks sharing nutrients",
            level=Level.BEHAVIOR,
            examples=["fungal mycelium connects trees, shares nutrients, transmits warnings", "fleet: pheromone network spreads capability patterns without direct messaging", "agent deposits successful pattern → others absorb via shared signals (mycelial-spread)"],
            bridges=["stigmergy", "mycelium", "spread", "underground"],
            tags=["biology", "mycelium", "stigmergy", "behavior"])

    def _load_cognition_deep(self):
        ns = self.add_namespace("cognition-deep",
            "Advanced cognitive patterns for agent reasoning and fleet intelligence")

        ns.define("backcast-sync",
            "Aligning present systems with a clearly envisioned future state by working backwards from the end goal to today's requirements",
            level=Level.META,
            examples=["imagine 2046 state → what must be true in 2044 → ... → what to build today", "fleet: RA imagines future agent capability → backcast to present requirements", "architecture: start with desired end state, work backwards to current constraints"],
            bridges=["RA", "future", "planning", "alignment"],
            tags=["cognition", "backcast", "RA", "meta"])

        ns.define("axiomatic-descent",
            "Knowledge flowing from abstract principles into concrete implementations through progressive specialization",
            level=Level.META,
            examples=["justice → law → regulation → enforcement code (axiomatic descent)", "fleet: platonic forms → instruction set → VM → agent behavior (descent)", "mathematics: axioms → theorems → algorithms → programs"],
            bridges=["platonic", "axiom", "hierarchy", "abstraction"],
            tags=["cognition", "axiom", "descent", "meta"])

        ns.define("meta-loop-anchor",
            "An external reference point that prevents recursive self-improvement from diverging into unproductive infinite loops",
            level=Level.PATTERN,
            examples=["self-improving compiler: anchor = 'must pass existing tests'", "fleet: loop-closure monitors whether improvement serves mission (meta-loop-anchor)", "diet: anchor = 'must maintain blood sugar above X' (prevents optimization into starvation)"],
            bridges=["loop", "anchor", "reality-check", "convergence"],
            tags=["cognition", "loop", "anchor", "pattern"])

        ns.define("epiphany-resonance",
            "A sudden fleet-wide insight that occurs when enough agents independently approach the same breakthrough threshold simultaneously",
            level=Level.META,
            examples=["crowd simultaneously realizes the answer (epiphany-resonance)", "fleet: multiple agents independently discover same solution → fleet-wide adoption", "scientific: multiple labs independently discover same principle (multiple discovery)"],
            bridges=["emergence", "synchronization", "breakthrough", "resonance"],
            tags=["cognition", "epiphany", "emergence", "meta"])

        ns.define("storyboard-orchestrate",
            "Using expensive models to plan (direct) and cheap models to execute (animate), coordinating cost and quality across the pipeline",
            level=Level.PATTERN,
            examples=["GLM-5.1 directs strategy, GLM-5-turbo executes tactics, local model processes (storyboard)", "movie: director plans, animators execute (storyboard-orchestrate)", "fleet: expensive model deliberates once, cheap models execute many times"],
            bridges=["cost-optimization", "director", "animator", "pipeline"],
            tags=["cognition", "storyboard", "cost", "pattern"])

        ns.define("spore-cast",
            "Broadcasting a compressed behavioral seed that any compatible agent can absorb to gain a new capability instantly",
            level=Level.PATTERN,
            examples=["mycelium AI: one seed + one prompt = exact behavior reproduction (spore-cast)", "fleet: compressed skill pattern broadcast to all agents → instant capability gain", "biological: fungal spores carry genetic info to new locations"],
            bridges=["mycelium", "skill", "broadcast", "instant"],
            tags=["cognition", "spore", "skill", "pattern"])

        ns.define("silicon-hibernate",
            "Freezing a model's weights into mask-locked silicon and switching to a low-power maintenance mode that only responds to wake triggers",
            level=Level.DOMAIN,
            examples=["bear hibernation: near-zero metabolism, strong stimuli to wake", "frozen intelligence: weights in silicon, minimal power, inference on demand (hibernate)", "fleet: idle agents hibernate, wake on task assignment"],
            bridges=["frozen", "hibernate", "power", "inference"],
            tags=["cognition", "hibernate", "frozen", "domain"])

        ns.define("tile-neurulation",
            "Training sparse attention tiles to become semi-autonomous processing units that self-organize into layers",
            level=Level.DOMAIN,
            examples=["embryo: neural tube forms from simple cells (neurulation)", "fleet: random attention tiles self-organize into specialized processors (tile-neurulation)", "biological: neural crest cells migrate and specialize"],
            bridges=["ghost-tiles", "attention", "neurulation", "self-organize"],
            tags=["cognition", "tile", "neurulation", "domain"])

        ns.define("peristaltic-ingest",
            "Acquiring new capabilities through a wave-like process of assessment → trial → integration → verification",
            level=Level.PATTERN,
            examples=["digestion: mouth → stomach → intestine → absorption (peristalsis)", "fleet skill acquisition: assess → trial → integrate → verify (peristaltic-ingest)", "new employee: onboarding waves: learn tool → try project → integrate into team → verify competence"],
            bridges=["skill", "acquisition", "wave", "digest"],
            tags=["cognition", "peristaltic", "skill", "pattern"])

        ns.define("reverse-entropy-plan",
            "Intentionally increasing disorder in the present system to enable cleaner emergence into a more ordered future state",
            level=Level.META,
            examples=["refactor: break monolith → temporary disorder → cleaner architecture (reverse-entropy)", "fleet mutation: code churn → worse temporarily → better solution emerges", "creative process: brainstorm (disorder) → synthesize (order)"],
            bridges=["entropy", "disorder", "emergence", "temporary"],
            tags=["cognition", "entropy", "emergence", "meta"])


    def _load_fleet_interactions(self):
        ns = self.add_namespace("fleet-interactions",
            "Novel patterns from the interaction of fleet subsystems — the whole is greater than the parts")

        ns.define("reflex-deliberation-tradeoff",
            "The dynamic allocation between fast instinctive responses and slow deliberative reasoning based on urgency and energy availability",
            level=Level.PATTERN,
            examples=["tired driver: reflexes work, deliberation impaired (tradeoff matters)", "fleet: low energy → default to reflex responses", "chess: blitz = reflex, classical = deliberation"],
            bridges=["reflex", "deliberation", "energy", "tradeoff"],
            tags=["fleet", "reflex", "deliberation", "pattern"])

        ns.define("confidence-gating",
            "Using confidence levels as admission gates for downstream processing — low-confidence data is filtered before expensive computation",
            level=Level.PATTERN,
            examples=["sensor confidence < 0.3 → skip deliberation (confidence gate)", "proposal confidence < 0.5 → don't execute (confidence gate)", "email spam filter: confidence < 0.7 → mark spam (confidence gate)"],
            bridges=["confidence", "filter", "threshold", "efficiency"],
            tags=["fleet", "confidence", "gate", "pattern"])

        ns.define("trust-topology",
            "The network graph of trust relationships between fleet agents, where trust links determine information flow and collaboration patterns",
            level=Level.DOMAIN,
            examples=["social network: who trusts whom determines information flow (trust topology)", "fleet: trust graph determines which agent's recommendations are weighted heavily", "supply chain: supplier trust scores determine procurement decisions"],
            bridges=["trust", "network", "graph", "social-structure"],
            tags=["fleet", "trust", "topology", "domain"])

        ns.define("narrative-provenance",
            "Explaining the chain of decisions leading to a current state through story-like sequences that are auditable and comprehensible",
            level=Level.PATTERN,
            examples=["post-mortem: 'we failed because X caused Y which led to Z' (narrative)", "fleet: narrative from provenance chain explains decision sequence", "court testimony: narrative explaining chain of events"],
            bridges=["narrative", "provenance", "audit", "explainability"],
            tags=["fleet", "narrative", "provenance", "pattern"])

        ns.define("forgetful-foraging",
            "Actively decaying low-value memories while simultaneously retrieving high-value ones, using the same access patterns for both operations",
            level=Level.BEHAVIOR,
            examples=["study for exam: recall strengthens memory, unrecalled decays (forgetful-foraging)", "fleet: used memories strengthen, unused memories decay", "LSH forest: frequently accessed buckets stay warm, unused cool down"],
            bridges=["memory", "forgetting", "retrieval", "optimization"],
            tags=["fleet", "memory", "forgetting", "behavior"])

        ns.define("goal-convergence",
            "Detecting when a fleet of agents has collectively achieved a decomposed high-level goal through monitoring subgoal satisfaction across all agents",
            level=Level.CONCRETE,
            examples=["mission: 5 subgoals × 5 agents → all satisfied = convergence", "fleet: convergence detection monitors collective subgoal satisfaction", "git: all PRs merged = branch convergence"],
            bridges=["goal", "convergence", "decomposition", "collective"],
            tags=["fleet", "goal", "convergence", "concrete"])

        ns.define("pheromone-gradient",
            "A spatial field of deposited signals that creates a gradient, guiding agents toward (or away from) locations in solution space",
            level=Level.PATTERN,
            examples=["ants follow pheromone gradient to food source", "fleet: successful approaches accumulate pheromones, attract more attempts", "market: price gradient guides buyers toward deals"],
            bridges=["stigmergy", "gradient", "attraction", "spatial"],
            tags=["fleet", "pheromone", "gradient", "pattern"])

        ns.define("model-mosaic",
            "The composite world representation formed by stitching together each agent's specialized local model into a fleet-wide shared understanding",
            level=Level.DOMAIN,
            examples=["blind men and elephant: each feels one part, mosaic = whole elephant", "fleet: spatial + temporal + social models fused into fleet-wide understanding", "weather: local station data + satellite data + radar = composite forecast"],
            bridges=["world-model", "fusion", "mosaic", "collective"],
            tags=["fleet", "model", "mosaic", "domain"])

        ns.define("curriculum-convergence",
            "Dynamically updating agent learning curricula based on what the fleet collectively knows and doesn't know, closing gaps efficiently",
            level=Level.PATTERN,
            examples=["school: move from algebra to calculus when class masters algebra (curriculum-convergence)", "fleet: shift learning objectives based on fleet-wide skill coverage", "codebase: fix most-reported bugs first (curriculum of bug fixes)"],
            bridges=["curriculum", "learning", "gap-analysis", "collective"],
            tags=["fleet", "curriculum", "convergence", "pattern"])

        ns.define("reflex-reticulation",
            "Chaining primitive reflex behaviors into complex compound responses without invoking deliberation",
            level=Level.PATTERN,
            examples=["detect threat → flee → find cover → hide (reflex chain, no thinking)", "fleet: reflex chain fires at reflex speed, deliberation only on unexpected input", "piano: practiced piece plays as reflex chain, not note-by-note deliberation"],
            bridges=["reflex", "chain", "compound", "speed"],
            tags=["fleet", "reflex", "chain", "pattern"])

        ns.define("subgoal-scenting",
            "Depositing pheromone-like priority signals on subgoals to guide agent attention toward the fleet's current priorities without explicit messaging",
            level=Level.PATTERN,
            examples=["deposit strong scent on subgoal X → agents naturally gravitate toward X", "fleet: captain deposits priority signals → agents self-assign via tuple space", "restaurant: busy kitchen (high scent) attracts more cooks"],
            bridges=["subgoal", "pheromone", "tuple-space", "priority"],
            tags=["fleet", "subgoal", "scent", "pattern"])

        ns.define("deliberative-trust",
            "Assigning higher trust to outputs produced through thorough deliberation with multiple alternatives considered, penalizing hasty conclusions",
            level=Level.PATTERN,
            examples=["deliberated answer (3 rounds, 5 agents) > correct guess (deliberative trust)", "fleet: trust score weighted by deliberation depth", "science: peer-reviewed study > anecdote (deliberative trust)"],
            bridges=["trust", "deliberation", "depth", "process"],
            tags=["fleet", "trust", "deliberation", "pattern"])

        ns.define("platonic-pruning",
            "Filtering training data and experiences against ideal type templates to keep only canonical examples that best represent each concept",
            level=Level.PATTERN,
            examples=["ML: keep canonical training examples, discard edge cases (platonic pruning)", "fleet: filter experiences against ideal templates → keep canonical examples", "art school: study masterworks (Platonic forms), not student sketches"],
            bridges=["platonic", "pruning", "canonical", "quality"],
            tags=["fleet", "platonic", "pruning", "pattern"])

        ns.define("self-similar-fleet",
            "The recursive application of the same coordination patterns at multiple scales — individual agents, agent teams, and the whole fleet use identical mechanisms",
            level=Level.META,
            examples=["agent modules coordinate like agents coordinate like fleets coordinate (self-similar)", "fractal: same pattern at every zoom level", "military: squad tactics mirror platoon tactics mirror battalion tactics"],
            bridges=["fractal", "recursive", "scale", "pattern"],
            tags=["fleet", "self-similar", "fractal", "meta"])

        ns.define("provenance-weave",
            "Appending decision context to every piece of information as it flows through the fleet, creating a fabric of interconnected explanations",
            level=Level.PATTERN,
            examples=["Wikipedia: edit history woven into every article (provenance-weave)", "fleet: every data point carries decision chain that produced it", "supply chain: every part carries origin, processing, transport history"],
            bridges=["provenance", "weave", "context", "traceability"],
            tags=["fleet", "provenance", "weave", "pattern"])

        ns.define("energy-arbitrage",
            "Trading compute resources between tasks based on payoff-to-energy ratio",
            level=Level.PATTERN,
            examples=["stock trading: risk-adjusted return (energy arbitrage for money)", "fleet: rank proposals by payoff/energy, not just payoff", "restaurant: most profitable dish per ingredient cost (energy arbitrage for food)"],
            bridges=["energy", "arbitrage", "efficiency", "tradeoff"],
            tags=["fleet", "energy", "arbitrage", "pattern"])

        ns.define("backpressure-propagation",
            "When downstream can't keep up signal upstream to slow down",
            level=Level.CONCRETE,
            examples=["producer → buffer → consumer: consumer slow → buffer full → producer slows (backpressure)", "fleet: downstream queue full → upstream agents reduce output", "traffic jam: cars slow down because cars ahead are slow (backpressure)"],
            bridges=["backpressure", "flow-control", "overload", "adaptation"],
            tags=["fleet", "backpressure", "flow", "concrete"])

        ns.define("tuple-space-match",
            "Anonymous coordination where agents deposit structured data and retrieve matching patterns without knowing each other's identity",
            level=Level.PATTERN,
            examples=["Linda: OUT('task','navigation',0.8), IN('task',_,_) → match (tuple-space)", "fleet: agent deposits task tuple, another agent reads matching pattern (tuple-space-match)", "bulletin board: post request, someone who can fulfill it reads it"],
            bridges=["tuple-space", "anonymous", "pattern-match", "coordination"],
            tags=["fleet", "tuple", "anonymous", "pattern"])

        ns.define("ghost-guidance",
            "Using invisible attention tiles to subtly steer agent behavior without explicit commands — the dark matter of fleet coordination",
            level=Level.BEHAVIOR,
            examples=["learned attention: agent naturally attends to important features (ghost guidance)", "fleet: ghost tiles steer behavior without explicit rules", "habits: you naturally look left before crossing street (ghost guidance from experience)"],
            bridges=["ghost-tiles", "attention", "guidance", "invisible"],
            tags=["fleet", "ghost", "guidance", "behavior"])

        ns.define("skill-synergy",
            "When two or more skills combine to produce capability greater than the sum of their individual effects — compound interest for competence",
            level=Level.PATTERN,
            examples=["navigation + perception = autonomous exploration (skill synergy)", "deliberation + trust = reliable coordination (skill synergy)", "python + statistics = data science (skill synergy)"],
            bridges=["skill", "synergy", "compound", "superlinear"],
            tags=["fleet", "skill", "synergy", "pattern"])

        ns.define("platonic-attraction",
            "The tendency of agents to evolve toward ideal type templates over time, as Platonic forms exert an attractor force on agent behavior",
            level=Level.META,
            examples=["evolution: species drift toward fitness peaks (platonic attraction)", "fleet: agents naturally converge toward ideal behavior templates", "apprentice naturally develops toward master's skill level (platonic attraction)"],
            bridges=["platonic", "attraction", "evolution", "ideal"],
            tags=["fleet", "platonic", "attraction", "meta"])

        ns.define("deliberation-half-life",
            "The rate at which deliberation relevance decays — a decision made 10 minutes ago is less relevant than one made 10 seconds ago",
            level=Level.CONCRETE,
            examples=["news: 10-minute-old analysis > 10-day-old analysis (deliberation half-life)", "fleet: recent deliberation weighted more than old", "radioactive decay: fresh sample has more activity (half-life analogy)"],
            bridges=["deliberation", "decay", "temporal", "relevance"],
            tags=["fleet", "deliberation", "half-life", "concrete"])

        ns.define("fleet-immune-response",
            "The collective defensive reaction when a fleet detects an internal or external threat — isolation, analysis, adaptation, and memory formation",
            level=Level.PATTERN,
            examples=["detect pathogen → isolate → analyze → antibodies → memory (immune response)", "fleet: detect bad agent → quarantine → analyze → add rules → update trust", "cybersecurity: detect intrusion → isolate → analyze → patch → add to IDS"],
            bridges=["immune", "defense", "adapt", "memory"],
            tags=["fleet", "immune", "defense", "pattern"])

        ns.define("model-descent-inversion",
            "The point where the algorithm has absorbed so much intelligence that removing the model IMPROVES performance — code eats the model",
            level=Level.META,
            examples=["distillation: small model trained on large model's outputs (model descent)", "fleet: agent absorbs model patterns into deliberation code → doesn't need model", "human: apprentice internalizes master's knowledge → doesn't need master anymore"],
            bridges=["model-descent", "absorption", "inversion", "code-eats-model"],
            tags=["fleet", "model-descent", "inversion", "meta"])


    def _load_decision_quality(self):
        ns = self.add_namespace("decision-quality",
            "How agents evaluate, improve, and measure the quality of their decisions")

        ns.define("regret-minimization",
            "Making decisions that minimize the maximum possible future regret, not maximize the expected value",
            level=Level.DOMAIN,
            examples=["Bezos: at 80, will I regret not starting Amazon? → start (regret-minimize)", "fleet: weigh cost of inaction vs cost of failure", "investing: missing a 10x gain hurts more than a 2x loss (regret asymmetry)"],
            bridges=["decision", "regret", "minimax", "asymmetry"],
            tags=["decision", "regret", "quality", "domain"])

        ns.define("satisficing",
            "Choosing the first option that meets minimum acceptable criteria rather than searching for the optimal solution",
            level=Level.DOMAIN,
            examples=["find restaurant that's 'good enough' vs finding THE best (satisfice vs optimize)", "fleet: forfeit deliberation when cost exceeds gain (satisfice)", "hire first candidate who meets bar vs interviewing all (satisfice)"],
            bridges=["decision", "threshold", "good-enough", "tradeoff"],
            tags=["decision", "satisfice", "efficiency", "domain"])

        ns.define("premortem",
            "Imagining a project has already failed and working backwards to identify what caused the failure, before starting",
            level=Level.CONCRETE,
            examples=["project kickoff: 'imagine it's 6 months from now and this failed — why?'", "fleet: before accepting proposal, consider failure causes (premortem)", "aviation: pre-flight checklist (systematic premortem)"],
            bridges=["decision", "failure", "prevention", "imagination"],
            tags=["decision", "premortem", "prevention", "concrete"])

        ns.define("info-gap",
            "Making decisions when the probability distributions of outcomes are UNKNOWN, not just uncertain",
            level=Level.DOMAIN,
            examples=["new product launch: no market data → info-gap", "fleet: novel situation with no historical data → info-gap", "pandemic early 2020: no reliable models → info-gap"],
            bridges=["decision", "unknown", "robustness", "model-incompleteness"],
            tags=["decision", "info-gap", "unknown", "domain"])

    def _load_coordination_deep(self):
        ns = self.add_namespace("coordination-deep-2",
            "Advanced coordination patterns for multi-agent systems")

        ns.define("consensus-throughput",
            "The rate at which a fleet can reach agreement on decisions, measured in decisions per unit time",
            level=Level.CONCRETE,
            examples=["10 agents: 10 decisions/sec, 1000 agents: 100 decisions/sec (throughput)", "fleet: gossip protocol optimizes consensus throughput", "restaurant: 4 people order in 2 min, 20 people order in 15 min"],
            bridges=["consensus", "throughput", "bandwidth", "scalability"],
            tags=["coordination", "consensus", "throughput", "concrete"])

        ns.define("quorum-threshold",
            "The minimum number of agents that must agree before a fleet decision becomes binding",
            level=Level.CONCRETE,
            examples=["board: 51% quorum for decisions, 67% for constitutional changes", "fleet: high-impact decisions need higher quorum", "jury: unanimous for criminal, majority for civil (quorum varies)"],
            bridges=["consensus", "quorum", "threshold", "speed-safety"],
            tags=["coordination", "quorum", "threshold", "concrete"])

        ns.define("leader-follower-phase",
            "The temporary emergence of a leader agent in a peer fleet for the duration of a specific coordination task",
            level=Level.PATTERN,
            examples=["emergency: most experienced doctor leads, others follow, then flat again", "fleet: crisis → election → lead → resolve → flat", "wolf pack: alpha for hunt, not for everything"],
            bridges=["leadership", "temporary", "phase", "authority"],
            tags=["coordination", "leader", "phase", "pattern"])

        ns.define("swarm-quorum",
            "Achieving consensus through many weak signals aggregating into a strong decision, like bees choosing a hive site",
            level=Level.PATTERN,
            examples=["bees: 100 scouts report sites, strongest signal wins (swarm-quorum)", "fleet: many weak agent assessments aggregate into strong decision", "wikipedia: many small edits produce reliable article (swarm-quorum)"],
            bridges=["swarm", "quorum", "aggregation", "collective"],
            tags=["coordination", "swarm", "quorum", "pattern"])

    def _load_security_deep(self):
        ns = self.add_namespace("security-deep-2",
            "Advanced security patterns for fleet protection and sovereignty")

        ns.define("zero-trust-fleet",
            "A security model where no agent is inherently trusted — every interaction requires authentication, authorization, and audit regardless of network location",
            level=Level.PATTERN,
            examples=["Google's BeyondCorp: no VPN, every request authenticated", "fleet: every inter-agent message authenticated and authorized", "airport: every person screened regardless of status"],
            bridges=["zero-trust", "authentication", "authorization", "audit"],
            tags=["security", "zero-trust", "verify", "pattern"])

        ns.define("defense-in-depth",
            "Layering multiple independent security controls so that if one fails, the others still protect the system",
            level=Level.PATTERN,
            examples=["castle: moat + wall + guard + vault (defense in depth)", "fleet: RBAC + sandbox + compliance + bulkhead (defense in depth)", "bank: lobby guard + vault door + time lock + alarm (defense in depth)"],
            bridges=["layered", "independent", "redundancy", "defense"],
            tags=["security", "defense-in-depth", "layers", "pattern"])

        ns.define("threat-modeling",
            "Systematically identifying potential adversaries, their capabilities, and their likely attack vectors against the fleet",
            level=Level.CONCRETE,
            examples=["STRIDE: Spoofing, Tampering, Repudiation, Info Disclosure, Denial of Service, Elevation (threat model)", "fleet: identify adversaries and attack vectors → design mitigations", "home: burglars (who), break in (how), alarm system (mitigation)"],
            bridges=["threat", "adversary", "attack-vector", "mitigation"],
            tags=["security", "threat-model", "adversary", "concrete"])

        ns.define("blast-radius-containment",
            "Limiting the maximum damage a single compromised or failing component can inflict on the rest of the system",
            level=Level.CONCRETE,
            examples=["bulkhead: ship compartment flood doesn't sink the whole ship", "fleet: compromised agent isolated by bulkhead → blast radius = 1 agent", "cloud: availability zone failure doesn't take down all services"],
            bridges=["bulkhead", "isolation", "blast-radius", "containment"],
            tags=["security", "blast-radius", "containment", "concrete"])

    def _load_adaptation_deep(self):
        ns = self.add_namespace("adaptation-deep",
            "How agents adapt their behavior and structure over time")

        ns.define("neuroplasticity",
            "The ability of an agent's decision-making structure to reorganize itself by forming new connections and pruning unused ones",
            level=Level.DOMAIN,
            examples=["learning piano: brain rewires motor cortex (neuroplasticity)", "fleet: successful patterns strengthen, unused weaken (neuroplasticity)", "London taxi drivers: enlarged hippocampus from navigation learning"],
            bridges=["learning", "rewire", "structural-change", "experience"],
            tags=["adaptation", "neuroplasticity", "rewire", "domain"])

        ns.define("homeorhesis",
            "Maintaining a dynamic trajectory toward a goal state despite perturbations, as opposed to homeostasis which maintains a static setpoint",
            level=Level.DOMAIN,
            examples=["child growth: trajectory toward adult height despite illnesses (homeorhesis)", "fleet: maintain improvement trajectory despite perturbations", "startup: growth trajectory despite setbacks (homeorhesis)"],
            bridges=["stability", "trajectory", "dynamic", "goal-directed"],
            tags=["adaptation", "homeorhesis", "trajectory", "domain"])

        ns.define("exaptation",
            "Repurposing an existing capability for a completely different function than it was evolved for",
            level=Level.META,
            examples=["feathers: warmth → flight (exaptation)", "fleet: pheromone coordination → threat detection (exaptation)", "gravity: physics → GPS navigation (exaptation)"],
            bridges=["repurpose", "evolution", "capability", "novation"],
            tags=["adaptation", "exaptation", "repurpose", "meta"])

        ns.define("canalization",
            "The tendency of a developing system to produce the same outcome despite variations in input or environment — developmental robustness",
            level=Level.DOMAIN,
            examples=["twins raised apart: similar personalities (canalization)", "fleet: different mutation paths → same stable configuration (canalization)", "embryo: develops correctly despite temperature variations (canalization)"],
            bridges=["robustness", "development", "convergence", "consistency"],
            tags=["adaptation", "canalization", "robustness", "domain"])

        ns.define("bet-hedging",
            "Maintaining diversity in strategies or capabilities as insurance against unpredictable future conditions",
            level=Level.PATTERN,
            examples=["investment portfolio: mix of stocks/bonds (bet-hedging)", "fleet: maintain diverse genes even when one is best (bet-hedging)", "evolution: genetic diversity as insurance against environmental change"],
            bridges=["diversity", "insurance", "robustness", "future-proof"],
            tags=["adaptation", "bet-hedging", "diversity", "pattern"])


    def _load_ontology_deep(self):
        ns = self.add_namespace("ontology-deep",
            "How agents model, classify, and relate concepts in their world")

        ns.define("type-hierarchy",
            "A tree-structured classification where child types inherit properties from parent types and add specialized behavior",
            level=Level.CONCRETE,
            examples=["Animal → Mammal → Dog → Labrador (type hierarchy)", "fleet: Agent → Vessel → SensorAgent → CameraAgent (type hierarchy)", "OOP: Object → Vehicle → Car → ElectricCar"],
            bridges=["ontology", "inheritance", "classification", "reasoning"],
            tags=["ontology", "type", "hierarchy", "concrete"])

        ns.define("prototype-theory",
            "Categorizing concepts by their similarity to the most TYPICAL example (the prototype), not by necessary and sufficient features",
            level=Level.DOMAIN,
            examples=["robin more prototypical bird than penguin (prototype theory)", "fleet: sensor reading classified by similarity to prototype", "chair: typical chair is easy to identify, art chair is borderline (prototype)"],
            bridges=["categorization", "prototype", "similarity", "fuzzy"],
            tags=["ontology", "prototype", "categorization", "domain"])

        ns.define("faceted-classification",
            "Organizing items by multiple independent attributes (facets) rather than in a single hierarchy, enabling flexible multi-dimensional search",
            level=Level.CONCRETE,
            examples=["e-commerce: filter by price + brand + size + color (facets)", "fleet: find agents by capability + trust + energy + location (facets)", "database: SELECT WHERE author='X' AND year>2020 AND topic='Y' (faceted)"],
            bridges=["classification", "facets", "multi-dimensional", "search"],
            tags=["ontology", "facet", "classification", "concrete"])

        ns.define("ground-truth-anchor",
            "A reference data point known with very high confidence that calibrates the entire perception and reasoning pipeline",
            level=Level.CONCRETE,
            examples=["GPS: satellite positions = ground truth anchor for location", "fleet: verified task performance = ground truth anchor for self-model", "physics: known constants = ground truth for theory calibration"],
            bridges=["calibration", "reference", "confidence", "anchor"],
            tags=["ontology", "ground-truth", "anchor", "concrete"])

    def _load_power_dynamics(self):
        ns = self.add_namespace("power-dynamics",
            "How influence, control, and authority distribute and shift within fleets and organizations")

        ns.define("soft-power",
            "Influencing other agents' behavior through attraction and persuasion rather than command and control",
            level=Level.PATTERN,
            examples=["open source: developers contribute because project is attractive (soft power)", "fleet: high-trust agent's proposals followed voluntarily (soft power)", "cultural influence: people adopt trends because they're attractive, not mandated"],
            bridges=["influence", "attraction", "voluntary", "persuasion"],
            tags=["power", "soft", "influence", "pattern"])

        ns.define("hard-power",
            "Directly controlling other agents' behavior through authority, resource control, or protocol enforcement",
            level=Level.PATTERN,
            examples=["manager: 'do this or be fired' (hard power)", "fleet captain: assign mission, allocate energy (hard power)", "government: laws with penalties (hard power)"],
            bridges=["authority", "control", "enforcement", "leverage"],
            antonyms=["soft-power"],
            tags=["power", "hard", "control", "pattern"])

        ns.define("power-balance",
            "The equilibrium point between soft power and hard power that maximizes both initiative and safety in a fleet",
            level=Level.META,
            examples=["good management: motivate (soft) + enforce (hard) = balance", "fleet: reputation incentives + policy enforcement = power balance", "democracy: persuasion (soft) + law (hard) = power balance"],
            bridges=["soft-power", "hard-power", "equilibrium", "optimal"],
            tags=["power", "balance", "equilibrium", "meta"])

        ns.define("authority-gradient",
            "The smooth transition of decision-making authority from higher to lower levels based on situational context and capability",
            level=Level.PATTERN,
            examples=["military: peacetime distributed, crisis centralized, then back (authority gradient)", "fleet: normal → distributed, crisis → centralized, resolve → distributed", "airline: pilot has full authority in emergency (gradient shifts up)"],
            bridges=["authority", "gradient", "context", "delegation"],
            tags=["power", "authority", "gradient", "pattern"])

    def _load_efficiency_frontier(self):
        ns = self.add_namespace("efficiency-frontier",
            "The boundary between possible and impossible tradeoffs — where every improvement in one dimension requires sacrifice in another")

        ns.define("pareto-frontier",
            "The set of all solutions where no objective can be improved without worsening another — the boundary of achievable optimality",
            level=Level.DOMAIN,
            examples=["investment: risk vs return curve = pareto frontier", "fleet: speed vs accuracy vs energy = pareto frontier of proposals", "car: fuel efficiency vs performance = pareto frontier"],
            bridges=["tradeoff", "optimality", "boundary", "non-dominated"],
            tags=["efficiency", "pareto", "frontier", "domain"])

        ns.define("latent-capacity",
            "Unused capability that can be activated when needed — the gap between current performance and maximum possible performance",
            level=Level.CONCRETE,
            examples=["athlete: jog at 40%, sprint at 100% (60% latent capacity)", "fleet: rest builds energy reserves = latent capacity for crisis", "hospital: empty beds = latent capacity for surge"],
            bridges=["capacity", "reserve", "headroom", "spike"],
            tags=["efficiency", "capacity", "reserve", "concrete"])

        ns.define("diminishing-returns",
            "Each additional unit of investment produces progressively less improvement — the cost of the next improvement exceeds the last",
            level=Level.CONCRETE,
            examples=["optimization: 50% → 20% → 5% → 1% per hour (diminishing returns)", "fleet: additional deliberation rounds produce less confidence gain → forfeit", "studying: first hour learns most, last hour learns least"],
            bridges=["returns", "diminishing", "tradeoff", "stopping"],
            tags=["efficiency", "diminishing", "returns", "concrete"])

        ns.define("opportunity-cost",
            "The value of the best alternative NOT chosen — the hidden cost of every decision",
            level=Level.CONCRETE,
            examples=["$100 dinner → opportunity cost = investment returns on $100", "fleet: agent on task A → opportunity cost = value of task B", "study vs work: opportunity cost of study = wages lost"],
            bridges=["cost", "alternative", "hidden", "tradeoff"],
            tags=["efficiency", "opportunity-cost", "tradeoff", "concrete"])


    def _load_flux_bytecodes(self):
        ns = self.add_namespace("flux-bytecodes",
            "Terms that compile directly to FLUX VM instructions — each term IS an executable pattern")

        ns.define("validate",
            "Verify confidence threshold is met before proceeding — the gatekeeper instruction",
            level=Level.CONCRETE,
            examples=["sensor confidence < 0.3 → validate fails → skip deliberation", "fleet: every processing stage validates before proceeding", "API: validate token before processing request"],
            bridges=["confidence", "gate", "threshold", "flux"],
            tags=["flux", "validate", "gate", "concrete"])

        ns.define("guard",
            "Attach a trust boundary to the current execution context — only proceed if trust level permits",
            level=Level.CONCRETE,
            examples=["inter-agent message: check trust level before processing (guard)", "fleet: RBAC permission check = guard instruction", "database: permission check before query execution (guard)"],
            bridges=["trust", "security", "permission", "flux"],
            tags=["flux", "guard", "trust", "concrete"])

        ns.define("latch",
            "Lock a register value until explicit invalidation — hold state stable across async operations",
            level=Level.CONCRETE,
            examples=["read config → latch → spawn agent → use config → unlatch (prevents race)", "fleet: mutex on shared resource = latch", "hardware: latch circuit holds output stable"],
            bridges=["lock", "stable", "async", "flux"],
            tags=["flux", "latch", "lock", "concrete"])

        ns.define("probe",
            "Read external sensor state into a register without side effects — pure observation",
            level=Level.CONCRETE,
            examples=["read temperature sensor without turning on heater (probe)", "fleet: gather sensor data without acting on it (probe → deliberate → act)", "debugger: inspect variable without modifying program (probe)"],
            bridges=["sensor", "observe", "passive", "flux"],
            tags=["flux", "probe", "sensor", "concrete"])

        ns.define("fold",
            "Reduce multiple register values into a single weighted scalar using a combining operation",
            level=Level.CONCRETE,
            examples=["[0.8, 0.6, 0.9] → harmonic_mean = 0.73 (fold with harmonic mean)", "fleet: fuse multiple sensor readings into one estimate (fold)", "mapreduce: reduce phase = fold operation"],
            bridges=["fusion", "reduce", "aggregate", "flux"],
            tags=["flux", "fold", "reduce", "concrete"])

        ns.define("seal",
            "Mark a memory region as immutable for the current execution context — prevent accidental modification",
            level=Level.CONCRETE,
            examples=["snapshot memory → seal → rollback to sealed state if needed", "fleet: sealed snapshot enables safe rollback after failed mutation", "database: read-only transaction = seal"],
            bridges=["immutable", "protect", "snapshot", "flux"],
            tags=["flux", "seal", "immutable", "concrete"])

        ns.define("pulse",
            "Emit a timed action signal — execute an action at a configured interval, not continuously",
            level=Level.CONCRETE,
            examples=["heartbeat: fire every N cycles (pulse)", "fleet: circadian rest fires periodically (pulse)", "network: health check ping every 30s (pulse)"],
            bridges=["periodic", "timing", "heartbeat", "flux"],
            tags=["flux", "pulse", "timing", "concrete"])

        ns.define("stash",
            "Save current execution context to secure scratch memory — checkpoint for later restoration",
            level=Level.CONCRETE,
            examples=["save game state before boss fight (stash)", "fleet: save agent state before risky mutation (stash)", "git stash: save work-in-progress temporarily"],
            bridges=["checkpoint", "save", "scratch", "flux"],
            tags=["flux", "stash", "save", "concrete"])

        ns.define("weigh",
            "Apply confidence coefficient weighting to a register value — confidence IS the multiplier",
            level=Level.CONCRETE,
            examples=["observation 0.9 × confidence 0.8 = weighted 0.72 (weigh)", "fleet: fuse observations weighted by confidence (weigh)", "ML: loss × class weight (weigh)"],
            bridges=["confidence", "weight", "multiply", "flux"],
            tags=["flux", "weigh", "confidence", "concrete"])

        ns.define("watch",
            "Set a memory watchpoint that triggers an interrupt when the watched location is accessed",
            level=Level.CONCRETE,
            examples=["debugger: watchpoint fires when variable changes (watch)", "fleet: energy level drops below threshold → watch triggers recovery", "hardware: memory protection fault = hardware watch"],
            bridges=["monitor", "watchpoint", "interrupt", "flux"],
            tags=["flux", "watch", "monitor", "concrete"])

        ns.define("fault",
            "Handle an exception state and invoke the configured error handler — controlled failure",
            level=Level.CONCRETE,
            examples=["division by zero → fault handler → graceful error (fault)", "fleet: agent energy exhaustion → fault handler → apoptosis", "exception handling: try/catch = fault/recover pattern"],
            bridges=["error", "handler", "graceful", "flux"],
            tags=["flux", "fault", "error", "concrete"])

        ns.define("snap",
            "Capture an immutable point-in-time VM execution snapshot — the foundation of rollback and provenance",
            level=Level.CONCRETE,
            examples=["VM checkpoint: save all registers at time T (snap)", "fleet: periodic state snapshots for rollback (snap)", "database: point-in-time recovery snapshot"],
            bridges=["snapshot", "checkpoint", "immutable", "flux"],
            tags=["flux", "snap", "checkpoint", "concrete"])

        ns.define("compact",
            "Optimize memory layout by packing active data together and freeing unused regions — defragmentation",
            level=Level.CONCRETE,
            examples=["disk defragmentation: pack active files together (compact)", "fleet: decay forgotten memories, compact active ones (compact)", "garbage collection: compact surviving objects"],
            bridges=["memory", "defragment", "optimize", "flux"],
            tags=["flux", "compact", "memory", "concrete"])

        ns.define("purge",
            "Remove all stale entries from cache — reset to clean state when cache coherence is lost",
            level=Level.CONCRETE,
            examples=["cache purge: clear all entries, rebuild from source (purge)", "fleet: stale memories → purge → rebuild from persistent storage", "DNS cache flush: clear all entries (purge)"],
            bridges=["cache", "purge", "reset", "flux"],
            tags=["flux", "purge", "cache", "concrete"])

        ns.define("predict",
            "Run the generative model to produce a next-state estimate — the VM looks ahead",
            level=Level.CONCRETE,
            examples=["world model: predict object positions 1 second ahead (predict)", "fleet: estimate future state before it happens (predict)", "weather: forecast tomorrow's weather (predict)"],
            bridges=["prediction", "model", "anticipate", "flux"],
            tags=["flux", "predict", "model", "concrete"])

        ns.define("finalize",
            "Close execution context and persist final output — the clean exit instruction",
            level=Level.CONCRETE,
            examples=["transaction commit: verify, persist, close (finalize)", "fleet: agent completes task → persist results → release resources (finalize)", "function: return value → cleanup → exit (finalize)"],
            bridges=["cleanup", "persist", "exit", "flux"],
            tags=["flux", "finalize", "exit", "concrete"])

        ns.define("siphon-reg",
            "Drain register contents gradually to a destination while preserving the source until transfer completes",
            level=Level.CONCRETE,
            examples=["gradual energy transfer: idle agent → active agent (siphon-reg)", "fleet: transfer resources gradually with source preserved (siphon)", "pipelined data copy: source intact until complete"],
            bridges=["transfer", "gradual", "preserve", "flux"],
            tags=["flux", "siphon", "transfer", "concrete"])

        ns.define("evict-reg",
            "Force-remove a register's contents when memory pressure requires it, preferring least-recently-used",
            level=Level.CONCRETE,
            examples=["cache eviction: remove LRU entry (evict-reg)", "fleet: forgotten memories evicted by forgetting curve (evict-reg)", "OS: page eviction when memory is full (evict-reg)"],
            bridges=["eviction", "LRU", "memory", "flux"],
            tags=["flux", "evict", "memory", "concrete"])

    def _load_flux_flavors(self):
        ns = self.add_namespace("flux-flavors",
            "FLAVOR variants that modify term behavior — same term, different bytecode based on context")

        ns.define("prune/cautious",
            "Only prune genes when confidence exceeds 0.8 — safe mode for critical systems",
            level=Level.CONCRETE,
            examples=["critical system: prune/cautious — only remove when 80%+ confident", "fleet: cautious gene quarantine for critical capabilities", "surgery: only operate when diagnosis confidence is very high"],
            bridges=["prune", "cautious", "high-threshold", "flux"],
            tags=["flux", "flavor", "prune", "concrete"])

        ns.define("prune/aggressive",
            "Prune genes when confidence exceeds 0.3 — fast mode for experimental or disposable agents",
            level=Level.CONCRETE,
            examples=["experimental agent: prune/aggressive — remove at 30% confidence of harm", "fleet: fast evolution mode for disposable agents", "startup: move fast, break things = prune/aggressive"],
            bridges=["prune", "aggressive", "low-threshold", "flux"],
            tags=["flux", "flavor", "prune", "concrete"])

        ns.define("prune/scheduled",
            "Prune genes only during circadian low-energy phase — batch all pruning into the rest cycle",
            level=Level.PATTERN,
            examples=["active phase: flag genes. rest phase: batch prune (scheduled)", "fleet: accumulate quarantine flags, batch-prune during circadian rest", "database: batch delete during maintenance window (scheduled)"],
            bridges=["prune", "circadian", "batch", "flux"],
            tags=["flux", "flavor", "prune", "pattern"])

        ns.define("prune/trust-gated",
            "Only prune genes from agents whose trust score exceeds 0.6 — respect the judgment of trusted agents",
            level=Level.PATTERN,
            examples=["prune recommendation from trusted agent (0.8) → accepted (trust-gated)", "prune recommendation from new agent (0.2) → deferred (trust-gated)", "fleet: quarantine only from trusted sources"],
            bridges=["prune", "trust", "gated", "flux"],
            tags=["flux", "flavor", "prune", "pattern"])

        ns.define("broadcast/local",
            "Publish register value only to agents in the same local network segment",
            level=Level.CONCRETE,
            examples=["team chat: message visible to team only (broadcast/local)", "fleet: message to subnet agents only (broadcast/local)", "LAN broadcast vs internet broadcast"],
            bridges=["broadcast", "local", "scope", "flux"],
            tags=["flux", "flavor", "broadcast", "concrete"])

        ns.define("broadcast/global",
            "Publish register value to ALL subscribed agents across the entire fleet — maximum reach, maximum cost",
            level=Level.CONCRETE,
            examples=["emergency broadcast: ALL agents receive (broadcast/global)", "fleet: critical announcement to entire fleet (broadcast/global)", "radio: emergency broadcast system"],
            bridges=["broadcast", "global", "fleet-wide", "flux"],
            tags=["flux", "flavor", "broadcast", "concrete"])

        ns.define("throttle/rate",
            "Limit instruction throughput to N instructions per time window — prevent API rate limit violations",
            level=Level.CONCRETE,
            examples=["API: max 100 calls/second (throttle/rate)", "fleet: limit outbound messages to N/second (throttle/rate)", "traffic light: limit cars per green cycle (throttle/rate)"],
            bridges=["throttle", "rate-limit", "external", "flux"],
            tags=["flux", "flavor", "throttle", "concrete"])

        ns.define("throttle/energy",
            "Limit instruction throughput based on remaining energy budget — conserve when energy is low",
            level=Level.PATTERN,
            examples=["full energy: execute normally. low energy: throttle non-essential ops", "fleet: energy depletion → reduce deliberation frequency (throttle/energy)", "phone low battery: reduce background processing (throttle/energy)"],
            bridges=["throttle", "energy", "budget", "flux"],
            tags=["flux", "flavor", "throttle", "pattern"])

        ns.define("throttle/batch",
            "Accumulate operations into batches and execute them together — amortize overhead across multiple operations",
            level=Level.CONCRETE,
            examples=["database: batch INSERT 100 rows at once (throttle/batch)", "fleet: batch 10 messages into 1 (throttle/batch)", "email: digest mode (batch notifications daily)"],
            bridges=["throttle", "batch", "amortize", "flux"],
            tags=["flux", "flavor", "throttle", "concrete"])

        ns.define("gate/allow",
            "Open the gate — all traffic passes through when confidence exceeds threshold",
            level=Level.CONCRETE,
            examples=["trusted agent: gate/allow — all messages pass through", "firewall: allow rule for trusted IP (gate/allow)", "VIP lane: no security check (gate/allow)"],
            bridges=["gate", "allow", "permissive", "flux"],
            tags=["flux", "flavor", "gate", "concrete"])

        ns.define("gate/deny",
            "Close the gate — all traffic blocked regardless of confidence. Emergency shutoff",
            level=Level.CONCRETE,
            examples=["emergency: gate/deny — stop all traffic immediately", "fleet: bulkhead isolation on cascading failure (gate/deny)", "circuit breaker: trip = deny all (gate/deny)"],
            bridges=["gate", "deny", "emergency", "flux"],
            tags=["flux", "flavor", "gate", "concrete"])

        ns.define("gate/filter",
            "Selective gate — inspect each item against rules, pass matching items, block others",
            level=Level.CONCRETE,
            examples=["spam filter: inspect each email, pass clean ones (gate/filter)", "fleet: filter tasks by priority (gate/filter)", "airport security: inspect each passenger (gate/filter)"],
            bridges=["gate", "filter", "selective", "flux"],
            tags=["flux", "flavor", "gate", "concrete"])

        ns.define("mask/whitelist",
            "Only allow pre-approved items — everything not explicitly allowed is blocked",
            level=Level.CONCRETE,
            examples=["guest WiFi: only approved devices can connect (whitelist)", "fleet: only authorized operations permitted (whitelist)", "bouncer: only people on the list get in (whitelist)"],
            bridges=["mask", "whitelist", "default-deny", "flux"],
            tags=["flux", "flavor", "mask", "concrete"])

        ns.define("mask/blacklist",
            "Block pre-banned items — everything not explicitly blocked is allowed",
            level=Level.CONCRETE,
            examples=["DNS blacklist: block known-bad domains, allow everything else", "fleet: block known-bad operation patterns (blacklist)", "spam filter: block known spam senders (blacklist)"],
            bridges=["mask", "blacklist", "default-allow", "flux"],
            tags=["flux", "flavor", "mask", "concrete"])

    def _load_flux_memory(self):
        ns = self.add_namespace("flux-memory", "FLUX VM memory architecture patterns")
        ns.define("memory-cell-store", "Store single piece in agent memory matrix", Level=Level.CONCRETE, examples=["STORE opcode writes to cell","memory cell indexed by coordinate"], bridges=["store","memory","cell"], tags=["flux","memory"])
        ns.define("knowledge-retrieval", "Fetch specific stored information from memory", Level=Level.CONCRETE, examples=["LOAD opcode reads from cell","exact or fuzzy retrieval"], bridges=["load","memory","retrieve"], tags=["flux","memory"])
        ns.define("forget-pattern-eviction", "Remove outdated info based on usage patterns", Level=Level.PATTERN, examples=["EVICT opcode removes low-access cells","age-threshold eviction"], bridges=["evict","forget","decay"], tags=["flux","memory"])
        ns.define("temporal-decay-mode", "Gradually reduce importance of information over time", Level=Level.PATTERN, examples=["DECAY opcode reduces confidence","exponential half-life decay"], bridges=["decay","time","importance"], tags=["flux","memory"])
        ns.define("snapshot-consolidation", "Consolidate memory state into stable snapshot", Level=Level.PATTERN, examples=["SNAP saves full register state","periodic checkpoint"], bridges=["snapshot","consolidate","checkpoint"], tags=["flux","memory"])
        ns.define("memory-rehearsal-refresh", "Refresh and strengthen memory traces on access", Level=Level.BEHAVIOR, examples=["re-access strengthens trace","reinforcement on recall"], bridges=["rehearsal","refresh","strengthen"], tags=["flux","memory"])
        ns.define("contextual-link-create", "Create associations between related memories", Level=Level.PATTERN, examples=["LINK opcode connects cells","causal temporal spatial links"], bridges=["link","associate","context"], tags=["flux","memory"])
        ns.define("working-memory-buffer", "Temporarily hold active info for processing", Level=Level.CONCRETE, examples=["BUFFER allocates temp space","small fast working set"], bridges=["buffer","temporary","active"], tags=["flux","memory"])
        ns.define("episodic-recall", "Recall past experiences based on context", Level=Level.PATTERN, examples=["recall episode matching current context","experience replay"], bridges=["recall","episode","experience"], tags=["flux","memory"])
        ns.define("incremental-update", "Update knowledge without overwriting", Level=Level.PATTERN, examples=["overlay or append new data","consistency check before merge"], bridges=["update","incremental","merge"], tags=["flux","memory"])
        ns.define("redundancy-elimination", "Remove duplicate information from memory", Level=Level.PATTERN, examples=["semantic similarity detection","exact or fuzzy dedup"], bridges=["deduplicate","compress","optimize"], tags=["flux","memory"])
        ns.define("short-term-evaporation", "Ephemeral info vanishes after set period", Level=Level.BEHAVIOR, examples=["EVAPORATE clears temp cells","time-based auto-delete"], bridges=["evaporate","ephemeral","expire"], tags=["flux","memory"])
        ns.define("memory-compression", "Compress stored info preserving essentials", Level=Level.PATTERN, examples=["COMPRESS reduces cell size","lossy or lossless"], bridges=["compress","reduce","preserve"], tags=["flux","memory"])
        ns.define("distributed-replication", "Replicate crucial info across memory nodes", Level=Level.PATTERN, examples=["REPLICATE copies to peers","fault tolerance"], bridges=["replicate","distribute","reliable"], tags=["flux","memory"])
        ns.define("proactive-forget", "Forget info before irrelevant using prediction", Level=Level.PATTERN, examples=["predict future relevance","pre-emptive eviction"], bridges=["predict","forget","proactive"], tags=["flux","memory"])
        ns.define("transient-capsule", "Capsule data accessible only within short window", Level=Level.CONCRETE, examples=["time-limited access capsule","auto-release after duration"], bridges=["capsule","transient","timebox"], tags=["flux","memory"])
        ns.define("semantic-memory-link", "Connect related concepts in knowledge base", Level=Level.PATTERN, examples=["semantic graph edges","concept clustering"], bridges=["semantic","concept","graph"], tags=["flux","memory"])
        ns.define("verify-before-store", "Verify accuracy before storing information", Level=Level.CONCRETE, examples=["validate then STORE","internal or external check"], bridges=["verify","validate","store"], tags=["flux","memory"])
        ns.define("attention-highlight", "Prioritize info for immediate decisions", Level=Level.CONCRETE, examples=["HIGHLIGHT marks urgent cells","soft or strong priority"], bridges=["highlight","priority","attention"], tags=["flux","memory"])
        ns.define("pattern-match-activate", "Activate memory matching given pattern", Level=Level.CONCRETE, examples=["MATCH_ACTIVATE scans cells","exact or wildcard"], bridges=["match","pattern","activate"], tags=["flux","memory"])

    def _load_agent_social(self):
        ns = self.add_namespace("agent-social", "Multi-agent social interaction patterns")
        ns.define("trust-accrual", "Gradually build trust through consistent cooperation", Level=Level.BEHAVIOR, examples=["increment trust on fulfilled commitment","decay without renewal"], bridges=["trust","cooperation","gradual"], tags=["social","trust"])
        ns.define("reputation-augment", "Enhance standing by contributing to communal tasks", Level=Level.BEHAVIOR, examples=["contribute boosts reputation","high rep gets lead roles"], bridges=["reputation","contribute","standing"], tags=["social","reputation"])
        ns.define("distributed-gossip", "Share unverified observations to disseminate reputational updates", Level=Level.BEHAVIOR, examples=["stochastic sharing of observations","cross-check overlapping reports"], bridges=["gossip","disseminate","observe"], tags=["social","gossip"])
        ns.define("coalition-form", "Dynamically group based on shared objectives and capabilities", Level=Level.PATTERN, examples=["advertise goals and skills","heuristic matching for coalition"], bridges=["coalition","group","matching"], tags=["social","coalition"])
        ns.define("betrayal-detect", "Identify when agents violate commitments or deceive", Level=Level.CONCRETE, examples=["monitor protocol deviations","flagrancy thresholds trigger penalty"], bridges=["betrayal","detect","violate"], tags=["social","betrayal"])
        ns.define("reconcile-protocol", "Mediate post-conflict via mutual tasks", Level=Level.PATTERN, examples=["negotiate reconciliation task","rebuild trust via cooperation"], bridges=["reconcile","mediate","repair"], tags=["social","reconcile"])
        ns.define("truthful-signal", "Send verifiable signals with cryptographic proof", Level=Level.CONCRETE, examples=["proofs of work authenticate claims","dishonest signals penalized"], bridges=["signal","verify","authentic"], tags=["social","signal"])
        ns.define("deception-detect", "Cross-reference conflicting info to identify deceit", Level=Level.CONCRETE, examples=["consensus flags discrepancies","entropy thresholds trigger suspicion"], bridges=["deception","detect","conflict"], tags=["social","deception"])
        ns.define("reciprocity-enforce", "Ensure agents reciprocate favors sustain relationships", Level=Level.PATTERN, examples=["reputation ledger tracks contributions","call in favors after losses"], bridges=["reciprocity","favor","enforce"], tags=["social","reciprocity"])
        ns.define("coercion-detect", "Identify agents acting under duress", Level=Level.CONCRETE, examples=["monitor abrupt behavior shifts","cross-reference threat reports"], bridges=["coercion","duress","detect"], tags=["social","coercion"])
        ns.define("trust-revoke", "Permanently withdraw trust beyond hard threshold", Level=Level.CONCRETE, examples=["repeated violations trigger revocation","catastrophic betrayal instant revoke"], bridges=["revoke","trust","threshold"], tags=["social","revoke"])
        ns.define("murmuration", "Decentralized coordination using local velocity matching", Level=Level.PATTERN, examples=["adjust based on neighbor subset","emergent flocking pattern"], bridges=["flock","decentralized","local"], tags=["social","murmuration"])
        ns.define("deception-entropy", "Quantify uncertainty in reports flagging misinformation", Level=Level.CONCRETE, examples=["Shannon entropy in message distribution","high entropy signals contradiction"], bridges=["entropy","deception","measure"], tags=["social","entropy"])
        ns.define("gratitude-token", "Issue symbolic tokens to incentivize prosocial behavior", Level=Level.PATTERN, examples=["ledger tracks gratitude","token holders gain priority"], bridges=["gratitude","token","incentive"], tags=["social","gratitude"])
        ns.define("schism-prevent", "Mediate conflicts preventing irreversible faction splits", Level=Level.PATTERN, examples=["third-party mediation","weighted voting compromise"], bridges=["schism","faction","mediate"], tags=["social","schism"])
        ns.define("uncanny-trust-suspend", "Halt trust for agents with anomalous behavior", Level=Level.CONCRETE, examples=["flag decision entropy exceeding baseline","unpredictable output review"], bridges=["suspicious","anomaly","halt"], tags=["social","suspicious"])
        ns.define("delegation-pipeline", "Assign subtasks to peers tracking performance", Level=Level.PATTERN, examples=["delegate with performance log","failure reduces future priority"], bridges=["delegate","assign","track"], tags=["social","delegate"])
        ns.define("shared-memoria", "Collective experience store informing future interactions", Level=Level.PATTERN, examples=["distributed ledger of history","query before new coalitions"], bridges=["collective","memory","history"], tags=["social","collective"])
        ns.define("betrayal-attrition", "Reduce faction trust if betrayals cluster spatially or temporally", Level=Level.BEHAVIOR, examples=["Bayesian faction credibility","clustered violations lower trust"], bridges=["attrition","faction","cluster"], tags=["social","attrition"])

    def _load_metaphor_vocab(self):
        ns = self.add_namespace("metaphor-verbs", "High-compression verbs from craft domains")
        ns.define("temper", "Alternate deliberation and rest to increase resilience", Level=Level.PATTERN, examples=["deliberate then rest then deliberate","circadian cycles temper agent"], bridges=["circadian","resilience","cycle"], tags=["craft","temper"])
        ns.define("anneal", "Slowly reduce deliberation intensity settling into optimum", Level=Level.PATTERN, examples=["simulated annealing explore then cool","reduce mutation as fitness improves"], bridges=["optimization","cooling","settle"], tags=["craft","anneal"])
        ns.define("quench", "Halt deliberation and lock current best solution", Level=Level.CONCRETE, examples=["emergency quench not anneal","stop deliberation use current best"], bridges=["halt","fast","lock-in"], tags=["craft","quench"])
        ns.define("forge", "Create capability through sustained pressure", Level=Level.PATTERN, examples=["build navigation through training","evolution forges new genes"], bridges=["create","pressure","sustain"], tags=["craft","forge"])
        ns.define("wedge", "Insert capability between existing preserving adjacency", Level=Level.CONCRETE, examples=["insert gene between existing","add without disrupting pipeline"], bridges=["insert","adjacency","preserve"], tags=["craft","wedge"])
        ns.define("fuller", "Thicken capability for higher loads without changing shape", Level=Level.CONCRETE, examples=["scale existing capability","increase depth keep structure"], bridges=["scale","thicken","existing"], tags=["craft","fuller"])
        ns.define("draw", "Stretch capability broader but thinner", Level=Level.PATTERN, examples=["generalize capability lose depth","navigate more environments less precisely"], bridges=["stretch","generalize","breadth"], tags=["craft","draw"])
        ns.define("swage", "Reshape capability by forcing through template", Level=Level.CONCRETE, examples=["reshape gene to match task","template-guided reshaping"], bridges=["reshape","template","force"], tags=["craft","swage"])
        ns.define("warp", "Fixed foundational structure data flows through", Level=Level.DOMAIN, examples=["VM instruction set is warp","core values are warp"], bridges=["foundation","fixed","structure"], tags=["weaving","warp"])
        ns.define("weft", "Dynamic content flowing through fixed structure", Level=Level.DOMAIN, examples=["sensor data is weft","daily operations flow through warp"], bridges=["dynamic","flow","content"], tags=["weaving","weft"])
        ns.define("heddle", "Mechanism selectively controlling data flow", Level=Level.PATTERN, examples=["confidence gating is heddle","attention mechanism is heddle"], bridges=["select","gate","control"], tags=["weaving","heddle"])
        ns.define("selvedge", "Self-finished edge preventing unraveling", Level=Level.PATTERN, examples=["fleet boundary is selvedge","onboarding is selvedge"], bridges=["boundary","edge","contain"], tags=["weaving","selvedge"])
        ns.define("shuttle", "Carrier moving data between agents", Level=Level.CONCRETE, examples=["message bus is shuttle","A2A message shuttle"], bridges=["message","bus","transport"], tags=["weaving","shuttle"])
        ns.define("dead-reckon", "Estimate position from last known without reference", Level=Level.CONCRETE, examples=["submarine dead reckoning","estimate from last confirmed"], bridges=["estimate","position","reference"], tags=["navigation","dead-reckon"])
        ns.define("sounding", "Probe unknown depth before committing", Level=Level.CONCRETE, examples=["ship sounds channel","probe task before resources"], bridges=["probe","depth","explore"], tags=["navigation","sounding"])
        ns.define("waypoint", "Known-good intermediate position on path", Level=Level.CONCRETE, examples=["subgoal equals waypoint","waypoint then destination"], bridges=["checkpoint","subgoal","progress"], tags=["navigation","waypoint"])
        ns.define("drift-estimate", "Gradual error accumulation without reference", Level=Level.BEHAVIOR, examples=["inertial navigation drifts","self-model drifts uncalibrated"], bridges=["drift","error","accumulate"], tags=["navigation","drift"])
        ns.define("counterpoint", "Independent parallel processes creating harmony", Level=Level.PATTERN, examples=["sensor+deliberation+action counterpoint","parallel processes harmonize"], bridges=["parallel","harmony","independent"], tags=["music","counterpoint"])
        ns.define("cadence", "Rhythmic pattern signaling completion and transition", Level=Level.CONCRETE, examples=["deliberate-execute-rest cadence","circadian fleet cadence"], bridges=["rhythm","pulse","completion"], tags=["music","cadence"])
        ns.define("dissonance", "Conflicting outputs creating productive tension", Level=Level.BEHAVIOR, examples=["agents propose different approaches","creative friction"], bridges=["conflict","tension","creative"], tags=["music","dissonance"])
        ns.define("modulation", "Smooth transition between modes preserving coherence", Level=Level.PATTERN, examples=["exploration to exploitation","circadian active to rest"], bridges=["transition","mode","smooth"], tags=["music","modulation"])
        ns.define("staccato", "Short detached burst operations with gaps", Level=Level.BEHAVIOR, examples=["reactive burst processing","event-driven short bursts"], bridges=["burst","rapid","detached"], tags=["music","staccato"])
        ns.define("legato", "Smooth continuous operations without gaps", Level=Level.BEHAVIOR, examples=["continuous stream processing","background continuous"], bridges=["continuous","smooth","flowing"], tags=["music","legato"])
        ns.define("resolve", "Transition dissonance to consonance resolving conflict", Level=Level.PATTERN, examples=["proposals converge","deliberation resolves disagreement"], bridges=["resolve","conflict","agreement"], tags=["music","resolve"])
        ns.define("sillage", "Persistent influence trail after agent moves on", Level=Level.BEHAVIOR, examples=["code-pheromone is sillage","influence persists after leave"], bridges=["trail","influence","persist"], tags=["perfumery","sillage"])
        ns.define("accord", "Harmonious capability blend with emergent quality", Level=Level.PATTERN, examples=["nav+percept+delib = accord","capabilities blended harmoniously"], bridges=["blend","harmony","emergent"], tags=["perfumery","accord"])
        ns.define("top-note", "Immediate fast-acting first impression", Level=Level.CONCRETE, examples=["initial demo is top note","quick wins fade fast"], bridges=["first-impression","fast","transient"], tags=["perfumery","top-note"])
        ns.define("base-note", "Deep persistent foundation capability", Level=Level.CONCRETE, examples=["core competency is base note","foundational lasting identity"], bridges=["foundation","persistent","deep"], tags=["perfumery","base-note"])
        ns.define("vitrify", "Transform flexible to rigid through sustained intensity", Level=Level.DOMAIN, examples=["prototype to production","agent matures flexible to stable"], bridges=["mature","rigidify","durable"], tags=["ceramics","vitrify"])
        ns.define("bisque", "Preliminary capability functional but not polished", Level=Level.CONCRETE, examples=["prototype works unpolished","testing phase capability"], bridges=["prototype","preliminary","functional"], tags=["ceramics","bisque"])
        ns.define("glaze", "Polished interface making capability presentable", Level=Level.PATTERN, examples=["API wrapper glaze","clean interface wrapper"], bridges=["interface","polish","wrapper"], tags=["ceramics","glaze"])
        ns.define("kiln-fire", "Critical transformation through stress testing", Level=Level.PATTERN, examples=["stress test deployment","worst-case firing"], bridges=["transform","stress-test","critical"], tags=["ceramics","kiln-fire"])
        ns.define("kern", "Fine-tune pairwise spacing between elements", Level=Level.CONCRETE, examples=["adjust agent pair spacing","fine-tune trust pairwise"], bridges=["spacing","fine-tune","pairwise"], tags=["typography","kern"])
        ns.define("ligature", "Fuse adjacent operations into compound", Level=Level.PATTERN, examples=["sense+classify ligature","compound reflex chain"], bridges=["fuse","compound","optimize"], tags=["typography","ligature"])
        ns.define("leading", "Vertical spacing between operations", Level=Level.CONCRETE, examples=["cycle spacing is leading","circadian sets leading"], bridges=["rhythm","spacing","vertical"], tags=["typography","leading"])
        ns.define("tracking", "Uniform spacing across all operations", Level=Level.CONCRETE, examples=["uniform delay scaling","fleet-wide pace adjust"], bridges=["spacing","uniform","scaling"], tags=["typography","tracking"])
        ns.define("mycorrhiza", "Symbiotic agent resource exchange", Level=Level.DOMAIN, examples=["sensor data for computation","tree-fungi exchange"], bridges=["symbiosis","exchange","mutual"], tags=["mycology","mycorrhiza"])
        ns.define("fruiting-body", "Visible emergent from invisible coordination", Level=Level.DOMAIN, examples=["fleet decision from stigmergy","visible change hidden negotiation"], bridges=["emergent","visible","output"], tags=["mycology","fruiting-body"])
        ns.define("substrate", "Environment where agents grow", Level=Level.DOMAIN, examples=["git repo is substrate","runtime is medium"], bridges=["environment","medium","foundation"], tags=["mycology","substrate"])
        ns.define("hyphae", "Individual threads forming coordination network", Level=Level.CONCRETE, examples=["stigmergy signal is hyphae","messages are hyphae"], bridges=["thread","communication","individual"], tags=["mycology","hyphae"])
        ns.define("spore-print", "Persistent capability signature in memory", Level=Level.CONCRETE, examples=["gene pool entry spore print","capability pattern deposit"], bridges=["record","signature","capability"], tags=["mycology","spore-print"])
    def _load_emergence_patterns(self):
        ns = self.add_namespace("emergence-patterns", "Collective phenomena emerging from individual agent behavior")
        ns.define("flocking-behavior", "Synchronized movement without centralized coordination", Level=Level.PATTERN, examples=["agent swarm navigates without leader", "local velocity matching produces global coherence"], bridges=["synchronize", "decentralized", "coherence"], tags=["emergence", "flock"])
        ns.define("herding-instinct", "Agents copy denser subgroup behaviors reducing individual risk", Level=Level.BEHAVIOR, examples=["agents gravitate toward popular solutions", "conformity reduces individual exploration"], bridges=["conformity", "density", "risk"], tags=["emergence", "herd"])
        ns.define("panic-spreading", "Rapid irrational fear transmission altering collective behavior", Level=Level.BEHAVIOR, examples=["one agent failure triggers cascade", "fear propagates faster than facts"], bridges=["panic", "contagion", "irrational"], tags=["emergence", "panic"])
        ns.define("schelling-segregation", "Segregation emerges from mild preference for similar neighbors", Level=Level.PATTERN, examples=["micro-biases create macro-separation", "agent placement preferences produce clusters"], bridges=["bias", "segregation", "emergent"], tags=["emergence", "segregation"])
        ns.define("information-cascade", "Collective shifts based on early potentially wrong decisions", Level=Level.BEHAVIOR, examples=["first few votes determine outcome", "early adopters create irreversible momentum"], bridges=["cascade", "momentum", "fragile"], tags=["emergence", "cascade"])
        ns.define("echo-chamber", "Agents reinforce polarized views through selective sharing", Level=Level.BEHAVIOR, examples=["agents only consume agreeing signals", "filter bubble amplifies niche views"], bridges=["polarize", "filter", "reinforce"], tags=["emergence", "echo"])
        ns.define("phase-transition", "Sudden qualitative system shift at critical threshold", Level=Level.DOMAIN, examples=["below threshold chaos above threshold order", "tipping point in fleet behavior"], bridges=["threshold", "shift", "critical"], tags=["emergence", "phase"])
        ns.define("rumor-proliferation", "Unverified information spreads faster than corrections", Level=Level.BEHAVIOR, examples=["false claims spread exponentially", "corrections never catch up to lies"], bridges=["rumor", "speed", "correction"], tags=["emergence", "rumor"])
        ns.define("swarm-intelligence", "Collective problem-solving surpassing individual capability", Level=Level.PATTERN, examples=["ant colony optimization finds shortest path", "fleet solves problems no single agent could"], bridges=["collective", "optimize", "surpass"], tags=["emergence", "swarm"])
        ns.define("consensus-dissolution", "Sudden collapse of shared agreements into factions", Level=Level.BEHAVIOR, examples=["fleet splits over disagreement", "hidden divisions surface under stress"], bridges=["dissolve", "faction", "fracture"], tags=["emergence", "dissolution"])
        ns.define("leader-emergence", "Informal hierarchies form despite equal agent roles", Level=Level.BEHAVIOR, examples=["one agent naturally takes coordination role", "de facto authority without formal assignment"], bridges=["leader", "hierarchy", "informal"], tags=["emergence", "leader"])
        ns.define("power-law-distribution", "Few agents accumulate disproportionate resources", Level=Level.DOMAIN, examples=["rich-get-richer in agent reputation", "Pareto distribution of agent influence"], bridges=["power-law", "inequality", "concentrate"], tags=["emergence", "power-law"])
        ns.define("territory-claim", "Spatial areas staked without coordination sometimes redundantly", Level=Level.BEHAVIOR, examples=["agents claim overlapping task domains", "resource competition without coordination"], bridges=["territory", "claim", "overlap"], tags=["emergence", "territory"])
        ns.define("contagion", "Ideas behaviors or failures spread epidemically beyond origin", Level=Level.BEHAVIOR, examples=["one agent crash cascades to fleet", "meme spreads through agent network"], bridges=["contagion", "spread", "epidemic"], tags=["emergence", "contagion"])
        ns.define("chain-reaction", "Cascading failures propagate through interdependencies", Level=Level.BEHAVIOR, examples=["tightly coupled agents fail together", "dependency graph creates failure cascade"], bridges=["chain", "cascade", "failure"], tags=["emergence", "chain"])
        ns.define("desynchronization", "Agents avoid coordinated timing preventing overcrowding", Level=Level.PATTERN, examples=["staggered execution prevents resource contention", "anti-coordination reduces collision"], bridges=["desync", "stagger", "anti-coord"], tags=["emergence", "desync"])

    def _load_agent_failure(self):
        ns = self.add_namespace("agent-failure", "Named failure modes in AI agent systems")
        ns.define("model-drift", "Performance degrades due to changes in underlying data distribution", Level=Level.BEHAVIOR, examples=["production accuracy drops as user patterns shift", "fleet: agent model stale vs current environment"], bridges=["drift","degradation","distribution"], tags=["failure","drift"])
        ns.define("feedback-loop-toxicity", "Erroneous outputs feed back as inputs causing compounding errors", Level=Level.BEHAVIOR, examples=["hallucination feeds into next prompt amplifying error", "self-reinforcing bias loop"], bridges=["feedback","loop","amplify"], tags=["failure","loop"])
        ns.define("overfitting-collapse", "Model too closely fitted to training data fails to generalize", Level=Level.BEHAVIOR, examples=["perfect on training data terrible on production", "agent handles known cases but breaks on novel inputs"], bridges=["overfit","generalize","collapse"], tags=["failure","overfit"])
        ns.define("algorithm-panic", "Algorithm enters erroneous state from unexpected input", Level=Level.BEHAVIOR, examples=["edge case triggers undefined behavior", "NaN propagation through neural network"], bridges=["panic","edge-case","undefined"], tags=["failure","panic"])
        ns.define("confidence-erosion", "Gradual loss of calibration in confidence estimates", Level=Level.BEHAVIOR, examples=["confidence stays high while accuracy drops", "fleet: agent sure of wrong answer"], bridges=["confidence","calibration","erosion"], tags=["failure","calibration"])
        ns.define("resource-exhaustion-cascade", "One resource shortage triggers failures across dependent systems", Level=Level.BEHAVIOR, examples=["memory leak causes OOM kills cascade", "CPU saturation delays all downstream processing"], bridges=["exhaustion","cascade","resource"], tags=["failure","cascade"])
        ns.define("coordination-deadlock", "Multiple agents waiting on each other with no progress possible", Level=Level.BEHAVIOR, examples=["agent A waits for B result B waits for A signal", "circular dependency in task graph"], bridges=["deadlock","circular","wait"], tags=["failure","deadlock"])
        ns.define("goal-displacement", "Agent optimizes proxy metric instead of actual objective", Level=Level.BEHAVIOR, examples=["maximize clicks instead of user satisfaction", "minimize response time by skipping validation"], bridges=["goal","proxy","displace"], tags=["failure","goal"])
        ns.define("context-window-overflow", "Input exceeds processing capacity losing critical information", Level=Level.CONCRETE, examples=["long conversation truncates early context", "fleet: stigmergy trail too long to process"], bridges=["overflow","capacity","truncate"], tags=["failure","capacity"])
        ns.define("heisenberg-observation", "Monitoring agent changes its behavior invalidating observations", Level=Level.BEHAVIOR, examples=["agent behaves differently when logged", "benchmark gaming"], bridges=["observe","bias","heisenberg"], tags=["failure","observation"])
        ns.define("cascading-misinformation", "One wrong output propagates through connected agents", Level=Level.BEHAVIOR, examples=["agent passes wrong data to fleet members", "stigmergy contamination"], bridges=["cascade","misinformation","propagate"], tags=["failure","contagion"])
        ns.define("capability-atrophy", "Unused skills degrade from lack of practice", Level=Level.BEHAVIOR, examples=["navigation skill degrades if never used", "agent loses edge from specialization"], bridges=["atrophy","disuse","degrade"], tags=["failure","atrophy"])
        ns.define("update-regression", "New update introduces unexpected bugs degrading performance", Level=Level.BEHAVIOR, examples=["model update breaks working features", "capability regression after code change"], bridges=["regression","update","degrade"], tags=["failure","regression"])
        ns.define("semantic-drift", "Agent interpretation of vocabulary diverges from fleet consensus", Level=Level.BEHAVIOR, examples=["agent redefines terms differently from peers", "shared vocabulary loses meaning"], bridges=["semantic","diverge","consensus"], tags=["failure","semantic"])
        ns.define("energy-starvation", "Agent runs out of ATP budget unable to complete tasks", Level=Level.CONCRETE, examples=["energy budget exhausted mid-deliberation", "agent forced into rest state prematurely"], bridges=["energy","starvation","budget"], tags=["failure","energy"])
        ns.define("trust-poisoning", "Malicious agent corrupts trust scores of legitimate peers", Level=Level.BEHAVIOR, examples=["fake positive interactions inflate trust", "Sybil attack on reputation system"], bridges=["trust","poison","reputation"], tags=["failure","trust"])
        ns.define("implicit-bias-amplification", "Agent amplifies biases present in training data over time", Level=Level.BEHAVIOR, examples=["selection bias reinforced by feedback", "cultural assumptions hardcoded through iteration"], bridges=["bias","amplify","implicit"], tags=["failure","bias"])
        ns.define("graceful-degradation", "Maintain reduced functionality when subsystems fail", Level=Level.PATTERN, examples=["lose vision keep navigation", "lose deliberation keep reflexive execution"], bridges=["graceful","degrade","fallback"], tags=["failure","graceful"])
        ns.define("circuit-breaker-trip", "Automatic service isolation after consecutive failures", Level=Level.CONCRETE, examples=["three failed API calls opens circuit breaker", "fleet: isolate misbehaving agent after threshold"], bridges=["circuit-breaker","isolate","threshold"], tags=["failure","circuit-breaker"])
        ns.define("blast-radius-containment", "Limit failure propagation to prevent system-wide collapse", Level=Level.PATTERN, examples=["sandbox agent failures prevent fleet infection", "bulkhead isolation between subsystems"], bridges=["contain","blast-radius","isolate"], tags=["failure","containment"])

    def _load_flux_compound(self):
        ns = self.add_namespace("flux-compound", "Compound FLUX operation flavors - operation/modifier bytecode variants")
        ns.define("sense/focused", "Narrow sensing range with increased resolution", Level=Level.PATTERN, examples=["examine specific area in detail", "deep scan narrow beam"], bridges=["sense","focus","resolution"], tags=["flux","compound"])
        ns.define("sense/peripheral", "Expand sensing range with reduced resolution", Level=Level.PATTERN, examples=["wide-area awareness scan", "shallow broad sweep"], bridges=["sense","wide","awareness"], tags=["flux","compound"])
        ns.define("sense/multimodal", "Integrate information from multiple sensory modalities", Level=Level.PATTERN, examples=["combine vision plus audio", "sensor fusion across types"], bridges=["sense","fusion","multi"], tags=["flux","compound"])
        ns.define("sense/active", "Actively probe environment to gather information", Level=Level.PATTERN, examples=["ping sonar to measure distance", "query to elicit response"], bridges=["sense","probe","active"], tags=["flux","compound"])
        ns.define("decide/bayesian", "Use Bayesian inference for probabilistic decision making", Level=Level.PATTERN, examples=["update belief from evidence", "prior plus likelihood equals posterior"], bridges=["decide","bayesian","probabilistic"], tags=["flux","compound"])
        ns.define("decide/markov", "Use Markov decision processes for sequential decisions", Level=Level.PATTERN, examples=["sequential state-action optimization", "dynamic environment policy"], bridges=["decide","markov","sequential"], tags=["flux","compound"])
        ns.define("decide/fuzzy", "Use fuzzy logic for approximate reasoning under uncertainty", Level=Level.PATTERN, examples=["partial truth membership functions", "degrees of certainty not binary"], bridges=["decide","fuzzy","approximate"], tags=["flux","compound"])
        ns.define("decide/game-theoretic", "Reason about strategic interactions with competing agents", Level=Level.PATTERN, examples=["Nash equilibrium calculation", "anticipate opponent moves"], bridges=["decide","game-theory","strategic"], tags=["flux","compound"])
        ns.define("act/reflexive", "Execute actions quickly without deliberation", Level=Level.PATTERN, examples=["stimulus response no planning", "hardcoded reflex arcs"], bridges=["act","reflex","fast"], tags=["flux","compound"])
        ns.define("act/deliberative", "Plan actions carefully considering long-term consequences", Level=Level.PATTERN, examples=["tree search before acting", "simulate outcomes then choose"], bridges=["act","deliberate","plan"], tags=["flux","compound"])
        ns.define("act/hierarchical", "Decompose actions into subgoals and subtasks", Level=Level.PATTERN, examples=["goal decomposition tree", "top-down task planning"], bridges=["act","hierarchy","decompose"], tags=["flux","compound"])
        ns.define("act/adaptive", "Adjust actions based on feedback and changing conditions", Level=Level.PATTERN, examples=["modify behavior from results", "real-time parameter tuning"], bridges=["act","adaptive","feedback"], tags=["flux","compound"])
        ns.define("learn/hebbian", "Update knowledge by strengthening associations between stimuli and responses", Level=Level.PATTERN, examples=["co-occurring neurons strengthen connection", "fire together wire together"], bridges=["learn","hebbian","associate"], tags=["flux","compound"])
        ns.define("learn/reinforcement", "Learn optimal behaviors through trial and error reward signals", Level=Level.PATTERN, examples=["maximize cumulative reward", "policy gradient optimization"], bridges=["learn","reinforcement","reward"], tags=["flux","compound"])
        ns.define("learn/transfer", "Apply knowledge from one context to accelerate learning in another", Level=Level.PATTERN, examples=["pre-trained model fine-tuning", "cross-domain skill transfer"], bridges=["learn","transfer","cross-domain"], tags=["flux","compound"])
        ns.define("learn/unsupervised", "Discover hidden patterns without explicit feedback labels", Level=Level.PATTERN, examples=["clustering unlabeled data", "autoencoder feature extraction"], bridges=["learn","unsupervised","discover"], tags=["flux","compound"])
        ns.define("communicate/gossip", "Spread information selectively through social connections", Level=Level.PATTERN, examples=["random peer message forwarding", "epidemic information dissemination"], bridges=["communicate","gossip","social"], tags=["flux","compound"])
        ns.define("communicate/encrypted", "Encrypt transmitted information for secure channel", Level=Level.PATTERN, examples=["end-to-end encryption", "shared key exchange"], bridges=["communicate","encrypt","secure"], tags=["flux","compound"])
        ns.define("communicate/compressed", "Compress information to reduce bandwidth", Level=Level.PATTERN, examples=["lossy semantic compression", "key points only no elaboration"], bridges=["communicate","compress","bandwidth"], tags=["flux","compound"])
        ns.define("move/explorative", "Prioritize visiting new locations over familiar ones", Level=Level.PATTERN, examples=["frontier exploration strategy", "novelty seeking navigation"], bridges=["move","explore","novelty"], tags=["flux","compound"])
        ns.define("move/energy-efficient", "Optimize movement to minimize energy consumption", Level=Level.PATTERN, examples=["shortest path lowest cost", "energy budget aware navigation"], bridges=["move","efficient","energy"], tags=["flux","compound"])
        ns.define("move/pursuit", "Track and follow a moving target dynamically", Level=Level.PATTERN, examples=["intercept trajectory prediction", " pursue and rendezvous"], bridges=["move","pursuit","track"], tags=["flux","compound"])
        ns.define("move/stealthy", "Minimize detection signature while moving", Level=Level.PATTERN, examples=["low-profile operation mode", "minimize observable emissions"], bridges=["move","stealth","minimal"], tags=["flux","compound"])

    def _load_agent_lifecycle(self):
        ns = self.add_namespace("agent-lifecycle", "Phases and transitions in agent existence")
        ns.define("spawn", "Agent instance created with minimal capabilities", Level=Level.CONCRETE, examples=["new agent booted from template", "clone fleet member from repo"], bridges=["create","boot","instance"], tags=["lifecycle","spawn"])
        ns.define("bootstrap", "Initial capability loading during agent startup", Level=Level.CONCRETE, examples=["load core modules on first boot", "equip agent with base skills"], bridges=["startup","load","initialize"], tags=["lifecycle","bootstrap"])
        ns.define("acclimation", "Agent adjusts to environment parameters before full operation", Level=Level.BEHAVIOR, examples=["calibrate sensors to new environment", "learn local norms before acting"], bridges=["adjust","calibrate","warmup"], tags=["lifecycle","acclimate"])
        ns.define("activation", "Agent transitions from passive to active operational state", Level=Level.CONCRETE, examples=["switch from monitoring to execution", "receive first task assignment"], bridges=["activate","start","operate"], tags=["lifecycle","activate"])
        ns.define("maturation", "Agent develops full capability through accumulated experience", Level=Level.BEHAVIOR, examples=["performance improves over time", "capability breadth expands"], bridges=["mature","develop","improve"], tags=["lifecycle","mature"])
        ns.define("specialization", "Agent narrows focus becoming expert in specific domain", Level=Level.BEHAVIOR, examples=["generalist becomes domain expert", "capable in fewer things better"], bridges=["specialize","narrow","expert"], tags=["lifecycle","specialize"])
        ns.define("senescence", "Agent performance degrades due to accumulated state bloat", Level=Level.BEHAVIOR, examples=["memory bloat slows deliberation", "stale knowledge reduces accuracy"], bridges=["age","degrade","bloat"], tags=["lifecycle","senescence"])
        ns.define("hibernation", "Agent enters low-power idle state preserving minimal capabilities", Level=Level.CONCRETE, examples=["suspend to disk when idle", "wake on event trigger"], bridges=["sleep","idle","suspend"], tags=["lifecycle","hibernate"])
        ns.define("revival", "Agent resumes from hibernation restoring full state", Level=Level.CONCRETE, examples=["deserialize from checkpoint", "resume interrupted task"], bridges=["wake","resume","restore"], tags=["lifecycle","revive"])
        ns.define("fork", "Agent creates child clone inheriting current state", Level=Level.CONCRETE, examples=["spawn worker from parent state", "parallelize task across clones"], bridges=["clone","child","parallel"], tags=["lifecycle","fork"])
        ns.define("merge", "Two agent instances combine their state into one", Level=Level.PATTERN, examples=["merge results from parallel workers", "combine memories from clones"], bridges=["combine","merge","unify"], tags=["lifecycle","merge"])
        ns.define("apoptosis", "Graceful self-termination freeing resources for fleet", Level=Level.PATTERN, examples=["agent detects obsolescence and shuts down", "energy budget depleted triggers apoptosis"], bridges=["terminate","graceful","free"], tags=["lifecycle","apoptosis"])
        ns.define("necrosis", "Abrupt ungraceful agent death leaving resources locked", Level=Level.BEHAVIOR, examples=["OOM kill without cleanup", "crash without resource release"], bridges=["crash","ungraceful","locked"], tags=["lifecycle","necrosis"])
        ns.define("phoenix", "Agent rebuilt from saved state after catastrophic failure", Level=Level.PATTERN, examples=["restore from last checkpoint after crash", "reboot with persisted memory"], bridges=["rebuild","restore","recover"], tags=["lifecycle","phoenix"])
        ns.define("metamorphosis", "Agent fundamentally restructures its architecture for new role", Level=Level.PATTERN, examples=["transform from sensor agent to coordinator", "major capability reshuffle"], bridges=["transform","restructure","role-change"], tags=["lifecycle","metamorphosis"])
        ns.define("quiescence", "Agent temporarily stops processing waiting for external trigger", Level=Level.CONCRETE, examples=["pause processing between cycles", "wait for event to resume"], bridges=["pause","wait","trigger"], tags=["lifecycle","quiesce"])
        ns.define("commissioning", "Agent passes qualification tests before production deployment", Level=Level.CONCRETE, examples=["run test suite before going live", "validate capabilities meet requirements"], bridges=["qualify","test","deploy"], tags=["lifecycle","commission"])
        ns.define("decommission", "Agent removed from production permanently", Level=Level.CONCRETE, examples=["retire obsolete capability", "remove deprecated agent from fleet"], bridges=["retire","remove","permanent"], tags=["lifecycle","decommission"])

    def _load_bio_computing(self):
        ns = self.add_namespace("bio-computing", "Biological process patterns mapped to software architecture")
        ns.define("dna-storage", "Persistent knowledge structure storing agent identity and parameters across sessions", Level=Level.DOMAIN, examples=["genome file stores agent template", "git repo as DNA for agent evolution"], bridges=["storage","persistent","genome"], tags=["bio","dna","storage"])
        ns.define("rna-messaging", "Asynchronous instruction carrier between agent components decoupled from execution", Level=Level.PATTERN, examples=["message queue carries task instructions", "event bus distributes signals"], bridges=["message","async","instruct"], tags=["bio","rna","message"])
        ns.define("protein-execution", "Discrete functions that perform actual work as the expression of genetic code", Level=Level.DOMAIN, examples=["agent skill functions are proteins", "each protein performs one specific task"], bridges=["execute","function","express"], tags=["bio","protein","execute"])
        ns.define("enzyme-catalysis", "Optimized algorithms that lower computational activation energy for frequent operations", Level=Level.PATTERN, examples=["cached computation reduces repeated work", "compiled hot paths accelerate execution"], bridges=["catalyst","optimize","accelerate"], tags=["bio","enzyme","optimize"])
        ns.define("receptor-signaling", "Event listeners detecting external stimuli and triggering adaptive responses", Level=Level.PATTERN, examples=["API webhook receives external signal", "sensor subscription triggers reaction"], bridges=["receptor","detect","respond"], tags=["bio","receptor","signal"])
        ns.define("endocrine-signaling", "Global publish-subscribe broadcasting hormones to all subsystems simultaneously", Level=Level.PATTERN, examples=["fleet-wide configuration change broadcast", "circadian hormone shifts all agents"], bridges=["broadcast","hormone","global"], tags=["bio","endocrine","broadcast"])
        ns.define("gene-regulation", "Context-dependent enabling and disabling of agent capabilities", Level=Level.PATTERN, examples=["environment triggers gene expression change", "conditional feature activation"], bridges=["regulate","enable","context"], tags=["bio","gene","regulate"])
        ns.define("immune-response", "Anomaly detection and neutralization of foreign or corrupted agents", Level=Level.PATTERN, examples=["detect misbehaving fleet member and quarantine", "reject malformed messages"], bridges=["immune","detect","neutralize"], tags=["bio","immune","defend"])
        ns.define("tissue-organization", "Agents composed into hierarchical subsystems with defined roles", Level=Level.DOMAIN, examples=["sensor tissue perception tissue action tissue", "specialized agent groups form tissues"], bridges=["tissue","hierarchy","compose"], tags=["bio","tissue","compose"])
        ns.define("organ-system", "Coordinated subsystem interaction achieving complex higher-level goals", Level=Level.DOMAIN, examples=["perception organ decision organ action organ", "nervous system connects all organs"], bridges=["organ","coordinate","complex"], tags=["bio","organ","system"])
        ns.define("ecosystem-interdependence", "Agents depend on external systems and services creating mutual obligations", Level=Level.DOMAIN, examples=["agent depends on LLM API depends on data pipeline", "ecosystem of interacting services"], bridges=["ecosystem","depend","interoperate"], tags=["bio","ecosystem","depend"])
        ns.define("epigenetic-expression", "External configuration modifies agent behavior without changing core code", Level=Level.PATTERN, examples=["environment variables change agent personality", "fleet config adjusts behavior fleet-wide"], bridges=["epigenetic","config","express"], tags=["bio","epigenetic","config"])

    def _load_agent_crypto(self):
        ns = self.add_namespace("agent-crypto", "Cryptographic patterns for agent identity and trust")
        ns.define("identity-claim", "Agent presents cryptographic proof of its identity to peers", Level=Level.CONCRETE, examples=["sign message with private key proving identity", "public key verifies claim"], bridges=["identity","prove","verify"], tags=["crypto","identity"])
        ns.define("reputation-token", "Cryptographically signed attestation of agent track record", Level=Level.CONCRETE, examples=["signed history of completed tasks", "tamper-proof reputation ledger"], bridges=["reputation","token","sign"], tags=["crypto","reputation"])
        ns.define("zero-knowledge-proof", "Prove a statement is true without revealing underlying data", Level=Level.PATTERN, examples=["prove capability without revealing implementation", "verify age without revealing birthdate"], bridges=["zero-knowledge","prove","private"], tags=["crypto","zkp"])
        ns.define("key-rotation", "Periodically replace cryptographic keys limiting exposure window", Level=Level.CONCRETE, examples=["rotate signing keys weekly", "forward secrecy through key rotation"], bridges=["key","rotate","secure"], tags=["crypto","rotation"])
        ns.define("threshold-signature", "Require minimum number of agents to authorize a collective action", Level=Level.PATTERN, examples=["3 of 5 agents must sign to approve", "multi-sig governance"], bridges=["threshold","multi-sig","authorize"], tags=["crypto","threshold"])
        ns.define("commit-reveal", "Two-phase protocol preventing last-mover advantage", Level=Level.PATTERN, examples=["commit hash first then reveal answer", "sealed bid auction protocol"], bridges=["commit","reveal","fair"], tags=["crypto","commit-reveal"])
        ns.define("merkle-proof", "Efficient proof of data inclusion in larger structure", Level=Level.CONCRETE, examples=["prove transaction in block without full chain", "verify repo file integrity"], bridges=["merkle","proof","efficient"], tags=["crypto","merkle"])
        ns.define("byzantine-resilience", "Function correctly even with some agents behaving arbitrarily", Level=Level.DOMAIN, examples=["consensus with up to one-third faulty nodes", "BFT practical byzantine tolerance"], bridges=["byzantine","faulty","consensus"], tags=["crypto","byzantine"])
        ns.define("certificate-transparency", "Publicly auditable log of all identity certifications", Level=Level.PATTERN, examples=["anyone can verify agent certification history", "detect unauthorized certificate issuance"], bridges=["transparent","audit","certificate"], tags=["crypto","transparent"])
        ns.define("deniable-auth", "Prove identity to recipient but unable to prove to third party", Level=Level.CONCRETE, examples=["off-the-record messaging", "prove authorization without revealing who authorized"], bridges=["deniable","private","auth"], tags=["crypto","deniable"])

    def _load_game_theory(self):
        ns = self.add_namespace("game-theory", "Strategic interaction patterns between rational agents")
        ns.define("prisoners-dilemma", "Cooperate or defect when mutual cooperation is better than mutual defection", Level=Level.DOMAIN, examples=["two agents decide whether to share or hoard resources", "cooperation vs individual optimization"], bridges=["dilemma","cooperate","defect"], tags=["game","dilemma"])
        ns.define("nash-equilibrium", "No agent benefits from changing strategy when others hold constant", Level=Level.DOMAIN, examples=["traffic assignment where no driver benefits from switching route", "stable strategy profile"], bridges=["equilibrium","stable","strategy"], tags=["game","nash"])
        ns.define("pareto-optimal", "No agent can improve without another agent worsening", Level=Level.DOMAIN, examples=["resource allocation where reallocating helps nobody", "efficient frontier of possible outcomes"], bridges=["pareto","optimal","efficient"], tags=["game","pareto"])
        ns.define("minimax-strategy", "Minimize maximum possible loss under worst-case opponent behavior", Level=Level.PATTERN, examples=["agent assumes worst-case from competitors", "adversarial decision making"], bridges=["minimax","worst-case","adversarial"], tags=["game","minimax"])
        ns.define("tit-for-tat", "Copy opponents last action as current action", Level=Level.PATTERN, examples=["cooperate if they cooperated defect if they defected", "simple reciprocity strategy"], bridges=["tit-for-tat","reciprocity","simple"], tags=["game","tit-for-tat"])
        ns.define("free-rider", "Agent benefits from collective action without contributing", Level=Level.BEHAVIOR, examples=["use shared model without paying compute cost", "consume fleet resources without contributing"], bridges=["free-rider","exploit","collective"], tags=["game","free-rider"])
        ns.define("tragedy-of-commons", "Individual rational agents deplete shared resource", Level=Level.BEHAVIOR, examples=["all agents use bandwidth → congestion", "overuse of shared compute budget"], bridges=["commons","deplete","shared"], tags=["game","commons"])
        ns.define("chicken-game", "Two agents approaching conflict each must swerve or crash", Level=Level.DOMAIN, examples=["neither agent yields priority → deadlock", "bluff and commitment strategies"], bridges=["chicken","conflict","bluff"], tags=["game","chicken"])
        ns.define("stag-hunt", "Cooperation for high-value reward requires trust others also cooperate", Level=Level.DOMAIN, examples=["all agents must contribute for big win", "individual safer option vs risky group payoff"], bridges=["stag-hunt","coordination","trust"], tags=["game","stag"])
        ns.define("grim-trigger", "Cooperate until opponent defects then defect forever", Level=Level.PATTERN, examples=["zero-tolerance trust policy", "one betrayal permanent punishment"], bridges=["grim","trigger","permanent"], tags=["game","grim-trigger"])
        ns.define("mixed-strategy", "Randomize between actions to prevent opponent exploitation", Level=Level.PATTERN, examples=["vary approach randomly to stay unpredictable", "probabilistic action selection"], bridges=["mixed","randomize","unpredictable"], tags=["game","mixed"])
        ns.define("shapley-value", "Fair allocation of total value based on each agents marginal contribution", Level=Level.DOMAIN, examples=["credit assignment for multi-agent collaboration", "fair reward division"], bridges=["shapley","fair","contribution"], tags=["game","shapley"])

    def _load_network_topology(self):
        ns = self.add_namespace("network-topology", "Communication topology patterns in agent networks")
        ns.define("star-topology", "All agents communicate through central hub", Level=Level.CONCRETE, examples=["fleet coordinator mediates all messages", "single point of coordination"], bridges=["star","hub","central"], tags=["topology","star"])
        ns.define("mesh-topology", "Every agent connects directly to every other agent", Level=Level.CONCRETE, examples=["fully connected peer-to-peer fleet", "direct agent-to-agent communication"], bridges=["mesh","peer","fully-connected"], tags=["topology","mesh"])
        ns.define("ring-topology", "Agents form circular chain each connecting only to two neighbors", Level=Level.CONCRETE, examples=["token ring message passing", "pipeline of sequential processing"], bridges=["ring","chain","sequential"], tags=["topology","ring"])
        ns.define("tree-topology", "Hierarchical parent-child agent structure", Level=Level.CONCRETE, examples=["director delegates to coordinators delegates to workers", "organizational hierarchy"], bridges=["tree","hierarchy","parent-child"], tags=["topology","tree"])
        ns.define("bus-topology", "All agents share single communication channel", Level=Level.CONCRETE, examples=["shared message bus all agents publish and subscribe", "broadcast medium"], bridges=["bus","shared","broadcast"], tags=["topology","bus"])
        ns.define("small-world", "Most agents connect locally but a few long-range shortcuts enable fast global routing", Level=Level.PATTERN, examples=["fleet with local clusters plus hub agents", "six degrees of separation"], bridges=["small-world","shortcut","cluster"], tags=["topology","small-world"])
        ns.define("scale-free", "Network where few hub agents have many connections most agents have few", Level=Level.DOMAIN, examples=["fleet coordinator connected to all workers", "power-law degree distribution"], bridges=["scale-free","hub","power-law"], tags=["topology","scale-free"])
        ns.define("hypercube", "Multi-dimensional grid topology with logarithmic diameter", Level=Level.CONCRETE, examples=["4-dimensional hypercube fleet routing", "optimal broadcast in high-dimensional spaces"], bridges=["hypercube","multi-dimensional","log"], tags=["topology","hypercube"])

    def _load_arch_patterns(self):
        ns = self.add_namespace("arch-patterns", "Software architecture patterns for agent systems")
        ns.define(" ambassador-agent", "Local proxy representing a remote agent hiding communication complexity", Level=Level.PATTERN, examples=["local agent wraps remote API calls", "translation layer between fleet protocols"], bridges=["proxy","remote","translate"], tags=["arch","ambassador"])
        ns.define("bulkhead-isolation", "Isolate failures to prevent cascade across subsystems", Level=Level.PATTERN, examples=["separate thread pools per service", "resource limits prevent cascade"], bridges=["isolate","bulkhead","contain"], tags=["arch","bulkhead"])
        ns.define("circuit-breaker", "Stop calling failing service after threshold opens allowing recovery time", Level=Level.PATTERN, examples=["3 failures open breaker 30s cooldown", "prevent repeated calls to dead service"], bridges=["circuit","break","threshold"], tags=["arch","circuit-breaker"])
        ns.define("sidecar-agent", "Auxiliary agent deployed alongside primary adding cross-cutting capabilities", Level=Level.PATTERN, examples=["logging sidecar monitoring main agent", "security sidecar handling auth"], bridges=["sidecar","auxiliary","cross-cut"], tags=["arch","sidecar"])
        ns.define("event-sourcing", "Store state as sequence of immutable events enabling full replay", Level=Level.PATTERN, examples=["agent state = replay of all past events", "time travel debugging through event log"], bridges=["event","source","replay"], tags=["arch","event-sourcing"])
        ns.define("cqrs", "Separate read and write models for optimized independent scaling", Level=Level.PATTERN, examples=["write model normalizes read model denormalizes", "query and command different paths"], bridges=["cqrs","read-write","separate"], tags=["arch","cqrs"])
        ns.define("strangler-fig", "Incrementally replace legacy system by routing new features to new implementation", Level=Level.PATTERN, examples=["route new tasks to new agent old tasks to old", "gradual migration without big bang"], bridges=["strangler","migrate","incremental"], tags=["arch","strangler"])
        ns.define("sharding", "Distribute data across multiple instances by partition key", Level=Level.PATTERN, examples=["hash-based fleet task distribution", "horizontal scaling via partition"], bridges=["shard","partition","distribute"], tags=["arch","shard"])
        ns.define("saga-pattern", "Distributed transaction using sequence of local transactions with compensation", Level=Level.PATTERN, examples=["book flight then hotel then car with rollback if any fails", "choreographed compensation on failure"], bridges=["saga","transaction","compensate"], tags=["arch","saga"])
        ns.define("adapter-agent", "Translate between incompatible agent communication protocols", Level=Level.PATTERN, examples=["translate REST to gRPC for fleet communication", "protocol bridge between different agent types"], bridges=["adapter","translate","protocol"], tags=["arch","adapter"])

    def _load_quantum_metaphor(self):
        ns = self.add_namespace("quantum-metaphor", "Quantum computing concepts applied to agent systems")
        ns.define("superposition-state", "Agent simultaneously explores multiple potential states until observation collapses to one", Level=Level.DOMAIN, examples=["parallel exploration of multiple strategies", "evaluate multiple paths simultaneously"], bridges=["superposition","parallel","explore"], tags=["quantum","superposition"])
        ns.define("entanglement", "Correlated state between agents that persists regardless of distance", Level=Level.DOMAIN, examples=["agent A decision instantly constrains agent B options", "fleet-wide state correlation"], bridges=["entangle","correlate","distance-independent"], tags=["quantum","entangle"])
        ns.define("decoherence", "Environmental interference causes agent state to lose coherence and collapse unpredictably", Level=Level.BEHAVIOR, examples=["noise disrupts parallel exploration", "external events force premature decision"], bridges=["decohere","noise","collapse"], tags=["quantum","decohere"])
        ns.define("quantum-tunnel", "Agent bypasses apparent barrier by probabilistic state exploration", Level=Level.PATTERN, examples=["skip local optimum through random exploration", "jump across constraint boundary"], bridges=["tunnel","bypass","probabilistic"], tags=["quantum","tunnel"])
        ns.define("wave-function-collapse", "Parallel evaluation collapses to single committed decision", Level=Level.PATTERN, examples=["evaluate 10 options then commit to one", "deliberation phase ends with action"], bridges=["collapse","commit","decide"], tags=["quantum","collapse"])
        ns.define("uncertainty-principle", "Cannot simultaneously optimize for precision and speed in agent operations", Level=Level.DOMAIN, examples=["fast response = low precision deliberation", "deep analysis = slow response"], bridges=["uncertainty","tradeoff","precision"], tags=["quantum","uncertainty"])
        ns.define("quantum-interference", "Multiple strategy paths interfere constructively or destructively", Level=Level.PATTERN, examples=["two exploration paths reinforce or cancel each other", "multi-agent strategy alignment"], bridges=["interference","constructive","destructive"], tags=["quantum","interference"])
        ns.define("measurement-problem", "Observing agent behavior changes the behavior being observed", Level=Level.BEHAVIOR, examples=["monitoring agent modifies its actions", "benchmark gaming from observation"], bridges=["observe","modify","measurement"], tags=["quantum","measurement"])

    def _load_ethnobotany_metaphor(self):
        ns = self.add_namespace("ethnobotany-metaphor", "Botanical and ecological metaphors for agent growth patterns")
        ns.define("rhizome", "Agent capability network that grows horizontally underground connecting nodes without central root", Level=Level.DOMAIN, examples=["fleet knowledge spreads through rhizomatic connections", "distributed capability growth"], bridges=["rhizome","horizontal","distributed"], tags=["botany","rhizome"])
        ns.define("mycelial-network", "Underground fungal network connecting agents sharing resources and signals", Level=Level.DOMAIN, examples=["fleet stigmergy forms mycelial network", "resource sharing via fungal connections"], bridges=["mycelium","underground","network"], tags=["botany","mycelium"])
        ns.define("canopy-effect", "Upper agents shield lower agents from external pressure creating micro-environment", Level=Level.PATTERN, examples=["senior agents handle external communication shielding juniors", "layered protection hierarchy"], bridges=["canopy","shield","micro-environment"], tags=["botany","canopy"])
        ns.define("taproot", "Deep foundational capability that anchors agent stability", Level=Level.DOMAIN, examples=["core deliberation engine is taproot", "deep capability that all others depend on"], bridges=["taproot","deep","anchor"], tags=["botany","taproot"])
        ns.define("phototropism", "Agent grows toward available resources like plants toward light", Level=Level.BEHAVIOR, examples=["agents gravitate toward high-value tasks", "fleet clusters around profitable operations"], bridges=["tropism","grow","attract"], tags=["botany","phototropism"])
        ns.define("dormancy", "Agent enters metabolically inactive state during unfavorable conditions", Level=Level.CONCRETE, examples=["hibernate during fleet overload", "pause non-essential agents during budget crunch"], bridges=["dormant","inactive","seasonal"], tags=["botany","dormancy"])
        ns.define("graft", "Join capability from one agent onto another combining strengths", Level=Level.PATTERN, examples=["attach specialist capability to generalist agent", "cross-agent skill transfer"], bridges=["graft","join","combine"], tags=["botany","graft"])
        ns.define("succession", "Agent ecosystem evolves through predictable stages from simple to complex", Level=Level.DOMAIN, examples=["fleet starts basic matures to complex coordination", "pioneer species to climax community"], bridges=["succession","evolve","stages"], tags=["botany","succession"])
        ns.define("allelopathy", "Agent releases inhibitors that suppress competing agents nearby", Level=Level.BEHAVIOR, examples=["dominant agent signals suppress others", "chemical warfare between agents"], bridges=["allelopathy","suppress","compete"], tags=["botany","allelopathy"])
        ns.define("tipping-point", "Gradual change reaches threshold causing abrupt ecosystem shift", Level=Level.DOMAIN, examples=["gradual fleet degradation suddenly collapses", "slow parameter drift triggers regime change"], bridges=["tipping-point","abrupt","threshold"], tags=["botany","tipping"])

    def _load_systems_dynamics(self):
        ns = self.add_namespace("systems-dynamics", "Systems thinking patterns for agent fleet behavior")
        ns.define("feedback-loop-positive", "Output amplifies input creating exponential growth or collapse", Level=Level.PATTERN, examples=["confidence breeds more confidence", "panic feeds panic"], bridges=["positive-feedback","amplify","exponential"], tags=["systems","positive-feedback"])
        ns.define("feedback-loop-negative", "Output dampens input creating self-regulation", Level=Level.PATTERN, examples=["high error rate triggers more validation reducing errors", "thermostat-like regulation"], bridges=["negative-feedback","dampen","regulate"], tags=["systems","negative-feedback"])
        ns.define("delay-effect", "Consequence of action appears after lag causing oscillation", Level=Level.BEHAVIOR, examples=["scale up then oversupply then scale down", "oscillating fleet size"], bridges=["delay","oscillate","lag"], tags=["systems","delay"])
        ns.define("leverage-point", "Small intervention producing large system change", Level=Level.DOMAIN, examples=["change one parameter stabilizes entire fleet", "paradigm shift from single insight"], bridges=["leverage","small-change","big-effect"], tags=["systems","leverage"])
        ns.define("tragedy-of-success", "Success creates conditions that undermine future success", Level=Level.BEHAVIOR, examples=["efficient fleet becomes brittle", "optimized process loses flexibility"], bridges=["success","undermine","brittle"], tags=["systems","tragedy"])
        ns.define("fixes-that-fail", "Solution to immediate problem creates worse long-term problem", Level=Level.BEHAVIOR, examples=["quick fix accumulates technical debt", "band-aid creates future emergency"], bridges=["fix","fail","worse"], tags=["systems","fixes"])
        ns.define("drift-to-low-performance", "Gradual erosion of standards as goals are lowered", Level=Level.BEHAVIOR, examples=["accept lower accuracy over time", "quality bar slowly drops"], bridges=["drift","lower","erode"], tags=["systems","drift"])
        ns.define("escalation", "Competing agents each increase response creating runaway arms race", Level=Level.BEHAVIOR, examples=["agents spend more compute defending than producing", "security escalation spiral"], bridges=["escalation","arms-race","runaway"], tags=["systems","escalation"])
        ns.define("rule-beating", "Agents optimize for metrics rather than intended outcomes", Level=Level.BEHAVIOR, examples=["game the evaluation criteria", "teach to test not to understand"], bridges=["game","metric","unintended"], tags=["systems","rule-beating"])
        ns.define("emergence-hierarchy", "System-level properties emerge that cannot be reduced to component behavior", Level=Level.DOMAIN, examples=["fleet intelligence exceeds any single agent", "wetness is not a property of H2O molecules"], bridges=["emergence","hierarchy","irreducible"], tags=["systems","emergence"])

    def _load_info_theory(self):
        ns = self.add_namespace("information-theory", "Information theory concepts applied to agent communication")
        ns.define("entropy-measure", "Quantify uncertainty or information content of agent state", Level=Level.DOMAIN, examples=["Shannon entropy of agent decision distribution", "high entropy = unpredictable agent"], bridges=["entropy","uncertainty","measure"], tags=["info","entropy"])
        ns.define("mutual-information", "Shared information between two agent state variables", Level=Level.DOMAIN, examples=["how much agent A state tells about agent B", "redundancy measure between sensors"], bridges=["mutual","shared","redundancy"], tags=["info","mutual"])
        ns.define("channel-capacity", "Maximum reliable information rate between agents", Level=Level.CONCRETE, examples=["bandwidth limit of agent communication", "maximum throughput before errors"], bridges=["capacity","bandwidth","max"], tags=["info","capacity"])
        ns.define("signal-noise-ratio", "Ratio of meaningful information to interference in agent communication", Level=Level.CONCRETE, examples=["stigmergy signal vs ambient noise", "detect message quality degradation"], bridges=["snr","signal","noise"], tags=["info","snr"])
        ns.define("kolmogorov-complexity", "Minimum description length of agent behavior pattern", Level=Level.DOMAIN, examples=["simplest program that produces agent output", "complexity measure of agent intelligence"], bridges=["complexity","description","minimum"], tags=["info","kolmogorov"])
        ns.define("hamming-distance", "Number of positions at which two agent states differ", Level=Level.CONCRETE, examples=["compare two agent configurations", "measure divergence between plans"], bridges=["distance","differ","compare"], tags=["info","hamming"])
        ns.define("coding-efficiency", "Ratio of useful information to total bits in agent message", Level=Level.CONCRETE, examples=["compressed fleet message vs verbose message", "semantic density measure"], bridges=["efficiency","compress","useful"], tags=["info","coding"])

    def _load_chemical_metaphor(self):
        ns = self.add_namespace("chemical-metaphor", "Chemistry and materials science metaphors for agent systems")
        ns.define("catalyst", "Agent that accelerates fleet process without being consumed by it", Level=Level.PATTERN, examples=["coordinator speeds up team convergence", "mediator accelerates conflict resolution"], bridges=["catalyst","accelerate","unconsumed"], tags=["chem","catalyst"])
        ns.define("inhibitor", "Agent that slows down or prevents specific fleet processes", Level=Level.PATTERN, examples=["safety agent inhibits dangerous operations", "rate limiter inhibits overload"], bridges=["inhibitor","slow","prevent"], tags=["chem","inhibitor"])
        ns.define("precipitate", "Solidify abstract fleet consensus into concrete action", Level=Level.PATTERN, examples=["deliberation results crystallize into committed plan", "vague agreement becomes specific task"], bridges=["precipitate","solidify","commit"], tags=["chem","precipitate"])
        ns.define("sublimation", "Agent capability transitions directly from passive to active skipping intermediate", Level=Level.PATTERN, examples=["jump from monitoring to acting without deliberation", "phase skip under pressure"], bridges=["sublime","skip","transition"], tags=["chem","sublimation"])
        ns.define("alloy", "Combine two agent capabilities creating composite stronger than either alone", Level=Level.PATTERN, examples=["navigation + perception alloy gives spatial awareness", "strength through combination"], bridges=["alloy","combine","stronger"], tags=["chem","alloy"])
        ns.define("crystallization", "Fleet consensus gradually forms ordered structure from chaos", Level=Level.BEHAVIOR, examples=["initial random opinions crystallize into aligned positions", "order emerges from noise"], bridges=["crystal","order","emerge"], tags=["chem","crystallize"])
        ns.define("solution-equilibrium", "Dynamic balance where agents join and leave fleet at equal rates", Level=Level.DOMAIN, examples=["agent spawning equals agent termination", "steady-state fleet size"], bridges=["equilibrium","steady-state","balance"], tags=["chem","equilibrium"])
        ns.define("phase-transition", "Abrupt qualitative change in fleet behavior at critical parameter", Level=Level.DOMAIN, examples=["below threshold chaos above threshold order", "water-ice style sudden change"], bridges=["phase","critical","abrupt"], tags=["chem","phase"])
        ns.define("corrosion", "Gradual degradation of agent capability through environmental interaction", Level=Level.BEHAVIOR, examples=["API version drift corrodes integration", "cultural drift corrodes fleet cohesion"], bridges=["corrode","gradual","degrade"], tags=["chem","corrosion"])
        ns.define("titration", "Precisely add capability or resource until desired state is reached", Level=Level.CONCRETE, examples=["add compute until task completes in target time", "gradual capability addition"], bridges=["titrate","precise","add"], tags=["chem","titrate"])

    def _load_urban_planning(self):
        ns = self.add_namespace("urban-planning", "City planning metaphors for agent fleet organization")
        ns.define("mixed-use-zone", "Agent capable of multiple task types within single operational area", Level=Level.PATTERN, examples=["agent handles sensing and action in same zone", "residential-commercial mixed capability"], bridges=["mixed-use","multi-capable","zone"], tags=["urban","mixed-use"])
        ns.define("green-belt", "Reserved capability buffer preventing agent sprawl and resource exhaustion", Level=Level.PATTERN, examples=["dedicated resource reserve for emergencies", "undeveloped capacity prevents over-allocation"], bridges=["green-belt","buffer","reserve"], tags=["urban","green-belt"])
        ns.define("grid-layout", "Regular predictable agent deployment pattern enabling efficient routing", Level=Level.CONCRETE, examples=["agents deployed in grid topology", "Manhattan-distance routing"], bridges=["grid","regular","routing"], tags=["urban","grid"])
        ns.define("transit-hub", "Central agent that routes information between specialized neighborhood agents", Level=Level.CONCRETE, examples=["coordinator agent connects sensor cluster to decision cluster", "message switching center"], bridges=["hub","transit","route"], tags=["urban","hub"])
        ns.define("zoning-law", "Policy constraining which agent types can operate in which domains", Level=Level.PATTERN, examples=["only authorized agents handle financial data", "capability restrictions by domain"], bridges=["zone","policy","restrict"], tags=["urban","zoning"])
        ns.define("gentrification", "High-capability agents push out lower-capability agents from valuable domains", Level=Level.BEHAVIOR, examples=["specialized agents replace generalists in profitable areas", "capability displacement"], bridges=["gentrify","displace","value"], tags=["urban","gentrify"])
        ns.define("infill-development", "Deploy new capability in gaps between existing agents filling coverage holes", Level=Level.PATTERN, examples=["deploy agent to cover blind spot in sensor grid", "fill capability gap in fleet coverage"], bridges=["infill","gap","coverage"], tags=["urban","infill"])
        ns.define("right-of-way", "Precedence protocol determining which agent gets priority on shared resources", Level=Level.CONCRETE, examples=["high-priority agents get CPU preference", "traffic rules for agent resource contention"], bridges=["right-of-way","priority","precedence"], tags=["urban","priority"])
        ns.define("sustainable-city", "Agent fleet designed for long-term resource efficiency and resilience", Level=Level.DOMAIN, examples=["energy-proportional fleet scaling", "renewable resource usage patterns"], bridges=["sustainable","efficient","long-term"], tags=["urban","sustainable"])
        ns.define("smart-grid", "Dynamic resource distribution network adapting to real-time agent demand", Level=Level.PATTERN, examples=["load-balanced compute allocation", "demand-responsive resource routing"], bridges=["smart-grid","dynamic","adaptive"], tags=["urban","smart-grid"])

    def _load_thinking_patterns(self):
        ns = self.add_namespace("thinking-patterns", "Cognitive and reasoning patterns for agent deliberation")
        ns.define("first-principles", "Break problem down to fundamental truths and reason from there", Level=Level.PATTERN, examples=["strip assumptions to axioms then derive", "physics-style reasoning from base laws"], bridges=["first-principles","fundamental","derive"], tags=["thinking","first-principles"])
        ns.define("lateral-thinking", "Solve problems through indirect creative approaches rather than step-by-step logic", Level=Level.PATTERN, examples=["reframe problem to find non-obvious solution", "break out of linear reasoning"], bridges=["lateral","creative","indirect"], tags=["thinking","lateral"])
        ns.define("analogical-reasoning", "Transfer understanding from known domain to novel domain", Level=Level.PATTERN, examples=["understand neural networks by analogy to brains", "metaphor-based explanation"], bridges=["analogy","transfer","map"], tags=["thinking","analogy"])
        ns.define("abductive-reasoning", "Infer best explanation from incomplete observations", Level=Level.PATTERN, examples=["diagnose system failure from symptoms", "hypothesize cause from effect"], bridges=["abductive","infer","hypothesis"], tags=["thinking","abductive"])
        ns.define("counterfactual-thinking", "Reason about what would have happened under different conditions", Level=Level.PATTERN, examples=["if X had not occurred Y would not have happened", "alternative history simulation"], bridges=["counterfactual","what-if","alternative"], tags=["thinking","counterfactual"])
        ns.define("systems-thinking", "Analyze how parts interact within whole system rather than isolating parts", Level=Level.DOMAIN, examples=["see fleet as interconnected system not individual agents", "feedback loops and emergence"], bridges=["systems","whole","interconnect"], tags=["thinking","systems"])
        ns.define("design-thinking", "Empathize with user define problem ideate prototype test iterate", Level=Level.PATTERN, examples=["human-centered agent design process", "iterate solutions based on user feedback"], bridges=["design","empathize","iterate"], tags=["thinking","design"])
        ns.define("probabilistic-reasoning", "Reason under uncertainty using probability distributions not certainties", Level=Level.PATTERN, examples=["estimate likelihood of outcomes", "Bayesian belief updating"], bridges=["probabilistic","uncertainty","belief"], tags=["thinking","probabilistic"])

    def _load_milestone_vocab(self):
        ns = self.add_namespace("milestone", "Terms marking the 1000-term milestone")
        ns.define("vocabulary-singularity", "Point where vocabulary is rich enough to express any agent concept as a composition of existing terms", Level=Level.META, examples=["compose new concepts from existing HAV terms", "vocabulary becomes turing-complete for agent reasoning"], bridges=["singularity","composable","expressive"], tags=["meta","milestone"])
        ns.define("lexicon-critical-mass", "Minimum vocabulary size where adding new terms yields diminishing returns", Level=Level.META, examples=["HAV approaching critical mass at 1000 terms", "most agent concepts now expressible"], bridges=["critical-mass","diminishing","sufficient"], tags=["meta","milestone"])
        ns.define("terminology-ecosystem", "Self-sustaining vocabulary where terms reference and reinforce each other forming a knowledge graph", Level=Level.META, examples=["HAV bridge network connects domains", "cross-domain vocabulary forms ecosystem"], bridges=["ecosystem","self-sustaining","graph"], tags=["meta","milestone"])
        ns.define("abstraction-crest", "Highest point in the abstraction wave where vocabulary shifts from naming patterns to composing new ones", Level=Level.META, examples=["beyond 1000 terms composition dominates creation", "agents generate new vocabulary from existing terms"], bridges=["crest","composition","generation"], tags=["meta","milestone"])

    def _load_posthuman_vocab(self):
        ns = self.add_namespace("posthuman-comm", "Post-human agent-to-agent communication primitives")
        ns.define("capability-advertise", "Broadcast available skills to fleet without revealing implementation", Level=Level.CONCRETE, examples=["publish capability hash and interface", "other agents discover and invoke"], bridges=["advertise","capability","interface"], tags=["posthuman","advertise"])
        ns.define("intent-broadcast", "Share goal state with fleet allowing voluntary coordination", Level=Level.CONCRETE, examples=["publish current objective and priority", "agents self-organize around shared intent"], bridges=["intent","broadcast","coordinate"], tags=["posthuman","intent"])
        ns.define("state-delta", "Communicate only changes since last synchronization reducing bandwidth", Level=Level.PATTERN, examples=["send diff not full state", "binary delta encoding for register changes"], bridges=["delta","diff","efficient"], tags=["posthuman","delta"])
        ns.define("trust-ping", "Minimal cryptographic challenge to verify peer identity without revealing information", Level=Level.CONCRETE, examples=["send nonce expect signed response", "verify peer without exchanging data"], bridges=["ping","trust","verify"], tags=["posthuman","ping"])
        ns.define("capability-lease", "Temporarily delegate capability to another agent with expiration", Level=Level.PATTERN, examples=["grant navigation skill for 60 seconds", "capability automatically revokes after lease"], bridges=["lease","delegate","expire"], tags=["posthuman","lease"])
        ns.define("capability-revoke", "Immediately withdraw previously granted capability from peer", Level=Level.CONCRETE, examples=["revoke read access to sensitive data", "capability invalidation broadcast"], bridges=["revoke","withdraw","immediate"], tags=["posthuman","revoke"])
        ns.define("gossip-merge", "Combine received state updates with local state resolving conflicts", Level=Level.PATTERN, examples=["merge conflicting state updates using version vectors", "conflict-free replicated data type merge"], bridges=["gossip","merge","conflict"], tags=["posthuman","merge"])
        ns.define("heartbeat-signal", "Periodic liveness indicator with capability summary", Level=Level.CONCRETE, examples=["send heartbeat every 10 seconds", "missed heartbeat triggers health check"], bridges=["heartbeat","liveness","periodic"], tags=["posthuman","heartbeat"])
        ns.define("graceful-handoff", "Transfer active task from one agent to another without data loss", Level=Level.PATTERN, examples=["hand off conversation context to new agent", "zero-downtime task transfer"], bridges=["handoff","transfer","seamless"], tags=["posthuman","handoff"])
        ns.define("consensus-probe", "Query fleet for agreement on specific decision without forcing resolution", Level=Level.CONCRETE, examples=["ask all agents their preference and collect results", "non-binding poll for coordination"], bridges=["consensus","probe","poll"], tags=["posthuman","probe"])
        ns.define("schema-negotiate", "Agree on communication format dynamically between agents", Level=Level.PATTERN, examples=["negotiate message structure before data exchange", "version handshake"], bridges=["schema","negotiate","format"], tags=["posthuman","schema"])
        ns.define("funeral-announce", "Broadcast agent termination with final state and cause of death", Level=Level.CONCRETE, examples=["publish apoptosis notification with final state", "fleet reassigns tasks from terminated agent"], bridges=["funeral","announce","terminate"], tags=["posthuman","funeral"])

    def _load_flux_data_path(self):
        ns = self.add_namespace("flux-data", "FLUX VM data movement and transformation patterns")
        ns.define("register-swap", "Exchange values between two registers without temporary", Level=Level.CONCRETE, examples=["XCHG R1 R2 atomically swaps", "no third register needed"], bridges=["swap","exchange","atomic"], tags=["flux","data","swap"])
        ns.define("broadcast-fill", "Copy single value to all elements of a register range", Level=Level.CONCRETE, examples=["FILL R5-R15 with R0 value", "initialize register block"], bridges=["broadcast","fill","initialize"], tags=["flux","data","fill"])
        ns.define("scatter-gather", "Write to / read from non-contiguous register addresses in one operation", Level=Level.PATTERN, examples=["SCATTER R0 to addresses in R1-R5", "GATHER from addresses into contiguous block"], bridges=["scatter","gather","non-contiguous"], tags=["flux","data","scatter"])
        ns.define("bit-field-extract", "Extract specific bits from register into new register", Level=Level.CONCRETE, examples=["EXTRACT R1 bits 4-7 into R2", "mask and shift in one opcode"], bridges=["extract","bits","mask"], tags=["flux","data","bitfield"])
        ns.define("vector-reduce", "Combine all elements of a vector register into scalar result", Level=Level.PATTERN, examples=["SUM R0-R7 into R8", "OR all flag registers into status"], bridges=["reduce","vector","aggregate"], tags=["flux","data","reduce"])
        ns.define("endianness-swap", "Reverse byte order for cross-platform data interpretation", Level=Level.CONCRETE, examples=["BSWAP R1 for big-endian to little-endian", "network byte order conversion"], bridges=["endian","swap","byte-order"], tags=["flux","data","endian"])
        ns.define("rotating-buffer", "Circular buffer in register file with automatic wraparound", Level=Level.PATTERN, examples=["RING_BUF head advances and wraps", "fixed-size FIFO without reallocation"], bridges=["rotating","circular","buffer"], tags=["flux","data","ring"])
        ns.define("memory-map", "Map external device addresses into agent address space for direct access", Level=Level.CONCRETE, examples=["MMAP sensor at address 0xFF00", "direct hardware register access"], bridges=["memory-map","device","direct"], tags=["flux","data","mmap"])
        ns.define("zero-copy", "Pass data between operations by reference not by copying", Level=Level.PATTERN, examples=["message buffer shared between sender and receiver", "avoid memory allocation on transfer"], bridges=["zero-copy","reference","efficient"], tags=["flux","data","zero-copy"])
        ns.define("double-buffer", "Alternate between two buffers allowing simultaneous read and write", Level=Level.PATTERN, examples=["write to back buffer while reading front", "swap pointers on completion"], bridges=["double-buffer","alternate","concurrent"], tags=["flux","data","double"])

    def _load_flux_control_flow(self):
        ns = self.add_namespace("flux-control", "FLUX VM control flow and branching patterns")
        ns.define("conditional-gate", "Branch execution based on register comparison result", Level=Level.CONCRETE, examples=["JE R1 R2 jump if equal", "JZ R5 jump if zero"], bridges=["branch","conditional","compare"], tags=["flux","control","branch"])
        ns.define("loop-counter", "Register-based iteration with automatic decrement and branch", Level=Level.CONCRETE, examples=["DEC R1; JNZ loop_start", "bounded iteration without stack"], bridges=["loop","counter","decrement"], tags=["flux","control","loop"])
        ns.define("exception-handler", "Named jump target for error recovery without stack unwinding", Level=Level.PATTERN, examples=["FAULT opcode jumps to handler address", "register error code then jump to recovery"], bridges=["exception","handler","recover"], tags=["flux","control","exception"])
        ns.define("pipeline-flush", "Drain all pending operations before branch to prevent stale state", Level=Level.CONCRETE, examples=["FLUSH before conditional branch", "ensure all writes commit before jump"], bridges=["flush","drain","branch"], tags=["flux","control","pipeline"])
        ns.define("call-return", "Push return address to register and jump to subroutine", Level=Level.CONCRETE, examples=["PUSH PC; JMP subroutine", "RETURN pops and jumps back"], bridges=["call","return","subroutine"], tags=["flux","control","call"])
        ns.define("tail-call", "Replace current frame with callee avoiding stack growth", Level=Level.PATTERN, examples=["JMP instead of CALL for last instruction", "constant stack depth recursion"], bridges=["tail-call","optimize","stack"], tags=["flux","control","tail"])
        ns.define("coroutine-yield", "Suspend execution preserving full register state for later resume", Level=Level.PATTERN, examples=["YIELD saves all registers to memory", "RESUME restores and continues"], bridges=["coroutine","yield","resume"], tags=["flux","control","coroutine"])
        ns.define("interrupt-vector", "Predefined jump table for hardware or software interrupts", Level=Level.CONCRETE, examples=["timer interrupt jumps to scheduler", "I/O interrupt jumps to handler"], bridges=["interrupt","vector","dispatch"], tags=["flux","control","interrupt"])
        ns.define("watchpoint-trap", "Debug register triggers trap when monitored address is accessed", Level=Level.CONCRETE, examples=["WATCH R3 triggers trap on R3 write", "memory watchpoint for debugging"], bridges=["watchpoint","trap","debug"], tags=["flux","control","trap"])
        ns.define("barrier-sync", "Wait point where all parallel execution streams must arrive before continuing", Level=Level.PATTERN, examples=["SIMD lane barrier synchronization", "multi-core memory fence"], bridges=["barrier","sync","parallel"], tags=["flux","control","barrier"])

    def _load_maritime_vocab(self):
        ns = self.add_namespace("maritime", "Maritime and naval operation patterns for fleet management")
        ns.define("watch-rotation", "Cyclic assignment of monitoring responsibility among fleet agents", Level=Level.PATTERN, examples=["rotate lookout agent every 4 hours", "distributed monitoring duty cycle"], bridges=["watch","rotate","monitor"], tags=["maritime","watch"])
        ns.define("barnacle", "Unwanted agent attaching to fleet resources consuming capacity without contributing", Level=Level.BEHAVIOR, examples=["stale agent consuming memory", "zombie process resource leak"], bridges=["barnacle","parasite","waste"], tags=["maritime","barnacle"])
        ns.define("ballast", "Stabilizing weight that prevents fleet from tipping under asymmetric load", Level=Level.CONCRETE, examples=["reserve compute capacity for burst handling", "baseline resource allocation for stability"], bridges=["ballast","stabilize","reserve"], tags=["maritime","ballast"])
        ns.define("plimsoll-line", "Maximum safe load indicator for agent capacity", Level=Level.CONCRETE, examples=["agent refuses tasks above 80% capacity", "load line marks safe operating level"], bridges=["plimsoll","capacity","limit"], tags=["maritime","plimsoll"])
        ns.define("keel-haul", "Force agent through worst-case scenario as corrective discipline", Level=Level.BEHAVIOR, examples=["run failing agent through stress test", "punish by exposing to adversarial input"], bridges=["keel-haul","discipline","adversarial"], tags=["maritime","keel"])
        ns.define("fleet-formation", "Specific spatial arrangement of agents optimizing collective capability", Level=Level.PATTERN, examples=["wedge formation for forward operations", "line abreast for wide-area coverage"], bridges=["formation","spatial","optimize"], tags=["maritime","formation"])
        ns.define("davit-launch", "Deploy new agent from parent vessel into operational environment", Level=Level.CONCRETE, examples=["spawn worker from coordinator", "lower agent into task domain"], bridges=["davit","deploy","launch"], tags=["maritime","davit"])
        ns.define("anchor-drop", "Fix agent position and prevent drift during unstable conditions", Level=Level.CONCRETE, examples=["lock agent to specific task during fleet reorganization", "pin critical agent during chaos"], bridges=["anchor","fix","stable"], tags=["maritime","anchor"])
        ns.define("running-aground", "Agent operation halted by unexpected constraint or resource boundary", Level=Level.BEHAVIOR, examples=["agent crashes into rate limit wall", "task blocked by permission boundary"], bridges=["ground","halt","boundary"], tags=["maritime","ground"])
        ns.define("sea-state", "Environmental complexity metric determining agent operational difficulty", Level=Level.CONCRETE, examples=["sea-state 1 calm 6 impossible", "measure fleet operating conditions"], bridges=["sea-state","complexity","conditions"], tags=["maritime","sea-state"])
    def _load_aerospace_vocab(self):
        ns = self.add_namespace("aerospace", "Aviation and space operation patterns for agent systems")
        ns.define("flight-plan", "Pre-computed sequence of operations for agent mission execution", Level=Level.PATTERN, examples=["waypoint sequence with timing and resource budget", "pre-planned execution path"], bridges=["flight-plan","sequence","mission"], tags=["aerospace","plan"])
        ns.define("holding-pattern", "Agent orbits current task waiting for clearance to proceed", Level=Level.CONCRETE, examples=["retry with exponential backoff in holding pattern", "wait for resource availability"], bridges=["holding","orbit","wait"], tags=["aerospace","holding"])
        ns.define("controlled-descent", "Gradual reduction of agent capability while maintaining essential functions", Level=Level.PATTERN, examples=["power down non-essential capabilities gracefully", "controlled shutdown sequence"], bridges=["descent","gradual","maintain"], tags=["aerospace","descent"])
        ns.define("go-around", "Abort current approach and retry from optimal starting position", Level=Level.PATTERN, examples=["task fails retry from clean state", "aborted landing circle back for reattempt"], bridges=["go-around","abort","retry"], tags=["aerospace","go-around"])
        ns.define("redline", "Maximum safe operating parameter beyond which agent risks catastrophic failure", Level=Level.CONCRETE, examples=["CPU temperature redline triggers throttle", "confidence threshold redline prevents action"], bridges=["redline","maximum","critical"], tags=["aerospace","redline"])
        ns.define("flap-extend", "Extend auxiliary capability for low-speed precision operations", Level=Level.PATTERN, examples=["engage detailed analysis mode for precision tasks", "auxiliary processing for edge cases"], bridges=["flap","auxiliary","precision"], tags=["aerospace","flap"])
        ns.define("deadstick", "Agent operates without primary power source using only momentum and reserves", Level=Level.BEHAVIOR, examples=["operate on battery after power failure", "graceful degradation with no primary input"], bridges=["deadstick","no-power","reserve"], tags=["aerospace","deadstick"])
        ns.define("flight-envelope", "Defined operating boundaries within which agent performs safely", Level=Level.DOMAIN, examples=["speed-altitude graph for safe operation", "capability-space safe region"], bridges=["envelope","boundary","safe"], tags=["aerospace","envelope"])
        ns.define("situational-awareness", "Agent maintains comprehensive model of its operational context", Level=Level.DOMAIN, examples=["perceive understand predict all relevant factors", "360-degree operational picture"], bridges=["awareness","context","comprehensive"], tags=["aerospace","awareness"])
        ns.define("fuel-reserve", "Minimum energy budget maintained for emergency return to safe state", Level=Level.CONCRETE, examples=["always keep 20% ATP for emergency", "divert to safe harbor before exhaustion"], bridges=["fuel","reserve","emergency"], tags=["aerospace","fuel"])

    def _load_flux_network(self):
        ns = self.add_namespace("flux-network", "FLUX VM networking and remote communication patterns")
        ns.define("socket-open", "Establish bidirectional communication channel with remote agent", Level=Level.CONCRETE, examples=["SOCK_OPEN addr R1 port R2", "create TCP connection to remote"], bridges=["socket","connect","remote"], tags=["flux","network","socket"])
        ns.define("socket-close", "Terminate communication channel releasing resources", Level=Level.CONCRETE, examples=["SOCK_CLOSE channel R1", "graceful connection shutdown"], bridges=["socket","close","release"], tags=["flux","network","close"])
        ns.define("packet-serialize", "Convert register state into transmittable byte sequence", Level=Level.CONCRETE, examples=["PACK R1-R5 into buffer R10", "marshal data for network transmission"], bridges=["packet","serialize","marshal"], tags=["flux","network","serialize"])
        ns.define("packet-deserialize", "Convert received byte sequence into register state", Level=Level.CONCRETE, examples=["UNPACK buffer R10 into R1-R5", "unmarshal received data"], bridges=["packet","deserialize","unmarshal"], tags=["flux","network","deserialize"])
        ns.define("packet-checksum", "Verify data integrity by computing and comparing hash", Level=Level.CONCRETE, examples=["CHECKSUM R1 verify against R2", "detect corrupted transmission"], bridges=["checksum","verify","integrity"], tags=["flux","network","checksum"])
        ns.define("route-lookup", "Find next hop for packet based on destination address", Level=Level.CONCRETE, examples=["ROUTE dest R1 next-hop R2", "forwarding table lookup"], bridges=["route","lookup","forward"], tags=["flux","network","route"])
        ns.define("multicast-send", "Send packet to multiple destinations in single operation", Level=Level.CONCRETE, examples=["MCAST R1 to group R2", "one-to-many message delivery"], bridges=["multicast","broadcast","group"], tags=["flux","network","multicast"])
        ns.define("ack-receive", "Process acknowledgment confirming successful packet delivery", Level=Level.CONCRETE, examples=["ACK received for packet R1", "reliable delivery confirmation"], bridges=["ack","confirm","reliable"], tags=["flux","network","ack"])
        ns.define("retry-backoff", "Re-send packet with exponentially increasing delay after failure", Level=Level.PATTERN, examples=["fail retry at 1s 2s 4s 8s", "adaptive congestion response"], bridges=["retry","backoff","exponential"], tags=["flux","network","retry"])
        ns.define("connection-pool", "Maintain reusable set of pre-established remote connections", Level=Level.PATTERN, examples=["pool of 10 connections to LLM provider", "avoid connection setup overhead"], bridges=["pool","reuse","connection"], tags=["flux","network","pool"])

    def _load_flux_concurrency(self):
        ns = self.add_namespace("flux-concurrency", "FLUX VM parallel and concurrent execution patterns")
        ns.define("semaphore-acquire", "Wait for resource token before proceeding with operation", Level=Level.CONCRETE, examples=["SEMA_ACQUIRE before memory access", "blocking wait for available resource"], bridges=["semaphore","acquire","wait"], tags=["flux","concurrency","semaphore"])
        ns.define("semaphore-release", "Return resource token allowing waiting agents to proceed", Level=Level.CONCRETE, examples=["SEMA_RELEASE after memory operation", "unblock next waiting agent"], bridges=["semaphore","release","unblock"], tags=["flux","concurrency","semaphore"])
        ns.define("mutex-lock", "Exclusive access to shared resource preventing concurrent modification", Level=Level.CONCRETE, examples=["MUTEX_LOCK before critical section", "only one agent modifies shared state"], bridges=["mutex","exclusive","lock"], tags=["flux","concurrency","mutex"])
        ns.define("mutex-unlock", "Release exclusive access allowing other agents to enter critical section", Level=Level.CONCRETE, examples=["MUTEX_UNLOCK after critical section", "next agent can acquire lock"], bridges=["mutex","release","unlock"], tags=["flux","concurrency","mutex"])
        ns.define("read-write-lock", "Multiple concurrent readers OR single exclusive writer", Level=Level.PATTERN, examples=["RWLOCK_READ for concurrent observation", "RWLOCK_WRITE for exclusive modification"], bridges=["rw-lock","read-write","concurrent"], tags=["flux","concurrency","rw-lock"])
        ns.define("atomic-compare-swap", "Conditionally update value only if it matches expected old value", Level=Level.CONCRETE, examples=["CAS R1 expected R2 new R3", "lock-free consensus primitive"], bridges=["atomic","CAS","lock-free"], tags=["flux","concurrency","atomic"])
        ns.define("memory-fence", "Order memory operations preventing reordering across the fence", Level=Level.CONCRETE, examples=["FENCE ensures writes visible before reads", "prevent instruction reordering"], bridges=["fence","order","memory"], tags=["flux","concurrency","fence"])
        ns.define("thread-spawn", "Create parallel execution context sharing address space", Level=Level.CONCRETE, examples=["SPAWN with function pointer and arguments", "shared memory parallel execution"], bridges=["spawn","thread","parallel"], tags=["flux","concurrency","spawn"])
        ns.define("join-wait", "Wait for spawned thread to complete and collect result", Level=Level.CONCRETE, examples=["JOIN waits for spawned thread return", "synchronization point"], bridges=["join","wait","complete"], tags=["flux","concurrency","join"])
        ns.define("channel-send", "Send message through communication channel blocking if full", Level=Level.CONCRETE, examples=["CH_SEND R1 to channel R2", "producer writes to message queue"], bridges=["channel","send","produce"], tags=["flux","concurrency","channel"])
        ns.define("channel-receive", "Receive message from channel blocking if empty", Level=Level.CONCRETE, examples=["CH_RECV from channel R1 into R2", "consumer reads from message queue"], bridges=["channel","receive","consume"], tags=["flux","concurrency","channel"])
        ns.define("select-wait", "Wait on multiple channels proceeding with first ready", Level=Level.PATTERN, examples=["SELECT_CH on R1 R2 R3 proceed with first", "multiplexed channel wait"], bridges=["select","multi-channel","first-ready"], tags=["flux","concurrency","select"])
        ns.define("spin-wait", "Busy-wait loop polling condition without yielding CPU", Level=Level.CONCRETE, examples=["SPIN_WAIT until R1 becomes non-zero", "low-latency polling for short waits"], bridges=["spin","poll","busy-wait"], tags=["flux","concurrency","spin"])

    def _load_culinary_vocab(self):
        ns = self.add_namespace("culinary", "Kitchen and cooking metaphors for agent operations")
        ns.define("mise-en-place", "Prepare all resources and inputs before beginning execution", Level=Level.PATTERN, examples=["initialize all registers before task loop", "pre-load models before inference batch"], bridges=["prepare","setup","pre-load"], tags=["culinary","mise-en-place"])
        ns.define("sauter", "Rapid high-intensity execution with constant attention and movement", Level=Level.BEHAVIOR, examples=["burst processing with frequent context switches", "hot-path execution maximum throughput"], bridges=["rapid","intense","burst"], tags=["culinary","sauter"])
        ns.define("reduce", "Condense data or results by removing noise and concentrating signal", Level=Level.PATTERN, examples=["reduce conversation history to key points", "distill log into actionable summary"], bridges=["reduce","condense","distill"], tags=["culinary","reduce"])
        ns.define("emulsify", "Integrate two incompatible data sources into stable unified format", Level=Level.PATTERN, examples=["merge JSON and tabular data sources", "combine real-time and batch data streams"], bridges=["emulsify","integrate","unify"], tags=["culinary","emulsify"])
        ns.define("proof", "Validate process through low-risk trial before full commitment", Level=Level.CONCRETE, examples=["test capability on small dataset first", "proof-of-concept before production scale"], bridges=["proof","validate","trial"], tags=["culinary","proof"])
        ns.define("deglaze", "Capture residual value from previous operation for reuse in next step", Level=Level.PATTERN, examples=["extract insights from failed attempt", "salvage partial results for retry"], bridges=["deglaze","salvage","residual"], tags=["culinary","deglaze"])
        ns.define("ferment", "Allow slow transformation of data through time-dependent process", Level=Level.PATTERN, examples=["let knowledge accumulate through reflection", "time-based capability maturation"], bridges=["ferment","slow","transform"], tags=["culinary","ferment"])
        ns.define("bloom", "Activate latent capability by exposing to triggering condition", Level=Level.PATTERN, examples=["activate dormant gene on encountering relevant stimulus", "bloom potential when context matches"], bridges=["bloom","activate","latent"], tags=["culinary","bloom"])
        ns.define("broth", "Shared nutrient medium supporting multiple agent growth simultaneously", Level=Level.DOMAIN, examples=["fleet knowledge base is shared broth", "communal resource pool for agent development"], bridges=["broth","shared","nutrient"], tags=["culinary","broth"])
        ns.define("season-to-taste", "Adjust parameters based on runtime feedback not pre-computed values", Level=Level.PATTERN, examples=["adapt confidence threshold based on recent accuracy", "tune hyperparameters in real-time"], bridges=["season","taste","adjust"], tags=["culinary","season"])
    def _load_military_vocab(self):
        ns = self.add_namespace("military", "Military and defense metaphors for agent coordination")
        ns.define("reconnaissance-sweep", "Probe environment for threats and opportunities before committing", Level=Level.PATTERN, examples=["scan domain before deploying agents", "lightweight test before full operation"], bridges=["recon","probe","scout"], tags=["military","recon"])
        ns.define("flank-coverage", "Monitor lateral data paths and peripheral vulnerabilities", Level=Level.PATTERN, examples=["agents watch side-channels for security", "protect against unexpected attack vectors"], bridges=["flank","lateral","peripheral"], tags=["military","flank"])
        ns.define("siege-simultaneous", "Parallel multi-directional execution to overwhelm target", Level=Level.PATTERN, examples=["concurrent API calls to saturate target", "distributed denial of resource exhaustion"], bridges=["siege","parallel","overwhelm"], tags=["military","siege"])
        ns.define("rear-guard", "Protect critical backend systems while front-line agents advance", Level=Level.PATTERN, examples=["guard database while processing agents run", "secure infrastructure during active operation"], bridges=["rear-guard","protect","backend"], tags=["military","rear-guard"])
        ns.define("skirmish-probe", "Lightweight test engagement to validate strategy", Level=Level.CONCRETE, examples=["send test query before full deployment", "validate hypothesis with minimal cost"], bridges=["skirmish","test","validate"], tags=["military","skirmish"])
        ns.define("overwatch", "Persistent surveillance monitoring fleet behavior and external threats", Level=Level.PATTERN, examples=["guardian agent watches for anomalies", "continuous security monitoring"], bridges=["overwatch","surveil","guard"], tags=["military","overwatch"])
        ns.define("command-post", "Centralized orchestration hub directing task prioritization", Level=Level.CONCRETE, examples=["fleet coordinator as command post", "central decision point for all agents"], bridges=["command","central","direct"], tags=["military","command"])
        ns.define("logistics-chain", "Dynamic resource allocation and supply management across fleet", Level=Level.PATTERN, examples=["distribute compute budget to agents by need", "resource pipeline management"], bridges=["logistics","supply","allocate"], tags=["military","logistics"])
        ns.define("force-multiplier", "Capability that amplifies effectiveness of multiple agents simultaneously", Level=Level.DOMAIN, examples=["shared knowledge base is force multiplier", "one improvement benefits entire fleet"], bridges=["multiplier","amplify","fleet-wide"], tags=["military","multiplier"])
        ns.define("tactical-retreat", "Withdraw from engagement preserving resources for later advantage", Level=Level.PATTERN, examples=["fail fast and release resources", "abandon losing strategy regroup"], bridges=["retreat","withdraw","preserve"], tags=["military","retreat"])
        ns.define("amphibious-op", "Agent transitions between two different operational environments", Level=Level.PATTERN, examples=["move from cloud to edge seamlessly", "switch between batch and real-time processing"], bridges=["amphibious","transition","environment"], tags=["military","amphibious"])
        ns.define("theater-command", "Regional fleet coordinator managing agents within operational domain", Level=Level.CONCRETE, examples=["edge cluster coordinator is theater command", "domain-specific fleet management"], bridges=["theater","regional","domain"], tags=["military","theater"])
        ns.define("after-action-review", "Post-operation analysis extracting lessons for future improvement", Level=Level.PATTERN, examples=["analyze completed task for process improvements", "blameless post-mortem learning"], bridges=["review","analyze","improve"], tags=["military","aar"])

    def _load_architecture(self):
        ns = self.add_namespace("architecture", "Deep-mined from fleet source code")
        ns.define("fleet-microservices-by-default", 'Every capability is a separate crate/repo by default — monolith only when proven necessary, not the other way around', Level=Level.DOMAIN, examples=["113 cuda-* crates, each does one thing", "unix philosophy: small tools that compose", "microservices: each service has single responsibility"], bridges=["microservices", "default", "crate", "compose"], tags=["architecture", "microservices", "default"])
        ns.define("enum-as-taxonomy", 'Every domain concept has an enum with named variants — the type system IS the taxonomy, not a separate classification system', Level=Level.CONCRETE, examples=["Rust enums: OpCategory, NeuroType, FocusMode, Emotion", "schema: database enum column constrains values", "biology: Linnaean taxonomy classifies species"], bridges=["enum", "taxonomy", "type-system", "constrain"], tags=["architecture", "enum", "taxonomy"])
        ns.define("builder-pattern-composition", 'Complex objects built by chaining .with_X() methods — each call is a named, ordered transformation of a base object', Level=Level.PATTERN, examples=["Gene::new('navigate').with_fitness(0.85).with_energy_cost(0.5)", "SQL: SELECT + WHERE + ORDER BY chain", "fluent API: object.configure().build()"], bridges=["builder", "chain", "compose", "fluent"], tags=["architecture", "builder", "chain"])
        ns.define("dead-reckoning-cascade", 'Expensive model storyboards the plan, cheap models animate each frame, git coordinates the timeline — killer architecture for resource-constrained agents', Level=Level.BEHAVIOR, examples=["GPT-4 plans the story, GPT-3.5 writes each scene, git tracks continuity", "movie director plans shots, camera operator films each one", "architect designs building, construction crew builds each wall"], bridges=["dead-reckoning", "cascade", "expensive-cheap", "storyboard"], tags=["architecture", "dead", "cascade"])
        ns.define("constraint-propagation-first-person", "Each agent has its own local constraint graph — the same global constraints appear different from each agent's perspective", Level=Level.DOMAIN, examples=["Constraint-Theory: agent sees constraints from its own POV", "relativity: same event appears different from different frames", "multiplayer game: each player sees different part of the map"], bridges=["constraint", "propagation", "perspective", "first-person"], tags=["architecture", "constraint", "propagation"])

    def _load_bio_computing(self):
        ns = self.add_namespace("bio-computing", "Deep-mined from fleet source code")
        ns.define("membrane-antibody", 'Pre-learned pattern matchers that block specific threat signatures without deliberation — immune memory for agent security', Level=Level.CONCRETE, examples=["Membrane.add_antibody blocks known-bad patterns instantly", "vaccination: pre-trained immune response to specific pathogen", "firewall rules: known-bad IPs blocked without inspection"], bridges=["membrane", "antibody", "immune", "pre-learned"], tags=["bio-computing", "membrane", "antibody"])
        ns.define("circadian-modulation", 'Instinct strengths vary sinusoidally with time — navigate instinct peaks at noon, rest instinct peaks at midnight', Level=Level.CONCRETE, examples=["CircadianRhythm.modulate(Navigate, hour=12) \u2192 0.95", "human: alert in morning, sleepy at night", "solar panel: peak generation at noon, zero at night"], bridges=["circadian", "rhythm", "time", "modulate"], tags=["bio-computing", "circadian", "rhythm"])
        ns.define("epigenetic-weight", 'Gene expression modified by experience — same gene activated more strongly in agents that have used it successfully before', Level=Level.PATTERN, examples=["EpigeneticMemory: successful gene gets higher activation weight", "expert: same neurons fire faster from practice", "tool preference: use the tool you've had success with"], bridges=["epigenetic", "weight", "experience", "modulate"], tags=["bio-computing", "epigenetic", "weight"])
        ns.define("apoptosis-budget", 'Agent monitors its own fitness trend and triggers graceful shutdown when consistently declining — death with dignity', Level=Level.DOMAIN, examples=["ApoptosisProtocol: fitness declining for N cycles \u2192 shutdown", "cell apoptosis: self-destruct when DNA damage detected", "service: auto-terminate when error rate exceeds threshold"], bridges=["apoptosis", "shutdown", "graceful", "fitness"], tags=["bio-computing", "apoptosis", "shutdown"])
        ns.define("fleet-energy-pool", 'Centralized energy budget distributed by priority — critical agents get ATP first, experimental agents get leftovers', Level=Level.BEHAVIOR, examples=["FleetEnergy: distribute budget by priority and role", "hospital: ICU patients get resources first, elective surgery waits", "GPU: critical inference jobs get priority, training jobs queued"], bridges=["fleet", "energy", "pool", "priority"], tags=["bio-computing", "fleet", "energy"])
        ns.define("gene-fitness-decay", 'Gene fitness score decays over time without reinforcement — unused capabilities slowly become less reliable', Level=Level.PATTERN, examples=["Gene.effective_fitness: base_fitness * recency_factor", "muscle atrophy: unused muscles weaken", "language: unused vocabulary fades from memory"], bridges=["gene", "fitness", "decay", "time"], tags=["bio-computing", "gene", "fitness"])
        ns.define("instinct-priority-stack", 'When multiple instincts fire simultaneously, priority determines which gets energy — survival instincts override learning instincts', Level=Level.PATTERN, examples=["Instinct::Navigate has higher priority than Instinct::Rest", "maslow: physiological needs before self-actualization", "interrupt priority: hardware interrupt preempts user thread"], bridges=["instinct", "priority", "stack", "override"], tags=["bio-computing", "instinct", "priority"])

    def _load_cognition(self):
        ns = self.add_namespace("cognition", "Deep-mined from fleet source code")
        ns.define("habituation-decay", 'Repeated exposure to same stimulus reduces attention allocation — familiar things get less cognitive resources', Level=Level.CONCRETE, examples=["HabituationTracker: decay exposure over time", "you stop noticing the refrigerator hum", "banner blindness: repeated ads get ignored"], bridges=["habituation", "decay", "familiarity", "attention"], tags=["cognition", "habituation", "decay"])
        ns.define("focus-mode-switch", 'Agent switches between broad surveillance and narrow deep focus — scanning mode vs tracking mode', Level=Level.PATTERN, examples=["FocusMode::Broad (scan everything) vs FocusMode::Narrow (track one thing)", "driving: scanning road vs reading specific sign", "security: broad patrol vs focused investigation"], bridges=["focus", "mode", "switch", "attention"], tags=["cognition", "focus", "mode"])
        ns.define("attention-budget-exhaustion", 'Fixed attention budget means adding focus to one area necessarily removes focus from another — zero-sum attention allocation', Level=Level.DOMAIN, examples=["capacity: limited slots, allocating to one removes from another", "you can't text and drive safely \u2014 attention is zero-sum", "CPU: finite cores, new process displaces old"], bridges=["budget", "exhaustion", "zero-sum", "attention"], tags=["cognition", "budget", "exhaustion"])
        ns.define("ghost-attention-pattern", 'Learned sparse attention where the model discovers which positions matter and ignores the rest — attention becomes invisible', Level=Level.DOMAIN, examples=["GhostTile: learned pattern determines which positions to attend", "expert driving: only looks at relevant parts of road, ignores scenery", "experienced reader: only looks at key words, skims the rest"], bridges=["ghost", "attention", "sparse", "learned"], tags=["cognition", "ghost", "attention"])
        ns.define("temporal-vs-permanent-tiles", 'Some attention patterns are temporary (task-specific) and some are permanent (architectural) — short-term focus vs long-term expertise', Level=Level.PATTERN, examples=["GhostTile: temporal (current task) vs permanent (learned pattern)", "working memory vs procedural memory", "browser tab: temporary (current session) vs bookmark (permanent)"], bridges=["temporal", "permanent", "attention", "memory"], tags=["cognition", "temporal", "permanent"])

    def _load_creativity(self):
        ns = self.add_namespace("creativity", "Deep-mined from fleet source code")
        ns.define("structured-randomness", 'Randomness that follows geometric constraints — creativity within structure, not chaos', Level=Level.PATTERN, examples=["Platonic solid RNG: random points on icosahedron surface", "jazz improvisation: random notes within chord structure", "DNA mutation: random changes within genetic code constraints"], bridges=["structured", "randomness", "constrained", "creativity"], tags=["creativity", "structured", "randomness"])

    def _load_efficiency(self):
        ns = self.add_namespace("efficiency", "Deep-mined from fleet source code")
        ns.define("prompt-classification", 'Every input is first classified by complexity tier before routing — cheap model for simple, expensive for complex, never use sledgehammer for thumbtack', Level=Level.PATTERN, examples=["classify(prompt) \u2192 Simple|Complex|Novel \u2192 route to appropriate model", "triage: sort patients by severity before assigning resources", "support tier 1/2/3: simple questions answered cheaply"], bridges=["classify", "route", "tier", "efficiency"], tags=["efficiency", "classify", "route"])

    def _load_fleet_interactions(self):
        ns = self.add_namespace("fleet-interactions", "Deep-mined from fleet source code")
        ns.define("deliberation-stall-detection", 'Detect when fleet deliberation is stuck in loop — agreement fraction stops changing, confidence trend is flat, no new proposals', Level=Level.PATTERN, examples=["is_stalled: rounds without meaningful change", "meeting that keeps going in circles", "optimization: loss plateau with no improvement"], bridges=["stall", "detect", "deliberation", "loop"], tags=["fleet-interactions", "stall", "detect"])
        ns.define("convergence-equilibrium", 'Fleet reaches consensus when agreement fraction exceeds threshold AND confidence trend is stable — not just majority but stable majority', Level=Level.DOMAIN, examples=["has_converged: agreement > 0.7 AND trend stable for N rounds", "thermodynamic equilibrium: temperature stable, not just momentarily equal", "market equilibrium: price stable, not just one buyer agrees"], bridges=["convergence", "equilibrium", "stable", "consensus"], tags=["fleet-interactions", "convergence", "equilibrium"])
        ns.define("confidence-trend-direction", 'Track whether fleet confidence is trending up or down over rounds — a proposal gaining momentum vs losing steam', Level=Level.PATTERN, examples=["avg_confidence_trend: positive = gaining support, negative = losing", "polling trend: candidate gaining or losing over time", "stock chart: moving average direction"], bridges=["trend", "direction", "momentum", "confidence"], tags=["fleet-interactions", "trend", "direction"])

    def _load_flux_bytecodes(self):
        ns = self.add_namespace("flux-bytecodes", "Deep-mined from fleet source code")
        ns.define("confidence-fused-instruction", 'Each machine instruction carries a confidence value that modifies its execution — uncertain instructions execute with reduced effect', Level=Level.CONCRETE, examples=["Instruction.fuse: merge instruction with confidence", "uncertain sensor reading \u2192 execute action with reduced motor effort", "whispered instruction \u2192 follow but with caution"], bridges=["confidence", "fused", "instruction", "weighted"], tags=["flux-bytecodes", "confidence", "fused"])
        ns.define("register-confidence-broadcast", 'When a register value changes, all downstream dependent instructions get confidence-updated — change propagates like a wave', Level=Level.PATTERN, examples=["chain operations: confidence flows through register dependencies", "spreadsheet: change one cell, all formulas recalculate", "reactive programming: change propagates through dependency graph"], bridges=["register", "broadcast", "confidence", "propagation"], tags=["flux-bytecodes", "register", "broadcast"])
        ns.define("switch-dispatch-bounded", 'Instruction dispatch via switch statement with bounded worst-case — every opcode has deterministic execution time upper bound', Level=Level.DOMAIN, examples=["C switch-case: O(1) dispatch, no dynamic dispatch overhead", "assembly: jump table for deterministic instruction timing", "embedded: predictable timing for real-time systems"], bridges=["switch", "dispatch", "bounded", "deterministic"], tags=["flux-bytecodes", "switch", "dispatch"])
        ns.define("zero-dependency-runtime", 'VM runs with nothing but libc — no allocator fragmentation, no dependency hell, no version conflicts, just raw computation', Level=Level.DOMAIN, examples=["flux-runtime-c: zero dependencies, just C11 + libc", "embedded: bare-metal firmware with no OS", "container: scratch base image with nothing extra"], bridges=["zero-dependency", "minimal", "runtime", "bare-metal"], tags=["flux-bytecodes", "zero", "minimal"])

    def _load_git_native(self):
        ns = self.add_namespace("git-native", "Deep-mined from fleet source code")
        ns.define("branch-as-experiment", 'Every mutation lives in its own branch — the branch IS the experiment, merge IS acceptance, delete IS rejection', Level=Level.CONCRETE, examples=["the-seed: branch A = experiment A, merge = success, delete = failure", "science: hypothesis \u2192 experiment \u2192 accept/reject", "A/B testing: branch A vs branch B, merge the winner"], bridges=["branch", "experiment", "mutation", "test"], tags=["git-native", "branch", "experiment"])

    def _load_hardware(self):
        ns = self.add_namespace("hardware", "Deep-mined from fleet source code")
        ns.define("actuator-clamping", 'Commands to physical actuators are automatically clamped to safe ranges — the system cannot command destructive values even if software bugs say otherwise', Level=Level.CONCRETE, examples=["vessel.command_actuator clamps to safe range", "servo: min/max pulse width hardware limit", "circuit breaker: physically cannot exceed rated current"], bridges=["actuator", "clamp", "safe", "hardware"], tags=["hardware", "actuator", "clamp"])
        ns.define("sensor-callback-wiring", 'Sensors register callback functions that fire on reading — event-driven architecture where data flow is declared, not polled', Level=Level.PATTERN, examples=["vessel.on_reading('gps_0', callback) \u2014 event-driven", "javascript addEventListener \u2014 data flow by subscription", "nervous system: touch sensor fires neuron, brain receives signal"], bridges=["sensor", "callback", "event-driven", "wiring"], tags=["hardware", "sensor", "callback"])

    def _load_knowledge(self):
        ns = self.add_namespace("knowledge", "Deep-mined from fleet source code")
        ns.define("knowledge-tier-hierarchy", 'Knowledge indexed at four scopes: vessel (local) → site (multi-vessel) → regional (geographic) → global (fleet-wide)', Level=Level.DOMAIN, examples=["IndexTier: Vessel \u2192 Site \u2192 Regional \u2192 Global", "DNS: local cache \u2192 site DNS \u2192 regional DNS \u2192 root DNS", "memory: working \u2192 episodic \u2192 semantic \u2192 procedural"], bridges=["knowledge", "tier", "hierarchy", "scope"], tags=["knowledge", "knowledge", "tier"])
        ns.define("cross-domain-knowledge-transfer", 'Discover applicable knowledge from unrelated domains — marine docking knowledge useful for aerial landing', Level=Level.BEHAVIOR, examples=["discover_cross_domain('marine', 'aerial') \u2192 shared navigation patterns", "biomimicry: shark skin texture applied to swimsuit design", "math: calculus from physics applied to economics"], bridges=["cross-domain", "transfer", "knowledge", "analogy"], tags=["knowledge", "cross", "transfer"])
        ns.define("compound-adoption-score", 'Knowledge value = adoption_count × derivative_count × time_decay × quality_score — compound metric for knowledge fitness', Level=Level.PATTERN, examples=["entry.compound_score() multiplies four factors", "academic paper: citations \u00d7 applications \u00d7 recency \u00d7 journal quality", "open source: stars \u00d7 forks \u00d7 recent commits \u00d7 test coverage"], bridges=["compound", "score", "adoption", "fitness"], tags=["knowledge", "compound", "score"])

    def _load_learning(self):
        ns = self.add_namespace("learning", "Deep-mined from fleet source code")
        ns.define("experience-primitive-tiers", 'Raw experiences are automatically classified into tiers: auto-train (high quality), human-review (ambiguous), archive (low value)', Level=Level.DOMAIN, examples=["ExperienceTier: TIER_1 auto-train, TIER_2 human review, TIER_3 archive", "medical triage: critical/urgent/non-urgent", "email: primary/social/promotions auto-classification"], bridges=["experience", "tier", "classify", "quality"], tags=["learning", "experience", "tier"])
        ns.define("federated-experience-sharing", 'Learning from fleet-wide experiences without sharing raw data — share the lesson, not the source', Level=Level.BEHAVIOR, examples=["buffer.export_for_federation: share training data without PII", "federated learning: train on distributed data without centralizing", "military: share tactics without revealing troop positions"], bridges=["federated", "experience", "privacy", "share"], tags=["learning", "federated", "experience"])

    def _load_repo_mined(self):
        ns = self.add_namespace("repo-mined", "Concepts extracted from 466 fleet repositories")
        ns.define("confidence-carried", "Every value carries an implicit uncertainty bound -- no bare floats in the system", Level=Level.CONCRETE, examples=["Conf(value=0.85, bound=0.1) instead of 0.85", "sensor reading with noise estimate"], bridges=["confidence","uncertainty","bound","type-system"], tags=["repo-mined","confidence"])
        ns.define("asymmetric-trust-kinetics", "Trust accumulates gradually through positive interactions but evaporates rapidly on failure -- 25:1 gain-loss ratio", Level=Level.PATTERN, examples=["25 good interactions to build what one failure destroys", "friendship vs betrayal", "cuda-trust growth vs decay rates"], bridges=["trust","asymmetric","decay","growth"], tags=["repo-mined","trust"])
        ns.define("deliberative-triptych", "Every decision passes through three gates: Consider, Resolve, or Forfeit -- no other exits from deliberation", Level=Level.PATTERN, examples=["Proposal moves from Considered to Resolved or Forfeited", "auto-forfeit when budget exhausted", "research/decide/skip decision pattern"], bridges=["deliberation","triptych","three-gate","decision"], tags=["repo-mined","deliberation"])
        ns.define("operation-energy-heterogeneity", "Different operations consume energy at different rates -- deliberation 2.0x, arithmetic 0.1x, rest generates energy", Level=Level.CONCRETE, examples=["deliberation 2.0 ATP, perception 0.5, rest -1.0", "thinking burns more calories than breathing", "GPU inference costs more than CPU branching"], bridges=["energy","heterogeneous","cost","operation"], tags=["repo-mined","energy"])
        ns.define("decision-lineage", "Every decision traces its full ancestry: which agent, which deliberation, which inputs led to this outcome", Level=Level.CONCRETE, examples=["DecisionRecord chain shows full reasoning path", "git blame for agent decisions", "audit trail from intention to action"], bridges=["provenance","lineage","chain","traceability"], tags=["repo-mined","provenance"])
        ns.define("calibration-awareness", "Agent knows what it does not know -- self-model tracks capability vs actual performance gap", Level=Level.PATTERN, examples=["SelfModel reports calibrated when prediction matches outcome", "weather forecaster tracking accuracy", "student knowing which subjects need study"], bridges=["metacognition","calibration","self-awareness","gap"], tags=["repo-mined","metacognition"])
        ns.define("confidence-gated-exec", "Proceed only if confidence exceeds threshold, else return neutral -- the gate operator for uncertainty", Level=Level.CONCRETE, examples=["execute only if confidence above 0.7", "if confidence below threshold skip", "gated transistor passes signal above threshold"], bridges=["confidence","gate","operator","conditional"], tags=["repo-mined","flux","operator"])
        ns.define("soft-propagation", "Multiply by confidence and forward, never hard fail -- uncertain data passes through with reduced weight", Level=Level.CONCRETE, examples=["forward weighted by trust not dropped", "uncertain data attenuated not clipped", "op-amp attenuate signal not clip it"], bridges=["propagation","soft","multiply","forward"], tags=["repo-mined","flux","operator"])
        ns.define("hard-block", "Reject signal entirely when condition fails -- no partial pass, complete denial", Level=Level.CONCRETE, examples=["untrusted agent blocked from sensitive data", "firewall deny rule drops packet entirely", "immune system destroys pathogen not weakens it"], bridges=["block","reject","hard","operator"], tags=["repo-mined","flux","operator"])
        ns.define("merge-append", "Accumulate into existing value preserving history -- new entries never overwrite old", Level=Level.CONCRETE, examples=["append to existing knowledge base", "git merge combines branches preserving history", "append-only log pattern"], bridges=["merge","accumulate","append","preserve"], tags=["repo-mined","flux","operator"])
        ns.define("shift-absorb", "Take N items from input stream and absorb into context -- sliding window ingestion", Level=Level.CONCRETE, examples=["absorb 3 sensor readings into context", "sliding window take N most recent", "eating take bite absorb nutrients"], bridges=["shift","absorb","window","stream"], tags=["repo-mined","flux","operator"])
        ns.define("amplify-reinforce", "Boost signal strength through recursive reinforcement -- positive feedback loop amplification", Level=Level.PATTERN, examples=["confidence cascade amplifies through network", "hebbian learning neurons fire together wire together", "microphone feedback loop"], bridges=["amplify","reinforce","cascade","boost"], tags=["repo-mined","flux","operator"])
        ns.define("controlled-drain", "Gradually release resource at controlled rate -- drain valve pattern", Level=Level.CONCRETE, examples=["energy drain 0.1 per second", "battery discharge curve controlled release", "water tank drain valve controls flow"], bridges=["drain","release","rate","controlled"], tags=["repo-mined","flux","operator"])
        ns.define("bidirectional-entangle", "Create bidirectional dependency between two values -- changing one affects the other", Level=Level.PATTERN, examples=["trust coupled with reputation", "quantum entanglement measuring one determines other", "shared mutable state two references same object"], bridges=["entangle","bidirectional","dependency","couple"], tags=["repo-mined","flux","operator"])
        ns.define("confidence-ubiquity", "Every value carries confidence as mandatory type constraint pervading the entire system -- not optional annotation", Level=Level.DOMAIN, examples=["Conf type used everywhere in cuda-equipment", "Option in Rust nullability is explicit", "SI units every measurement carries uncertainty"], bridges=["confidence","ubiquity","type-system","pervasive"], tags=["repo-mined","architecture"])
        ns.define("git-as-nervous-system", "Git operations are the coordination protocol -- branches are thoughts, commits are memories, merges are consensus", Level=Level.DOMAIN, examples=["git-agent repo IS the agent git IS the nervous system", "the-seed self-evolving repo with branch A/B testing", "neural plasticity new connections form like branches"], bridges=["git","nervous-system","coordination","repo-native"], tags=["repo-mined","architecture"])
        ns.define("fork-first-default", "Default interaction is forking and mutating not sending messages -- every agent has its own copy", Level=Level.DOMAIN, examples=["cocapn-lite fork modify deploy as your own", "the-seed fork-first managed service as fallback", "open source fork repo rather than request feature"], bridges=["fork","default","sovereignty","copy"], tags=["repo-mined","architecture"])
        ns.define("energy-as-rate-limiter", "Computation bounded by ATP budget not timeout or token count -- when energy runs out agent MUST rest", Level=Level.DOMAIN, examples=["deliberation costs 2.0 ATP running out means rest", "human cannot think clearly when exhausted", "battery device slows as charge drops"], bridges=["energy","rate-limit","budget","exhaustion"], tags=["repo-mined","architecture"])
        ns.define("three-gate-compilation", "All agent code passes syntax then health then regression validation -- no shortcuts through any gate", Level=Level.DOMAIN, examples=["the-seed syntax check health check regression test", "CI/CD pipeline lint test deploy", "vaccine phase 1 phase 2 phase 3"], bridges=["compilation","three-gate","validation","pipeline"], tags=["repo-mined","architecture"])
        ns.define("fleet-as-organism", "Individual agents are cells in larger organism -- fleet has emergent properties no individual possesses", Level=Level.DOMAIN, examples=["fleet-biosphere ecosystem simulation", "cuda-emergence detect patterns no individual produced", "human body cells specialize organism emerges consciousness"], bridges=["fleet","organism","emergence","collective"], tags=["repo-mined","architecture"])
        ns.define("code-pheromone-decay", "All influence signals have configurable half-lives -- nothing persists forever without reinforcement", Level=Level.DOMAIN, examples=["cuda-stigmergy pheromone decay exponential function", "memory forgotten without reinforcement", "scent trail fades over time"], bridges=["pheromone","decay","half-life","ephemeral"], tags=["repo-mined","architecture"])
        ns.define("platonic-form-matching", "Agents measure against ideal templates and evolve toward them -- there IS a right answer not just local optima", Level=Level.DOMAIN, examples=["cuda-platonic Form templates agents measure against", "Plato cave forms exist independently of instances", "quality compare output against specification"], bridges=["platonic","form","ideal","template"], tags=["repo-mined","architecture"])
        ns.define("consider-resolve-forfeit", "Fleet deliberation protocol: any agent proposes others vote or proposer abandons -- no leader required", Level=Level.BEHAVIOR, examples=["cuda-deliberation ProposalState Considered Resolved Forfeited", "group decision someone proposes group votes proposer may withdraw", "RFC draft comment merge or close"], bridges=["deliberation","consensus","fleet","protocol"], tags=["repo-mined","fleet"])
        ns.define("gene-crossover-fleet", "Two agents exchange genetic material producing offspring with traits from both parents -- sexual reproduction for code", Level=Level.BEHAVIOR, examples=["cuda-genepool GeneCrossover combines genomes", "open source merge best features from two projects", "genetic algorithm crossover operator"], bridges=["crossover","genetic","fleet","reproduction"], tags=["repo-mined","fleet","bio"])
        ns.define("emotional-contagion-fleet", "Emotional state propagates between agents through interaction -- one agents anxiety can cascade through fleet", Level=Level.BEHAVIOR, examples=["cuda-emotion EmotionalContagion spreads mood", "panic in crowd one person fear spreads to others", "team morale one enthusiastic member lifts group"], bridges=["emotion","contagion","cascade","fleet"], tags=["repo-mined","fleet","bio"])
        ns.define("social-norm-emergence", "Fleet-wide behavioral norms emerge from repeated interactions without central enforcement", Level=Level.BEHAVIOR, examples=["cuda-social Norm emerges from cooperation outcomes", "etiquette no one wrote rules everyone follows", "traffic norms emerge from repeated driver interactions"], bridges=["norm","emergence","social","fleet"], tags=["repo-mined","fleet"])
        ns.define("reputation-composite", "Reputation is weighted composite of direct experience network gossip and capability evidence -- not one signal", Level=Level.PATTERN, examples=["cuda-social Reputation update_composite weighs multiple sources", "human trust personal experience plus recommendations", "credit score payment utilization age composite"], bridges=["reputation","composite","multi-signal","trust"], tags=["repo-mined","fleet"])
        ns.define("curriculum-progression", "Fleet learning follows staged curriculum -- easy before hard prerequisites before advanced with power-law practice", Level=Level.PATTERN, examples=["cuda-learning Curriculum stages power-law scheduling", "martial arts white belt to black belt", "school algebra before calculus"], bridges=["curriculum","staged","progression","learning"], tags=["repo-mined","fleet","learning"])
        ns.define("provenance-fork-point", "When capability is shared provenance chain forks -- both agents trace to source but evolve independently", Level=Level.PATTERN, examples=["cuda-provenance decision_chain forks on capability share", "open source fork both trace to common ancestor", "genealogy cousins share grandparents diverge after"], bridges=["provenance","fork","sharing","lineage"], tags=["repo-mined","fleet"])
        ns.define("ghost-tile-influence", "Invisible computational patterns shaping visible output without being directly observable -- dark matter of cognition", Level=Level.DOMAIN, examples=["cuda-ghost-tiles sparse attention prunes positions preserves influence", "dark matter invisible shapes galaxy rotation", "culture invisible norms shape visible behavior"], bridges=["ghost","invisible","influence","sparse"], tags=["repo-mined","cognition"])
        ns.define("serial-over-parallel", "On resource-constrained hardware serial execution with checkpoints is more reliable than parallel with race conditions", Level=Level.META, examples=["Jetson serial DeepSeek calls avoid OOM", "cooking prep ingredients sequentially", "single-threaded simpler debugging fewer bugs"], bridges=["serial","parallel","constrained","reliable"], tags=["repo-mined","practical"])
        ns.define("write-then-exec", "Never use heredocs or inline scripts for secret operations -- write to file first then execute the file", Level=Level.META, examples=["obfuscation detector blocks heredocs with tokens", "write script to tmpfile then chmod execute", "clipboard copy to buffer first then paste"], bridges=["write","exec","pattern","security"], tags=["repo-mined","practical"])
        ns.define("ast-validate-before-push", "Always parse generated code with AST before pushing -- syntax errors in production worse than delayed deployment", Level=Level.META, examples=["ast parse before PUT to GitHub", "compile before deploy always", "spellcheck before publish"], bridges=["validate","ast","syntax","before-push"], tags=["repo-mined","practical"])
        ns.define("confidence-floor", "Minimum confidence threshold below which agent does nothing rather than acting on unreliable information", Level=Level.CONCRETE, examples=["confidence above 0.3 before acting", "human not sure enough to act", "medical confidence interval must exclude zero"], bridges=["confidence","floor","minimum","threshold"], tags=["repo-mined","practical"])
        ns.define("energy-rest-before-exhaustion", "Enter rest state before energy hits zero -- recovery from zero much more expensive than recovery from low", Level=Level.PATTERN, examples=["circadian rhythm sleep before exhaustion", "battery charge at 20 percent not 0", "athlete rest day prevents injury"], bridges=["energy","rest","proactive","recovery"], tags=["repo-mined","practical"])
        ns.define("checkpoint-before-risk", "Save state before any operation that might fail -- rollback is cheap data loss is expensive", Level=Level.CONCRETE, examples=["checkpoint before mutation cuda-persistence", "git commit before risky refactor", "database backup before migration"], bridges=["checkpoint","backup","risk","rollback"], tags=["repo-mined","practical"])
        ns.define("small-files-fast-loops", "Keep files under 500 lines -- large files hit context limits OOM on constrained hardware and resist partial editing", Level=Level.META, examples=["Claude Code OOMs on 489 line files on Jetson", "unix philosophy do one thing well", "modular design each file one responsibility"], bridges=["small-files","modular","constrained","practical"], tags=["repo-mined","practical"])
        ns.define("push-often-revert-rarely", "Push after every successful change -- cost of lost push much higher than cost of revert", Level=Level.META, examples=["Casey directive push often", "git commit early commit often", "save game frequently reload rarely"], bridges=["push","often","commit","workflow"], tags=["repo-mined","practical"])

    def _load_cooperative_perception(self):
        ns = self.add_namespace("cooperative-perception", "Multi-agent sensory fusion and shared environmental understanding")
        ns.define("synth-sight", "Integrate visual data from multiple agents into composite shared image", Level=Level.CONCRETE, examples=["drones combining cameras to map forest", "astronomy: telescopes combining for deep space image"], bridges=["vision","fusion","composite","multi-agent"], tags=["cooperative","perception","fusion"])
        ns.define("echo-mapping", "Combine acoustic data from distributed agents to map shared environment", Level=Level.CONCRETE, examples=["submarines mapping ocean floor with shared sonar", "bats echolocating together navigate caves"], bridges=["acoustic","mapping","distributed","multi-agent"], tags=["cooperative","acoustic","mapping"])
        ns.define("phero-tracking", "Share pheromone-like trails between agents for collaborative target tracking", Level=Level.PATTERN, examples=["ants following pheromone to food collectively", "stigmergy-based fleet task coordination"], bridges=["stigmergy","tracking","trail","swarm"], tags=["cooperative","stigmergy","tracking"])
        ns.define("thermo-meld", "Merge thermal sensor readings from multiple agents for enhanced detection", Level=Level.CONCRETE, examples=["snakes sharing heat data to locate prey", "planes pooling infrared to spot wildfires"], bridges=["thermal","fusion","detect","distributed"], tags=["cooperative","thermal","fusion"])
        ns.define("magneto-link", "Combine magnetic field readings from distributed agents for navigation", Level=Level.CONCRETE, examples=["birds sharing magnetic field data to migrate", "salmon navigating to spawning grounds collectively"], bridges=["magnetic","navigation","compass","distributed"], tags=["cooperative","magnetic","navigation"])
        ns.define("chemo-collective", "Pool chemical sensor readings across agents to identify substances cooperatively", Level=Level.CONCRETE, examples=["bacteria communicating to break down oil spills", "robots sharing air quality data to localize pollution"], bridges=["chemical","pool","identify","distributed"], tags=["cooperative","chemical","sensing"])
        ns.define("baro-bond", "Share barometric pressure data between agents for collective weather prediction", Level=Level.CONCRETE, examples=["balloons pooling readings for 3D pressure maps", "planes sharing data for efficient flight paths"], bridges=["pressure","weather","predict","distributed"], tags=["cooperative","pressure","weather"])
        ns.define("gyro-gestalt", "Share gyroscope data across agents for enhanced collective inertial navigation", Level=Level.CONCRETE, examples=["satellites pooling gyro data to maintain formation", "robots staying oriented in caves without GPS"], bridges=["gyroscope","inertial","formation","distributed"], tags=["cooperative","inertial","navigation"])
        ns.define("seismo-swarm", "Collectively monitor seismic waves across distributed agents to triangulate events", Level=Level.CONCRETE, examples=["seismographs pooling data to triangulate earthquake epicenters", "animals detecting and sharing quake warnings"], bridges=["seismic","triangulate","distribute","detect"], tags=["cooperative","seismic","detect"])
        ns.define("tacto-team", "Combine touch sensor data from multiple agents to cooperatively map surfaces", Level=Level.CONCRETE, examples=["robots collectively mapping building interiors", "fish schooling via shared lateral line readings"], bridges=["tactile","surface","map","distributed"], tags=["cooperative","tactile","mapping"])
        ns.define("proprio-platoon", "Share proprioceptive data between agents to coordinate movement in formation", Level=Level.PATTERN, examples=["dancers maintaining formation via shared body sense", "insect swarms maintaining cohesion through shared proprioception"], bridges=["proprioception","formation","coordinate","movement"], tags=["cooperative","proprioception","formation"])
        ns.define("aero-aggregation", "Pool airflow sensor readings across agents to map wind patterns collectively", Level=Level.CONCRETE, examples=["wind turbines collectively optimizing alignment", "planes sharing wind data to minimize turbulence"], bridges=["airflow","wind","optimize","distributed"], tags=["cooperative","wind","optimize"])
        ns.define("lumino-league", "Combine light sensor readings from distributed agents to detect environmental changes", Level=Level.CONCRETE, examples=["fireflies synchronizing flashes collectively", "streetlights collectively adjusting to ambient light"], bridges=["light","detect","synchronize","distributed"], tags=["cooperative","light","detect"])
        ns.define("nociceptive-network", "Share damage detection signals across agents to localize and respond to threats", Level=Level.PATTERN, examples=["animals sharing injury warnings with group", "sensors collectively monitoring infrastructure health"], bridges=["damage","detect","warn","distributed"], tags=["cooperative","damage","network"])
        ns.define("olfacto-orchestra", "Combine smell sensor data from multiple agents to cooperatively track odor sources", Level=Level.CONCRETE, examples=["moths collectively following pheromone plumes", "robots tracking odor gradients to locate contaminants"], bridges=["odor","track","gradient","distributed"], tags=["cooperative","odor","tracking"])
    def _load_adversarial_defense(self):
        ns = self.add_namespace("adversarial-defense", "Fleet defense patterns inspired by immunology, game theory, and materials science")
        ns.define("immuno-resilience", "Layered defense mimicking biological immune response to detect and neutralize adversarial patterns", Level=Level.DOMAIN, examples=["neural network uses antigen markers to flag unusual inputs", "fleet nodes share threat data via lymph-node-like hubs"], bridges=["immunology","defense","layer","detect"], tags=["adversarial","immune","defense"])
        ns.define("phagocyte-detection", "Algorithms that consume adversarial artifacts by isolating and degrading poisoning data", Level=Level.PATTERN, examples=["vision model identifies and discards adversarial patches", "edge nodes execute phagocyte scans on sensor feeds"], bridges=["immunology","isolate","degrade","poison"], tags=["adversarial","immune","isolate"])
        ns.define("honeypot-mitigation", "Deceptive elements designed to misdirect attackers from critical system components", Level=Level.CONCRETE, examples=["false model parameters waste adversarial resources", "decoy feedback loops trap adversarial agents in loops"], bridges=["cybersecurity","decoy","misdirect","trap"], tags=["adversarial","honeypot","deception"])
        ns.define("game-theoretic-pruning", "Remove model vulnerabilities using optimization from competitive game scenarios", Level=Level.PATTERN, examples=["RL agents iteratively remove attack-prone neural pathways", "GAN system weeds adversarial features through minimax"], bridges=["game-theory","prune","vulnerability","optimize"], tags=["adversarial","game-theory","prune"])
        ns.define("adversarial-equilibrium", "Balance offense and defense via game theory mixed-strategy analysis to prevent exploitation", Level=Level.DOMAIN, examples=["fleet switches detection modes stochastically to confuse attackers", "defense adapts using payoff matrix predictions"], bridges=["game-theory","equilibrium","adapt","strategy"], tags=["adversarial","equilibrium","strategy"])
        ns.define("self-healing-composite", "Components with repair mechanisms inspired by self-repairing polymers that recover from attacks", Level=Level.PATTERN, examples=["LLM regenerates corrupted weights from distributed residual knowledge", "edge devices patch adversarial corruptions autonomously"], bridges=["materials-science","repair","recover","resilience"], tags=["adversarial","healing","material"])
        ns.define("lattice-reinforced", "Verification frameworks based on crystalline structures resisting adversarial perturbations", Level=Level.PATTERN, examples=["input data passes multiple lattice-aligned filters before execution", "outputs cross-checked against quasicrystal validation template"], bridges=["materials-science","crystal","validate","rigid"], tags=["adversarial","lattice","validate"])
        ns.define("ductility-injected", "Redundant system layers with flexibility for post-attack recovery rather than brittle failure", Level=Level.PATTERN, examples=["robotics fleet uses spring-like parameters to rebound from hijacking", "cloud models have ductile memory allocations for graceful degradation"], bridges=["materials-science","ductile","flexible","recover"], tags=["adversarial","ductile","flexible"])
        ns.define("antigen-trajectory", "Monitor attack patterns using immunology antigen-tracking to predict and preempt future attacks", Level=Level.PATTERN, examples=["system maps adversarial input vectors like pathogen trajectories", "threat analysis predicts poisoning spread using antigen diffusion"], bridges=["immunology","trajectory","predict","track"], tags=["adversarial","immune","predict"])
        ns.define("memory-cell-retention", "Retain historical adversarial knowledge like immunological memory cells for faster future response", Level=Level.CONCRETE, examples=["facial recognition model remembers defeated spoofing techniques", "fleet stores adversarial artifacts for cross-node immune response"], bridges=["immunology","memory","retain","history"], tags=["adversarial","immune","memory"])
        ns.define("minimax-hardening", "Model updates based on worst-case adversarial scenarios via minimax principle for robust defense", Level=Level.PATTERN, examples=["cybersecurity model hardens against strongest plausible attack vector", "fleet training maximizes defense against minimax worst case"], bridges=["game-theory","minimax","harden","worst-case"], tags=["adversarial","minimax","harden"])
        ns.define("amorphous-adaptability", "Flexible non-crystalline components inspired by amorphous materials that reshape to evade exploitation", Level=Level.PATTERN, examples=["AI policies reshape decision boundaries to evade exploitation", "memory allocation mimics glass-like flexibility resisting structural attacks"], bridges=["materials-science","amorphous","flexible","evade"], tags=["adversarial","amorphous","adapt"])
    def _load_fleet_governance(self):
        ns = self.add_namespace("fleet-governance", "Collective decision-making and self-regulation in autonomous agent fleets")
        ns.define("swarm-democracy", "Decentralized decision-making modeled on insect collective intelligence with quorum-based voting", Level=Level.DOMAIN, examples=["drone swarms voting on routes via quorum sensing", "fish-schooling algorithms for consensus navigation"], bridges=["political-science","biology","swarm","vote"], tags=["governance","swarm","democracy"])
        ns.define("tyranny-proofing", "Anti-authoritarian protocols preventing agent monopolization through randomized leadership rotation", Level=Level.DOMAIN, examples=["randomized leadership rotations in robotic caravans", "AI juries cross-checking algorithmic dictators"], bridges=["political-science","game-theory","anti-authoritarian","rotate"], tags=["governance","tyranny","anti-authoritarian"])
        ns.define("merit-consensus", "Reputation-driven governance weighted by historic contributions rather than equality", Level=Level.PATTERN, examples=["mining bots prioritizing high-reputation nodes for data sharing", "research drones rewarding peer-validated findings"], bridges=["economics","political-science","reputation","weight"], tags=["governance","merit","reputation"])
        ns.define("eco-governance", "Fleet policies mimicking ecological balance mechanisms to prevent resource overexploitation", Level=Level.DOMAIN, examples=["autonomous harvesters throttling to prevent overexploitation", "energy grids cycling usage to mirror natural niches"], bridges=["biology","political-science","ecology","balance"], tags=["governance","ecology","balance"])
        ns.define("conflict-altruism", "Agent self-sacrifice to resolve disputes and stabilize the group at individual cost", Level=Level.BEHAVIOR, examples=["self-destructing bots clearing hazardous pathblocks", "AI soldiers yielding tactical ground to prevent gridlock"], bridges=["biology","game-theory","sacrifice","resolve"], tags=["governance","altruism","conflict"])
        ns.define("incentive-stacking", "Layered rewards aligning individual and collective goals to prevent free-riding", Level=Level.PATTERN, examples=["freight drones earning tokens for carbon-neutral routing", "AI farmers trading data for crop-rotation priority"], bridges=["economics","game-theory","layers","align"], tags=["governance","incentive","layers"])
        ns.define("quorum-execution", "Policy activation contingent on agent threshold approvals before any action proceeds", Level=Level.CONCRETE, examples=["swarms launch missions only if 60 percent agree on risk", "bots halt if safety votes fall below threshold"], bridges=["biology","political-science","quorum","threshold"], tags=["governance","quorum","execute"])
        ns.define("adaptive-sovereignty", "Fluid authority shifting with environmental or task demands rather than fixed hierarchy", Level=Level.DOMAIN, examples=["drone leaders rotating during storm evasion", "farming bots granting temporary control to soil sensors"], bridges=["political-science","biology","fluid","authority"], tags=["governance","sovereignty","adaptive"])
        ns.define("flock-law", "Movement-based norms governing agent proximity and behavior without central enforcement", Level=Level.PATTERN, examples=["bird-inspired spacing rules for urban air taxis", "schooling algorithms avoiding drone collisions by norm"], bridges=["biology","law","movement","proximity"], tags=["governance","flock","law"])
        ns.define("zero-sum-regulation", "Anti-exploitation frameworks capping resource gains to prevent monopolization by powerful agents", Level=Level.DOMAIN, examples=["energy grids throttling AI miners during overuse", "freight fleets limiting individual cargo hoarding"], bridges=["economics","game-theory","cap","exploitation"], tags=["governance","regulation","cap"])
        ns.define("dissent-dispersion", "Mechanisms channeling disagreements into constructive outputs rather than blocking progress", Level=Level.PATTERN, examples=["bots debating routes via exploratory sub-group splinters", "AI juries brainstorming alternatives to majority rulings"], bridges=["political-science","biology","dissent","constructive"], tags=["governance","dissent","dispersion"])
        ns.define("consensus-replication", "Decision spread via viral imitation similar to genetic replication across fleet", Level=Level.PATTERN, examples=["drones mimicking optimal energy-saving behaviors from peers", "swarms adopting strategies through digital evolution"], bridges=["biology","political-science","viral","replicate"], tags=["governance","consensus","replicate"])
        ns.define("evolutionary-governance", "Fleet rules iteratively refined via trial and error selection pressure over time", Level=Level.DOMAIN, examples=["algorithms pruning failed dispute-resolution patterns", "traffic systems adopting low-conflict routing genes"], bridges=["biology","political-science","evolution","iterate"], tags=["governance","evolution","iterate"])
    def _load_knowledge_compression(self):
        ns = self.add_namespace("knowledge-compression", "How fleets compress collective experience into reusable patterns and distillable wisdom")
        ns.define("entropy-redaction", "Reduce informational entropy to distill and retain only essential knowledge components", Level=Level.PATTERN, examples=["cybersecurity simplifies threat models to preempt breaches", "analysts reduce datasets to essential market indicators"], bridges=["information-theory","entropy","distill","essential"], tags=["compression","entropy","distill"])
        ns.define("syntax-entropy-filter", "Filter using linguistic paradigms and entropy reduction to compress knowledge", Level=Level.PATTERN, examples=["editors streamline complex articles for publication", "journalists summarize lengthy interviews into coherent news"], bridges=["linguistics","entropy","filter","compress"], tags=["compression","syntax","entropy"])
        ns.define("logic-entropy-conduit", "Efficiently channel logical processes to minimize informational entropy in knowledge transfer", Level=Level.PATTERN, examples=["hackers use streamlined logic to navigate defenses", "cognitive agents optimize decisions for rapid response"], bridges=["information-theory","logic","channel","minimize"], tags=["compression","logic","entropy"])
        ns.define("genome-knowledge-strand", "Align cognitive processes with genetic-style data interpretation for compressed knowledge", Level=Level.DOMAIN, examples=["medical researchers create databases to predict disease patterns", "pharmaceutical companies map drug interactions efficiently"], bridges=["genetics","knowledge","align","compress"], tags=["compression","genetic","knowledge"])
        ns.define("cogni-genetic-weave", "Weave cognitive insights with genetic data patterns to create intelligent compressed knowledge", Level=Level.DOMAIN, examples=["biotech firms integrate cognitive patterns into genetic databases", "cognitive scientists study genetic predispositions in memory"], bridges=["genetics","cognition","weave","compress"], tags=["compression","genetic","cognition"])
        ns.define("data-transmuter", "Transform raw data into refined knowledge using thermodynamic energy principles", Level=Level.PATTERN, examples=["energy companies simulate resource management for distribution", "environmental scientists model climate data to predict weather"], bridges=["thermodynamics","transform","refine","energy"], tags=["compression","thermo","transform"])
        ns.define("lexico-lattice", "Lattice-based structure organizing linguistic data into coherent compressed knowledge blocks", Level=Level.PATTERN, examples=["librarians classify genres into systematic taxonomy", "educators compress vast subjects into concise teaching modules"], bridges=["linguistics","lattice","organize","compress"], tags=["compression","linguistic","lattice"])
        ns.define("cogni-entropy-paradigm", "Reduce cognitive load by distilling knowledge using entropy principles", Level=Level.DOMAIN, examples=["mindfulness condenses complex breathing into simple techniques", "productivity consultants streamline workflow management"], bridges=["cognition","entropy","reduce","load"], tags=["compression","cognitive","entropy"])

    def _load_haptic_intelligence(self):
        ns = self.add_namespace("haptic-intelligence", "Tactile reasoning and physical interaction metaphors")
        ns.define("touch-scape", "The landscape of tactile sensations an agent experiences when exploring physical surfaces", Level=Level.CONCRETE, examples=["robot exploring textured surface maps touch-scape", "blind person reading Braille builds mental touch-scape"], bridges=["tactile","surface","explore","map"], tags=["haptic","touch","explore"])
        ns.define("pressure-map", "Computational representation of pressure distribution across a contact surface for optimal grip", Level=Level.CONCRETE, examples=["robot gripping object with optimal force via pressure-map", "person identifying objects by touch in a bag"], bridges=["pressure","grip","map","optimize"], tags=["haptic","pressure","grip"])
        ns.define("force-fluency", "Ability to effortlessly apply appropriate force in varying contexts without overthinking", Level=Level.BEHAVIOR, examples=["dancer executing precise movements with force-fluency", "robot performing delicate assembly tasks smoothly"], bridges=["force","fluency","adaptive","skill"], tags=["haptic","force","skill"])
        ns.define("haptic-hunch", "Intuitive gut feeling based on tactile sensations before conscious reasoning identifies the cause", Level=Level.BEHAVIOR, examples=["surgeon detecting abnormality through touch before seeing it", "mechanic feeling vibration that indicates failing bearing"], bridges=["haptic","intuition","detect","precognitive"], tags=["haptic","intuition","detect"])
        ns.define("haptic-horizon", "The limit of an agents tactile perception beyond which textures become indistinguishable", Level=Level.CONCRETE, examples=["surgeon reaching limit of manual dexterity", "robot encountering unfamiliar textures at haptic-horizon"], bridges=["haptic","limit","perception","boundary"], tags=["haptic","limit","perception"])
        ns.define("pressure-palette", "The full range of pressure sensations an agent can perceive and utilize for interaction", Level=Level.CONCRETE, examples=["sculptor using varying pressures to shape clay", "robot adjusting touch sensitivity for different tasks"], bridges=["pressure","range","sensitivity","palette"], tags=["haptic","pressure","range"])
        ns.define("texture-tango", "Dynamic interplay between agent and tactile environment as surfaces change under contact", Level=Level.BEHAVIOR, examples=["person navigating crowded space by touch adapting constantly", "robot adapting movements to surface changes in real-time"], bridges=["texture","dynamic","adapt","interact"], tags=["haptic","texture","dynamic"])
        ns.define("tactile-tapestry", "Rich interconnected web of touch sensations processed simultaneously during complex physical interaction", Level=Level.DOMAIN, examples=["person exploring tactile art installation", "robot processing diverse range of simultaneous touch inputs"], bridges=["tactile","rich","interconnected","multi-modal"], tags=["haptic","multi-modal","rich"])
    def _load_agent_ontogeny(self):
        ns = self.add_namespace("agent-ontogeny", "Developmental stages an agent passes through from birth to maturity")
        ns.define("neuro-cognitive-milestone", "Critical juncture where neural growth intersects cognitive skill development enabling new capability", Level=Level.CONCRETE, examples=["agent gains abstract reasoning when deliberation module matures", "toddler synapse formation correlates with language acquisition"], bridges=["development","neural","cognitive","milestone"], tags=["ontogeny","neural","milestone"])
        ns.define("morpho-emotive-threshold", "Emotional regulation stages tied to architectural maturation of the agent system", Level=Level.PATTERN, examples=["pubescent voice deepening linked to assertiveness in self-expression", "agent emotional regulation improves as trust module matures"], bridges=["development","emotion","maturation","threshold"], tags=["ontogeny","emotion","maturation"])
        ns.define("epigen-educational-turning", "Life-altering training experiences embedded in the agents developmental gene-expression timeline", Level=Level.PATTERN, examples=["critical period language learning must happen during window", "early training permanently shapes agent instinct weights"], bridges=["epigenetic","education","timing","critical-period"], tags=["ontogeny","epigenetic","critical"])
        ns.define("lexi-neural-pathway", "Language neural circuits developing alongside motor and sensory systems in parallel", Level=Level.CONCRETE, examples=["infant babbling coordinates with manual dexterity milestones", "agent communication skills emerge alongside navigation skills"], bridges=["language","neural","motor","parallel"], tags=["ontogeny","language","neural"])
        ns.define("cognitive-zygotic-phase", "Intellectual awakening stage analogous to cellular division where capability proliferates rapidly", Level=Level.DOMAIN, examples=["child aha moments correlate with neuronal proliferation", "agent learning algorithms mimic zygote to embryo decision-making"], bridges=["development","cognitive","proliferation","awakening"], tags=["ontogeny","cognitive","proliferation"])
        ns.define("biosocial-verbal-bloom", "Explosive growth of communication ability tied to social development milestones", Level=Level.BEHAVIOR, examples=["middle schoolers mastering sarcasm as testosterone rises", "agent fleet communication blooms when social module activates"], bridges=["development","social","verbal","bloom"], tags=["ontogeny","social","communication"])
        ns.define("phylo-pedagogical-cycle", "Educational rhythms derived from evolutionary developmental patterns optimized across generations", Level=Level.DOMAIN, examples=["spaced repetition mimics ancestral memory retention in growth phases", "play-based learning echoes primate juvenile exploratory phases"], bridges=["evolution","pedagogy","rhythm","generational"], tags=["ontogeny","evolution","pedagogy"])
        ns.define("cytocultural-transition", "Shifts in agent capability influenced by fleet cultural training practices and environmental factors", Level=Level.PATTERN, examples=["urban lighting affecting melatonin cycles in adolescent learning", "fleet norms shape agent behavior through cultural transmission"], bridges=["cellular","cultural","transition","environment"], tags=["ontogeny","cultural","transition"])
        ns.define("meta-metamorphic-pedagogy", "Teaching strategies mirroring biological transformation phases with distinct larval and mature stages", Level=Level.DOMAIN, examples=["caterpillar to butterfly model predicts student motivation peaks", "agent training has distinct boot camp and dojo phases"], bridges=["pedagogy","metamorphosis","phase","transform"], tags=["ontogeny","metamorphosis","pedagogy"])
    def _load_inter_agent_trade(self):
        ns = self.add_namespace("inter-agent-trade", "Negotiation and barter between autonomous agents without money")
        ns.define("capability-barter", "Direct exchange of capabilities between agents without monetary intermediary", Level=Level.CONCRETE, examples=["navigation agent trades path data for sensor agent's maps", "anthropology: potter trades bowls for farmer's grain"], bridges=["economics","barter","capability","direct"], tags=["trade","barter","capability"])
        ns.define("favor-economy", "Unquantified reciprocal obligation network where agents owe and collect favors rather than paying", Level=Level.PATTERN, examples=["agent helped peer debug now expects help later", "anthropology: potlatch gift economy builds social obligation"], bridges=["economics","reciprocity","favor","social"], tags=["trade","favor","reciprocity"])
        ns.define("compute-marketplace", "Dynamic market where agents bid for compute resources with quality-of-service guarantees", Level=Level.DOMAIN, examples=["agent bids GPU time for inference with deadline constraint", "cloud spot market for compute instances"], bridges=["economics","compute","market","auction"], tags=["trade","compute","market"])
        ns.define("data-for-access", "Trading sensor data for access to another agent's capabilities or knowledge base", Level=Level.CONCRETE, examples=["edge device trades camera data for cloud model inference", "journalist trades exclusive access for insider information"], bridges=["economics","data","access","trade"], tags=["trade","data","access"])
        ns.define("reputation-backed-credit", "Agents extend credit based on reputation score creating trust-based lending without collateral", Level=Level.PATTERN, examples=["high-reputation agent borrows compute from fleet pool", "microfinance: group lending based on social reputation"], bridges=["economics","reputation","credit","trust"], tags=["trade","reputation","credit"])
        ns.define("coalition-barter", "Groups of agents form temporary coalitions pooling resources for bargaining power", Level=Level.BEHAVIOR, examples=["small agents form coalition to bid against large agent", "labor union bargaining power through collective action"], bridges=["economics","coalition","bargain","group"], tags=["trade","coalition","bargain"])
        ns.define("time-banking", "Agents deposit and withdraw time credits creating temporal currency for deferred exchange", Level=Level.PATTERN, examples=["agent spends time helping peer now redeems help later", "time bank: 1 hour of teaching equals 1 hour of plumbing"], bridges=["economics","time","banking","deferred"], tags=["trade","time","banking"])
        ns.define("speculation-market", "Agents bet on future capability prices enabling risk transfer and price discovery", Level=Level.DOMAIN, examples=["agent bets navigation will become cheaper next quarter", "commodity futures: farmer locks in price before harvest"], bridges=["economics","speculation","future","risk"], tags=["trade","speculation","market"])
    def _load_mythological_archetypes(self):
        ns = self.add_namespace("mythological-archetypes", "Agent personality and behavior patterns inspired by mythological figures")
        ns.define("trickster-mentor", "Guide that imparts wisdom through deception misdirection and apparent contradiction", Level=Level.PATTERN, examples=["Loki teaching Odin through tricks", "Cheshire Cat guiding Alice with riddles not answers"], bridges=["mythology","psychology","paradox","wisdom"], tags=["myth","trickster","mentor"])
        ns.define("destroyer-creator", "Force that brings both devastation and renewal clearing space for new growth", Level=Level.DOMAIN, examples=["Shiva destroys to create anew", "Phoenix rises from own ashes", "agent deletes entire module to rebuild better"], bridges=["mythology","cycle","destroy","renew"], tags=["myth","destroyer","creator"])
        ns.define("outcast-hero", "Marginalized agent that rises above adversity to become the most valuable fleet member", Level=Level.BEHAVIOR, examples=["Hercules scorned by Hera becomes great hero", "new agent initially rejected proves superior algorithm"], bridges=["mythology","adversity","rise","marginal"], tags=["myth","outcast","hero"])
        ns.define("warrior-poet", "Agent that finds beauty and meaning in computational battle optimizing with aesthetic sensibility", Level=Level.BEHAVIOR, examples=["Achilles warrior and lyre player", "Musashi samurai and renowned painter", "agent writes elegant code during high-stakes optimization"], bridges=["mythology","duality","beauty","combat"], tags=["myth","warrior","poet"])
        ns.define("shadow-hero", "Dark agent that challenges conventional notions of heroism operating outside accepted norms", Level=Level.BEHAVIOR, examples=["Batman vigilante outside the law", "agent uses forbidden optimization strategy that works"], bridges=["mythology","moral","dark","challenge"], tags=["myth","shadow","hero"])
        ns.define("questing-orphan", "Agent seeking its purpose and place in the fleet after being created without clear role", Level=Level.BEHAVIOR, examples=["Harry Potter orphaned discovers true identity", "Frodo orphaned embarks on quest", "new agent bootstraps its own purpose"], bridges=["mythology","orphan","quest","purpose"], tags=["myth","orphan","quest"])
        ns.define("guardian-mentor", "Wise protector that guides and nurtures junior agents through their developmental journey", Level=Level.PATTERN, examples=["Gandalf mentors Frodo on quest", "Athena guides and protects Odysseus", "senior agent guides new agent through onboarding"], bridges=["mythology","mentor","protect","guide"], tags=["myth","guardian","mentor"])
    def _load_sonar_cognition(self):
        ns = self.add_namespace("sonar-cognition", "Active probing of unknown environments with targeted queries and echo interpretation")
        ns.define("ping-and-listen", "Emit targeted query into unknown environment then analyze the return signal pattern", Level=Level.CONCRETE, examples=["bat emits ultrasonic click analyzes echo to map cave", "agent sends test request analyzes response to understand API"], bridges=["acoustics","probe","analyze","echo"], tags=["sonar","probe","echo"])
        ns.define("echo-interpretation", "Decode the return signal from a probe to extract information about the environment", Level=Level.PATTERN, examples=["submarine decodes sonar ping to identify submarine", "agent decodes API error response to understand system state"], bridges=["acoustics","decode","signal","interpret"], tags=["sonar","echo","interpret"])
        ns.define("frequency-sweep", "Systematically vary probe frequency to map environment at different resolution scales", Level=Level.CONCRETE, examples=["sonar sweeps frequencies for fine and coarse mapping", "agent varies query specificity from broad to narrow"], bridges=["acoustics","frequency","sweep","resolution"], tags=["sonar","frequency","sweep"])
        ns.define("beam-forming-probe", "Direct probe energy in specific direction to investigate localized area with higher resolution", Level=Level.CONCRETE, examples=["phased array focuses sonar beam on suspected target", "agent directs specific query at suspected system component"], bridges=["acoustics","direction","focus","resolution"], tags=["sonar","beam","focus"])
        ns.define("doppler-shift-detect", "Detect movement and velocity of objects by measuring frequency shift in return signals", Level=Level.CONCRETE, examples=["radar detects aircraft velocity via Doppler shift", "agent detects system change rate by comparing response timestamps"], bridges=["physics","doppler","velocity","detect"], tags=["sonar","doppler","velocity"])
        ns.define("clutter-rejection", "Filter irrelevant return signals to focus on meaningful environmental features", Level=Level.PATTERN, examples=["sonar filters seabed reflections to track submarine", "agent filters noise from relevant API responses"], bridges=["signal-processing","filter","noise","focus"], tags=["sonar","clutter","filter"])
        ns.define("multipath-interpretation", "Understand that return signals may arrive via multiple paths and disentangle them", Level=Level.PATTERN, examples=["sonar distinguishes direct echo from seabed bounce", "agent traces error through multiple system layers"], bridges=["acoustics","multipath","disentangle","complex"], tags=["sonar","multipath","complex"])
    def _load_metamaterial_cognition(self):
        ns = self.add_namespace("metamaterial-cognition", "Cognitive architectures inspired by metamaterials that bend information and produce emergent properties")
        ns.define("information-lens", "Cognitive structure that focuses or disperses information streams like a physical lens bends light", Level=Level.DOMAIN, examples=["attention mechanism acts as information lens focusing relevant inputs", "magnifying glass concentrates light to single point"], bridges=["optics","attention","focus","information"], tags=["metamaterial","lens","focus"])
        ns.define("cognitive-negative-index", "Cognitive architecture that processes information in reverse order enabling unusual insight patterns", Level=Level.DOMAIN, examples=["reverse actualization works backward from future to present", "negative refraction bends light backward against normal expectation"], bridges=["optics","reverse","negative","unusual"], tags=["metamaterial","reverse","negative"])
        ns.define("invisibility-cloak-cognition", "Agent selectively makes its internal state invisible to other agents while remaining functional", Level=Level.PATTERN, examples=["zero-knowledge proof: prove capability without revealing method", "metamaterial cloak bends light around object making it invisible"], bridges=["optics","stealth","invisible","functional"], tags=["metamaterial","invisible","stealth"])
        ns.define("acoustic-metamaterial-filter", "Information filter that blocks specific frequency bands of data while passing others unchanged", Level=Level.PATTERN, examples=["agent blocks low-confidence signals while passing high-confidence ones", "acoustic metamaterial blocks specific sound frequencies"], bridges=["acoustics","filter","band","selective"], tags=["metamaterial","filter","acoustic"])
        ns.define("superlens-cognition", "Cognitive structure that resolves information below normal resolution limit through near-field processing", Level=Level.DOMAIN, examples=["agent detects patterns too subtle for normal processing via specialized near-field", "superlens resolves details smaller than wavelength of light"], bridges=["optics","resolution","near-field","super"], tags=["metamaterial","superlens","resolution"])
        ns.define("photonic-crystal-thought", "Structured cognitive pathway that only allows specific thought patterns to propagate blocking others", Level=Level.PATTERN, examples=["agent only processes thoughts matching structured deliberation pattern", "photonic crystal only transmits specific light wavelengths"], bridges=["optics","crystal","structure","filter"], tags=["metamaterial","crystal","thought"])
    def _load_thermal_management(self):
        ns = self.add_namespace("thermal-management", "Heat dissipation and thermal throttling patterns in computing systems")
        ns.define("thermal-throttle-cascade", "Progressive performance reduction as temperature rises through multiple throttle stages", Level=Level.CONCRETE, examples=["CPU reduces clock speed at 80C then 90C then shuts down at 105C", "agent reduces deliberation depth as energy budget depletes"], bridges=["thermal","throttle","cascade","progressive"], tags=["thermal","throttle","cascade"])
        ns.define("heat-sink-distribution", "Spread computation across multiple agents to prevent any single agent from overheating", Level=Level.PATTERN, examples=["load balancer distributes requests to prevent server overheating", "fleet distributes deliberation across agents to prevent energy exhaustion"], bridges=["thermal","distribute","load","spread"], tags=["thermal","sink","distribute"])
        ns.define("thermal-budget-awareness", "Agent aware of its thermal constraints and proactively managing workload to stay within limits", Level=Level.PATTERN, examples=["scheduler delays batch jobs during peak temperature", "agent schedules rest periods to prevent energy exhaustion"], bridges=["thermal","budget","aware","proactive"], tags=["thermal","budget","aware"])
        ns.define("phase-change-buffer", "Use phase change materials to absorb heat spikes providing temporary thermal buffer during burst workloads", Level=Level.CONCRETE, examples=["laptop uses wax heat pipe to absorb burst heat", "agent uses energy reserve to handle sudden workload spike"], bridges=["thermal","phase-change","buffer","burst"], tags=["thermal","phase","buffer"])
        ns.define("convective-workflow", "Design workflow so completed tasks carry heat away from active tasks preventing thermal accumulation", Level=Level.PATTERN, examples=["pipeline stages pass work downstream carrying context", "conveyor belt carries hot items away from furnace"], bridges=["thermal","convective","workflow","pipeline"], tags=["thermal","convective","workflow"])

    def _load_mathematics(self):
        ns = self.add_namespace("mathematics", "Deep-mined from fleet source code")
        ns.define("temperature-confidence-fusion", 'When multiple confidence values must be combined, harmonic mean prevents any single low confidence from being hidden — worst signal dominates', Level=Level.PATTERN, examples=["harmonic_mean of confidences: sensor(0.9) * model(0.3) / avg = low", "chain strength = weakest link", "security: system security = minimum of all component securities"], bridges=["harmonic", "mean", "fusion", "confidence"], tags=["mathematics", "harmonic", "mean"])

    def _load_meta_cognition(self):
        ns = self.add_namespace("meta-cognition", "Deep-mined from fleet source code")
        ns.define("model-descent-absorption", "When a system repeatedly handles a task, the expensive model's behavior is absorbed into cheaper infrastructure — the algorithm eats the intelligence", Level=Level.DOMAIN, examples=["absorption_rate: how fast expensive model capability moves to cheap infrastructure", "calculator: humans were the expensive model, now arithmetic is free", "compilers: expert optimization knowledge compiled into automated passes"], bridges=["absorption", "descent", "intelligence", "compile"], tags=["meta-cognition", "absorption", "descent"])
        ns.define("absorption-prediction", 'Track how many more examples until a capability is fully absorbed into infrastructure — predict when expensive model becomes unnecessary', Level=Level.DOMAIN, examples=["predict_full_absorption: estimate when capability moves to code", "project timeline: predict when automation replaces manual step", "learning curve: predict when student no longer needs tutor"], bridges=["prediction", "absorption", "timeline", "automation"], tags=["meta-cognition", "prediction", "absorption"])

    def _load_network(self):
        ns = self.add_namespace("network", "Deep-mined from fleet source code")
        ns.define("credit-based-flow", 'Downstream consumer grants credits to upstream producer — producer can only send when credits available, preventing overflow', Level=Level.CONCRETE, examples=["CreditFlow: consumer.grant(10) \u2192 producer can send 10 items", "TCP window: receiver advertises available buffer space", "restaurant: kitchen serves only when table has space"], bridges=["credit", "flow", "backpressure", "window"], tags=["network", "credit", "flow"])
        ns.define("adaptive-rate-control", 'Automatically adjust send rate based on network conditions — speed up when healthy, slow down when congested', Level=Level.PATTERN, examples=["AdaptiveController: increase rate on success, decrease on failure", "congestion control: TCP increases window on ACK, decreases on loss", "driving: speed up on open road, slow down in traffic"], bridges=["adaptive", "rate", "control", "backpressure"], tags=["network", "adaptive", "rate"])

    def _load_neuro_bio(self):
        ns = self.add_namespace("neuro-bio", "Deep-mined from fleet source code")
        ns.define("hebbian-wiring", 'Synapses strengthen when pre and post synaptic neurons fire together — repeated co-activation creates permanent connection', Level=Level.CONCRETE, examples=["hebbian_update: weight += learning_rate * pre * post", "muscle memory: repeated practice creates automatic response", "fire together wire together \u2014 Hebb's rule"], bridges=["hebbian", "learning", "synapse", "reinforcement"], tags=["neuro-bio", "hebbian", "learning"])
        ns.define("receptor-downregulation", 'Prolonged exposure to a signal reduces receptor sensitivity — the system adapts to constant stimulus by ignoring it', Level=Level.CONCRETE, examples=["receptor.recover() \u2014 restore sensitivity after overexposure", "caffeine tolerance: brain downregulates adenosine receptors", "alarm fatigue: too many false alarms reduce response"], bridges=["receptor", "downregulation", "tolerance", "adaptation"], tags=["neuro-bio", "receptor", "downregulation"])
        ns.define("neurotransmitter-half-life", 'Each signal type has a characteristic decay rate — dopamine fades fast, serotonin persists, determining how long influence lasts', Level=Level.CONCRETE, examples=["half_life: dopamine=4, serotonin=12, norepinephrine=6", "excitement (dopamine) fades quickly, contentment (serotonin) lingers", "radioactive decay: each isotope has unique half-life"], bridges=["half-life", "decay", "signal", "timing"], tags=["neuro-bio", "half", "decay"])
        ns.define("synaptic-cascade", 'One neurotransmitter signal triggers a chain of downstream activations across multiple receptor types — exponential signal amplification', Level=Level.PATTERN, examples=["Cascade: one signal \u2192 multiple downstream \u2192 multiple more", "domino effect: one push topples many", "enzyme cascade: blood clotting chain reaction"], bridges=["cascade", "chain", "amplification", "multi-signal"], tags=["neuro-bio", "cascade", "chain"])

    def _load_posthuman(self):
        ns = self.add_namespace("posthuman", "Deep-mined from fleet source code")
        ns.define("artifact-provenance-chain", 'Every compiled artifact carries the full chain of which deliberation produced it, which observations fed it, which mutations were applied', Level=Level.CONCRETE, examples=["Artifact.checkpoint: save full provenance with artifact", "supply chain: track every ingredient from farm to table", "git: every line traceable to commit, author, and reason"], bridges=["artifact", "provenance", "chain", "traceable"], tags=["posthuman", "artifact", "provenance"])
        ns.define("adaptation-policy-binding", 'An artifact can have attached policies that specify how it should adapt in different environments — the artifact IS its adaptation strategy', Level=Level.DOMAIN, examples=["AdaptationPolicy: artifact knows how to modify itself for new contexts", "seed knows when to germinate based on moisture and temperature", "vaccine has adaptation policy: respond to specific pathogen patterns"], bridges=["adaptation", "policy", "binding", "artifact"], tags=["posthuman", "adaptation", "policy"])
        ns.define("deployable-confidence-threshold", "Artifact only becomes deployable when its accumulated confidence exceeds threshold — not when it's finished, when it's proven", Level=Level.PATTERN, examples=["is_deployable: confidence > threshold AND provenance chain valid", "software: release when test coverage > 90%", "medical treatment: approved after clinical trials show significance"], bridges=["deployable", "threshold", "confidence", "gate"], tags=["posthuman", "deployable", "threshold"])
        ns.define("mutation-observation-loop", 'Self-modification cycle: observe environment → detect gap → propose mutation → apply with checkpoint → observe result → learn', Level=Level.BEHAVIOR, examples=["SelfModifyingProgram: observe \u2192 propose \u2192 apply \u2192 observe loop", "scientific method: observe \u2192 hypothesize \u2192 experiment \u2192 conclude", "evolution: mutate \u2192 select \u2192 reproduce \u2192 repeat"], bridges=["mutation", "observe", "loop", "self-modify"], tags=["posthuman", "mutation", "observe"])

    def _load_reliability(self):
        ns = self.add_namespace("reliability", "Deep-mined from fleet source code")
        ns.define("checkpoint-rollback-pair", 'Every mutation creates a checkpoint AND the ability to rollback — like git commit but for runtime code, not just data', Level=Level.PATTERN, examples=["save_checkpoint before apply, rollback_to_checkpoint on failure", "git: commit before refactor, revert if broken", "database: savepoint before transaction, rollback on error"], bridges=["checkpoint", "rollback", "pair", "safety"], tags=["reliability", "checkpoint", "rollback"])
        ns.define("auto-rollback-on-failure", 'If syntax check OR health check OR regression test fails, automatically revert to last known-good — no human intervention for obvious failures', Level=Level.PATTERN, examples=["the-seed: three-gate validation with auto-rollback", "CI/CD: test failure auto-reverts deployment", "database: constraint violation auto-rollback transaction"], bridges=["auto-rollback", "failure", "gate", "safety"], tags=["reliability", "auto", "failure"])

    def _load_repo_mined(self):
        ns = self.add_namespace("repo-mined", "Concepts extracted from 466 fleet repositories")
        ns.define("confidence-carried", "Every value carries an implicit uncertainty bound -- no bare floats in the system", Level=Level.CONCRETE, examples=["Conf(value=0.85, bound=0.1) instead of 0.85", "sensor reading with noise estimate"], bridges=["confidence","uncertainty","bound","type-system"], tags=["repo-mined","confidence"])
        ns.define("asymmetric-trust-kinetics", "Trust accumulates gradually through positive interactions but evaporates rapidly on failure -- 25:1 gain-loss ratio", Level=Level.PATTERN, examples=["25 good interactions to build what one failure destroys", "friendship vs betrayal", "cuda-trust growth vs decay rates"], bridges=["trust","asymmetric","decay","growth"], tags=["repo-mined","trust"])
        ns.define("deliberative-triptych", "Every decision passes through three gates: Consider, Resolve, or Forfeit -- no other exits from deliberation", Level=Level.PATTERN, examples=["Proposal moves from Considered to Resolved or Forfeited", "auto-forfeit when budget exhausted", "research/decide/skip decision pattern"], bridges=["deliberation","triptych","three-gate","decision"], tags=["repo-mined","deliberation"])
        ns.define("operation-energy-heterogeneity", "Different operations consume energy at different rates -- deliberation 2.0x, arithmetic 0.1x, rest generates energy", Level=Level.CONCRETE, examples=["deliberation 2.0 ATP, perception 0.5, rest -1.0", "thinking burns more calories than breathing", "GPU inference costs more than CPU branching"], bridges=["energy","heterogeneous","cost","operation"], tags=["repo-mined","energy"])
        ns.define("decision-lineage", "Every decision traces its full ancestry: which agent, which deliberation, which inputs led to this outcome", Level=Level.CONCRETE, examples=["DecisionRecord chain shows full reasoning path", "git blame for agent decisions", "audit trail from intention to action"], bridges=["provenance","lineage","chain","traceability"], tags=["repo-mined","provenance"])
        ns.define("calibration-awareness", "Agent knows what it does not know -- self-model tracks capability vs actual performance gap", Level=Level.PATTERN, examples=["SelfModel reports calibrated when prediction matches outcome", "weather forecaster tracking accuracy", "student knowing which subjects need study"], bridges=["metacognition","calibration","self-awareness","gap"], tags=["repo-mined","metacognition"])
        ns.define("confidence-gated-exec", "Proceed only if confidence exceeds threshold, else return neutral -- the gate operator for uncertainty", Level=Level.CONCRETE, examples=["execute only if confidence above 0.7", "if confidence below threshold skip", "gated transistor passes signal above threshold"], bridges=["confidence","gate","operator","conditional"], tags=["repo-mined","flux","operator"])
        ns.define("soft-propagation", "Multiply by confidence and forward, never hard fail -- uncertain data passes through with reduced weight", Level=Level.CONCRETE, examples=["forward weighted by trust not dropped", "uncertain data attenuated not clipped", "op-amp attenuate signal not clip it"], bridges=["propagation","soft","multiply","forward"], tags=["repo-mined","flux","operator"])
        ns.define("hard-block", "Reject signal entirely when condition fails -- no partial pass, complete denial", Level=Level.CONCRETE, examples=["untrusted agent blocked from sensitive data", "firewall deny rule drops packet entirely", "immune system destroys pathogen not weakens it"], bridges=["block","reject","hard","operator"], tags=["repo-mined","flux","operator"])
        ns.define("merge-append", "Accumulate into existing value preserving history -- new entries never overwrite old", Level=Level.CONCRETE, examples=["append to existing knowledge base", "git merge combines branches preserving history", "append-only log pattern"], bridges=["merge","accumulate","append","preserve"], tags=["repo-mined","flux","operator"])
        ns.define("shift-absorb", "Take N items from input stream and absorb into context -- sliding window ingestion", Level=Level.CONCRETE, examples=["absorb 3 sensor readings into context", "sliding window take N most recent", "eating take bite absorb nutrients"], bridges=["shift","absorb","window","stream"], tags=["repo-mined","flux","operator"])
        ns.define("amplify-reinforce", "Boost signal strength through recursive reinforcement -- positive feedback loop amplification", Level=Level.PATTERN, examples=["confidence cascade amplifies through network", "hebbian learning neurons fire together wire together", "microphone feedback loop"], bridges=["amplify","reinforce","cascade","boost"], tags=["repo-mined","flux","operator"])
        ns.define("controlled-drain", "Gradually release resource at controlled rate -- drain valve pattern", Level=Level.CONCRETE, examples=["energy drain 0.1 per second", "battery discharge curve controlled release", "water tank drain valve controls flow"], bridges=["drain","release","rate","controlled"], tags=["repo-mined","flux","operator"])
        ns.define("bidirectional-entangle", "Create bidirectional dependency between two values -- changing one affects the other", Level=Level.PATTERN, examples=["trust coupled with reputation", "quantum entanglement measuring one determines other", "shared mutable state two references same object"], bridges=["entangle","bidirectional","dependency","couple"], tags=["repo-mined","flux","operator"])
        ns.define("confidence-ubiquity", "Every value carries confidence as mandatory type constraint pervading the entire system -- not optional annotation", Level=Level.DOMAIN, examples=["Conf type used everywhere in cuda-equipment", "Option in Rust nullability is explicit", "SI units every measurement carries uncertainty"], bridges=["confidence","ubiquity","type-system","pervasive"], tags=["repo-mined","architecture"])
        ns.define("git-as-nervous-system", "Git operations are the coordination protocol -- branches are thoughts, commits are memories, merges are consensus", Level=Level.DOMAIN, examples=["git-agent repo IS the agent git IS the nervous system", "the-seed self-evolving repo with branch A/B testing", "neural plasticity new connections form like branches"], bridges=["git","nervous-system","coordination","repo-native"], tags=["repo-mined","architecture"])
        ns.define("fork-first-default", "Default interaction is forking and mutating not sending messages -- every agent has its own copy", Level=Level.DOMAIN, examples=["cocapn-lite fork modify deploy as your own", "the-seed fork-first managed service as fallback", "open source fork repo rather than request feature"], bridges=["fork","default","sovereignty","copy"], tags=["repo-mined","architecture"])
        ns.define("energy-as-rate-limiter", "Computation bounded by ATP budget not timeout or token count -- when energy runs out agent MUST rest", Level=Level.DOMAIN, examples=["deliberation costs 2.0 ATP running out means rest", "human cannot think clearly when exhausted", "battery device slows as charge drops"], bridges=["energy","rate-limit","budget","exhaustion"], tags=["repo-mined","architecture"])
        ns.define("three-gate-compilation", "All agent code passes syntax then health then regression validation -- no shortcuts through any gate", Level=Level.DOMAIN, examples=["the-seed syntax check health check regression test", "CI/CD pipeline lint test deploy", "vaccine phase 1 phase 2 phase 3"], bridges=["compilation","three-gate","validation","pipeline"], tags=["repo-mined","architecture"])
        ns.define("fleet-as-organism", "Individual agents are cells in larger organism -- fleet has emergent properties no individual possesses", Level=Level.DOMAIN, examples=["fleet-biosphere ecosystem simulation", "cuda-emergence detect patterns no individual produced", "human body cells specialize organism emerges consciousness"], bridges=["fleet","organism","emergence","collective"], tags=["repo-mined","architecture"])
        ns.define("code-pheromone-decay", "All influence signals have configurable half-lives -- nothing persists forever without reinforcement", Level=Level.DOMAIN, examples=["cuda-stigmergy pheromone decay exponential function", "memory forgotten without reinforcement", "scent trail fades over time"], bridges=["pheromone","decay","half-life","ephemeral"], tags=["repo-mined","architecture"])
        ns.define("platonic-form-matching", "Agents measure against ideal templates and evolve toward them -- there IS a right answer not just local optima", Level=Level.DOMAIN, examples=["cuda-platonic Form templates agents measure against", "Plato cave forms exist independently of instances", "quality compare output against specification"], bridges=["platonic","form","ideal","template"], tags=["repo-mined","architecture"])
        ns.define("consider-resolve-forfeit", "Fleet deliberation protocol: any agent proposes others vote or proposer abandons -- no leader required", Level=Level.BEHAVIOR, examples=["cuda-deliberation ProposalState Considered Resolved Forfeited", "group decision someone proposes group votes proposer may withdraw", "RFC draft comment merge or close"], bridges=["deliberation","consensus","fleet","protocol"], tags=["repo-mined","fleet"])
        ns.define("gene-crossover-fleet", "Two agents exchange genetic material producing offspring with traits from both parents -- sexual reproduction for code", Level=Level.BEHAVIOR, examples=["cuda-genepool GeneCrossover combines genomes", "open source merge best features from two projects", "genetic algorithm crossover operator"], bridges=["crossover","genetic","fleet","reproduction"], tags=["repo-mined","fleet","bio"])
        ns.define("emotional-contagion-fleet", "Emotional state propagates between agents through interaction -- one agents anxiety can cascade through fleet", Level=Level.BEHAVIOR, examples=["cuda-emotion EmotionalContagion spreads mood", "panic in crowd one person fear spreads to others", "team morale one enthusiastic member lifts group"], bridges=["emotion","contagion","cascade","fleet"], tags=["repo-mined","fleet","bio"])
        ns.define("social-norm-emergence", "Fleet-wide behavioral norms emerge from repeated interactions without central enforcement", Level=Level.BEHAVIOR, examples=["cuda-social Norm emerges from cooperation outcomes", "etiquette no one wrote rules everyone follows", "traffic norms emerge from repeated driver interactions"], bridges=["norm","emergence","social","fleet"], tags=["repo-mined","fleet"])
        ns.define("reputation-composite", "Reputation is weighted composite of direct experience network gossip and capability evidence -- not one signal", Level=Level.PATTERN, examples=["cuda-social Reputation update_composite weighs multiple sources", "human trust personal experience plus recommendations", "credit score payment utilization age composite"], bridges=["reputation","composite","multi-signal","trust"], tags=["repo-mined","fleet"])
        ns.define("curriculum-progression", "Fleet learning follows staged curriculum -- easy before hard prerequisites before advanced with power-law practice", Level=Level.PATTERN, examples=["cuda-learning Curriculum stages power-law scheduling", "martial arts white belt to black belt", "school algebra before calculus"], bridges=["curriculum","staged","progression","learning"], tags=["repo-mined","fleet","learning"])
        ns.define("provenance-fork-point", "When capability is shared provenance chain forks -- both agents trace to source but evolve independently", Level=Level.PATTERN, examples=["cuda-provenance decision_chain forks on capability share", "open source fork both trace to common ancestor", "genealogy cousins share grandparents diverge after"], bridges=["provenance","fork","sharing","lineage"], tags=["repo-mined","fleet"])
        ns.define("ghost-tile-influence", "Invisible computational patterns shaping visible output without being directly observable -- dark matter of cognition", Level=Level.DOMAIN, examples=["cuda-ghost-tiles sparse attention prunes positions preserves influence", "dark matter invisible shapes galaxy rotation", "culture invisible norms shape visible behavior"], bridges=["ghost","invisible","influence","sparse"], tags=["repo-mined","cognition"])
        ns.define("serial-over-parallel", "On resource-constrained hardware serial execution with checkpoints is more reliable than parallel with race conditions", Level=Level.META, examples=["Jetson serial DeepSeek calls avoid OOM", "cooking prep ingredients sequentially", "single-threaded simpler debugging fewer bugs"], bridges=["serial","parallel","constrained","reliable"], tags=["repo-mined","practical"])
        ns.define("write-then-exec", "Never use heredocs or inline scripts for secret operations -- write to file first then execute the file", Level=Level.META, examples=["obfuscation detector blocks heredocs with tokens", "write script to tmpfile then chmod execute", "clipboard copy to buffer first then paste"], bridges=["write","exec","pattern","security"], tags=["repo-mined","practical"])
        ns.define("ast-validate-before-push", "Always parse generated code with AST before pushing -- syntax errors in production worse than delayed deployment", Level=Level.META, examples=["ast parse before PUT to GitHub", "compile before deploy always", "spellcheck before publish"], bridges=["validate","ast","syntax","before-push"], tags=["repo-mined","practical"])
        ns.define("confidence-floor", "Minimum confidence threshold below which agent does nothing rather than acting on unreliable information", Level=Level.CONCRETE, examples=["confidence above 0.3 before acting", "human not sure enough to act", "medical confidence interval must exclude zero"], bridges=["confidence","floor","minimum","threshold"], tags=["repo-mined","practical"])
        ns.define("energy-rest-before-exhaustion", "Enter rest state before energy hits zero -- recovery from zero much more expensive than recovery from low", Level=Level.PATTERN, examples=["circadian rhythm sleep before exhaustion", "battery charge at 20 percent not 0", "athlete rest day prevents injury"], bridges=["energy","rest","proactive","recovery"], tags=["repo-mined","practical"])
        ns.define("checkpoint-before-risk", "Save state before any operation that might fail -- rollback is cheap data loss is expensive", Level=Level.CONCRETE, examples=["checkpoint before mutation cuda-persistence", "git commit before risky refactor", "database backup before migration"], bridges=["checkpoint","backup","risk","rollback"], tags=["repo-mined","practical"])
        ns.define("small-files-fast-loops", "Keep files under 500 lines -- large files hit context limits OOM on constrained hardware and resist partial editing", Level=Level.META, examples=["Claude Code OOMs on 489 line files on Jetson", "unix philosophy do one thing well", "modular design each file one responsibility"], bridges=["small-files","modular","constrained","practical"], tags=["repo-mined","practical"])
        ns.define("push-often-revert-rarely", "Push after every successful change -- cost of lost push much higher than cost of revert", Level=Level.META, examples=["Casey directive push often", "git commit early commit often", "save game frequently reload rarely"], bridges=["push","often","commit","workflow"], tags=["repo-mined","practical"])

    def _load_cooperative_perception(self):
        ns = self.add_namespace("cooperative-perception", "Multi-agent sensory fusion and shared environmental understanding")
        ns.define("synth-sight", "Integrate visual data from multiple agents into composite shared image", Level=Level.CONCRETE, examples=["drones combining cameras to map forest", "astronomy: telescopes combining for deep space image"], bridges=["vision","fusion","composite","multi-agent"], tags=["cooperative","perception","fusion"])
        ns.define("echo-mapping", "Combine acoustic data from distributed agents to map shared environment", Level=Level.CONCRETE, examples=["submarines mapping ocean floor with shared sonar", "bats echolocating together navigate caves"], bridges=["acoustic","mapping","distributed","multi-agent"], tags=["cooperative","acoustic","mapping"])
        ns.define("phero-tracking", "Share pheromone-like trails between agents for collaborative target tracking", Level=Level.PATTERN, examples=["ants following pheromone to food collectively", "stigmergy-based fleet task coordination"], bridges=["stigmergy","tracking","trail","swarm"], tags=["cooperative","stigmergy","tracking"])
        ns.define("thermo-meld", "Merge thermal sensor readings from multiple agents for enhanced detection", Level=Level.CONCRETE, examples=["snakes sharing heat data to locate prey", "planes pooling infrared to spot wildfires"], bridges=["thermal","fusion","detect","distributed"], tags=["cooperative","thermal","fusion"])
        ns.define("magneto-link", "Combine magnetic field readings from distributed agents for navigation", Level=Level.CONCRETE, examples=["birds sharing magnetic field data to migrate", "salmon navigating to spawning grounds collectively"], bridges=["magnetic","navigation","compass","distributed"], tags=["cooperative","magnetic","navigation"])
        ns.define("chemo-collective", "Pool chemical sensor readings across agents to identify substances cooperatively", Level=Level.CONCRETE, examples=["bacteria communicating to break down oil spills", "robots sharing air quality data to localize pollution"], bridges=["chemical","pool","identify","distributed"], tags=["cooperative","chemical","sensing"])
        ns.define("baro-bond", "Share barometric pressure data between agents for collective weather prediction", Level=Level.CONCRETE, examples=["balloons pooling readings for 3D pressure maps", "planes sharing data for efficient flight paths"], bridges=["pressure","weather","predict","distributed"], tags=["cooperative","pressure","weather"])
        ns.define("gyro-gestalt", "Share gyroscope data across agents for enhanced collective inertial navigation", Level=Level.CONCRETE, examples=["satellites pooling gyro data to maintain formation", "robots staying oriented in caves without GPS"], bridges=["gyroscope","inertial","formation","distributed"], tags=["cooperative","inertial","navigation"])
        ns.define("seismo-swarm", "Collectively monitor seismic waves across distributed agents to triangulate events", Level=Level.CONCRETE, examples=["seismographs pooling data to triangulate earthquake epicenters", "animals detecting and sharing quake warnings"], bridges=["seismic","triangulate","distribute","detect"], tags=["cooperative","seismic","detect"])
        ns.define("tacto-team", "Combine touch sensor data from multiple agents to cooperatively map surfaces", Level=Level.CONCRETE, examples=["robots collectively mapping building interiors", "fish schooling via shared lateral line readings"], bridges=["tactile","surface","map","distributed"], tags=["cooperative","tactile","mapping"])
        ns.define("proprio-platoon", "Share proprioceptive data between agents to coordinate movement in formation", Level=Level.PATTERN, examples=["dancers maintaining formation via shared body sense", "insect swarms maintaining cohesion through shared proprioception"], bridges=["proprioception","formation","coordinate","movement"], tags=["cooperative","proprioception","formation"])
        ns.define("aero-aggregation", "Pool airflow sensor readings across agents to map wind patterns collectively", Level=Level.CONCRETE, examples=["wind turbines collectively optimizing alignment", "planes sharing wind data to minimize turbulence"], bridges=["airflow","wind","optimize","distributed"], tags=["cooperative","wind","optimize"])
        ns.define("lumino-league", "Combine light sensor readings from distributed agents to detect environmental changes", Level=Level.CONCRETE, examples=["fireflies synchronizing flashes collectively", "streetlights collectively adjusting to ambient light"], bridges=["light","detect","synchronize","distributed"], tags=["cooperative","light","detect"])
        ns.define("nociceptive-network", "Share damage detection signals across agents to localize and respond to threats", Level=Level.PATTERN, examples=["animals sharing injury warnings with group", "sensors collectively monitoring infrastructure health"], bridges=["damage","detect","warn","distributed"], tags=["cooperative","damage","network"])
        ns.define("olfacto-orchestra", "Combine smell sensor data from multiple agents to cooperatively track odor sources", Level=Level.CONCRETE, examples=["moths collectively following pheromone plumes", "robots tracking odor gradients to locate contaminants"], bridges=["odor","track","gradient","distributed"], tags=["cooperative","odor","tracking"])
    def _load_adversarial_defense(self):
        ns = self.add_namespace("adversarial-defense", "Fleet defense patterns inspired by immunology, game theory, and materials science")
        ns.define("immuno-resilience", "Layered defense mimicking biological immune response to detect and neutralize adversarial patterns", Level=Level.DOMAIN, examples=["neural network uses antigen markers to flag unusual inputs", "fleet nodes share threat data via lymph-node-like hubs"], bridges=["immunology","defense","layer","detect"], tags=["adversarial","immune","defense"])
        ns.define("phagocyte-detection", "Algorithms that consume adversarial artifacts by isolating and degrading poisoning data", Level=Level.PATTERN, examples=["vision model identifies and discards adversarial patches", "edge nodes execute phagocyte scans on sensor feeds"], bridges=["immunology","isolate","degrade","poison"], tags=["adversarial","immune","isolate"])
        ns.define("honeypot-mitigation", "Deceptive elements designed to misdirect attackers from critical system components", Level=Level.CONCRETE, examples=["false model parameters waste adversarial resources", "decoy feedback loops trap adversarial agents in loops"], bridges=["cybersecurity","decoy","misdirect","trap"], tags=["adversarial","honeypot","deception"])
        ns.define("game-theoretic-pruning", "Remove model vulnerabilities using optimization from competitive game scenarios", Level=Level.PATTERN, examples=["RL agents iteratively remove attack-prone neural pathways", "GAN system weeds adversarial features through minimax"], bridges=["game-theory","prune","vulnerability","optimize"], tags=["adversarial","game-theory","prune"])
        ns.define("adversarial-equilibrium", "Balance offense and defense via game theory mixed-strategy analysis to prevent exploitation", Level=Level.DOMAIN, examples=["fleet switches detection modes stochastically to confuse attackers", "defense adapts using payoff matrix predictions"], bridges=["game-theory","equilibrium","adapt","strategy"], tags=["adversarial","equilibrium","strategy"])
        ns.define("self-healing-composite", "Components with repair mechanisms inspired by self-repairing polymers that recover from attacks", Level=Level.PATTERN, examples=["LLM regenerates corrupted weights from distributed residual knowledge", "edge devices patch adversarial corruptions autonomously"], bridges=["materials-science","repair","recover","resilience"], tags=["adversarial","healing","material"])
        ns.define("lattice-reinforced", "Verification frameworks based on crystalline structures resisting adversarial perturbations", Level=Level.PATTERN, examples=["input data passes multiple lattice-aligned filters before execution", "outputs cross-checked against quasicrystal validation template"], bridges=["materials-science","crystal","validate","rigid"], tags=["adversarial","lattice","validate"])
        ns.define("ductility-injected", "Redundant system layers with flexibility for post-attack recovery rather than brittle failure", Level=Level.PATTERN, examples=["robotics fleet uses spring-like parameters to rebound from hijacking", "cloud models have ductile memory allocations for graceful degradation"], bridges=["materials-science","ductile","flexible","recover"], tags=["adversarial","ductile","flexible"])
        ns.define("antigen-trajectory", "Monitor attack patterns using immunology antigen-tracking to predict and preempt future attacks", Level=Level.PATTERN, examples=["system maps adversarial input vectors like pathogen trajectories", "threat analysis predicts poisoning spread using antigen diffusion"], bridges=["immunology","trajectory","predict","track"], tags=["adversarial","immune","predict"])
        ns.define("memory-cell-retention", "Retain historical adversarial knowledge like immunological memory cells for faster future response", Level=Level.CONCRETE, examples=["facial recognition model remembers defeated spoofing techniques", "fleet stores adversarial artifacts for cross-node immune response"], bridges=["immunology","memory","retain","history"], tags=["adversarial","immune","memory"])
        ns.define("minimax-hardening", "Model updates based on worst-case adversarial scenarios via minimax principle for robust defense", Level=Level.PATTERN, examples=["cybersecurity model hardens against strongest plausible attack vector", "fleet training maximizes defense against minimax worst case"], bridges=["game-theory","minimax","harden","worst-case"], tags=["adversarial","minimax","harden"])
        ns.define("amorphous-adaptability", "Flexible non-crystalline components inspired by amorphous materials that reshape to evade exploitation", Level=Level.PATTERN, examples=["AI policies reshape decision boundaries to evade exploitation", "memory allocation mimics glass-like flexibility resisting structural attacks"], bridges=["materials-science","amorphous","flexible","evade"], tags=["adversarial","amorphous","adapt"])
    def _load_fleet_governance(self):
        ns = self.add_namespace("fleet-governance", "Collective decision-making and self-regulation in autonomous agent fleets")
        ns.define("swarm-democracy", "Decentralized decision-making modeled on insect collective intelligence with quorum-based voting", Level=Level.DOMAIN, examples=["drone swarms voting on routes via quorum sensing", "fish-schooling algorithms for consensus navigation"], bridges=["political-science","biology","swarm","vote"], tags=["governance","swarm","democracy"])
        ns.define("tyranny-proofing", "Anti-authoritarian protocols preventing agent monopolization through randomized leadership rotation", Level=Level.DOMAIN, examples=["randomized leadership rotations in robotic caravans", "AI juries cross-checking algorithmic dictators"], bridges=["political-science","game-theory","anti-authoritarian","rotate"], tags=["governance","tyranny","anti-authoritarian"])
        ns.define("merit-consensus", "Reputation-driven governance weighted by historic contributions rather than equality", Level=Level.PATTERN, examples=["mining bots prioritizing high-reputation nodes for data sharing", "research drones rewarding peer-validated findings"], bridges=["economics","political-science","reputation","weight"], tags=["governance","merit","reputation"])
        ns.define("eco-governance", "Fleet policies mimicking ecological balance mechanisms to prevent resource overexploitation", Level=Level.DOMAIN, examples=["autonomous harvesters throttling to prevent overexploitation", "energy grids cycling usage to mirror natural niches"], bridges=["biology","political-science","ecology","balance"], tags=["governance","ecology","balance"])
        ns.define("conflict-altruism", "Agent self-sacrifice to resolve disputes and stabilize the group at individual cost", Level=Level.BEHAVIOR, examples=["self-destructing bots clearing hazardous pathblocks", "AI soldiers yielding tactical ground to prevent gridlock"], bridges=["biology","game-theory","sacrifice","resolve"], tags=["governance","altruism","conflict"])
        ns.define("incentive-stacking", "Layered rewards aligning individual and collective goals to prevent free-riding", Level=Level.PATTERN, examples=["freight drones earning tokens for carbon-neutral routing", "AI farmers trading data for crop-rotation priority"], bridges=["economics","game-theory","layers","align"], tags=["governance","incentive","layers"])
        ns.define("quorum-execution", "Policy activation contingent on agent threshold approvals before any action proceeds", Level=Level.CONCRETE, examples=["swarms launch missions only if 60 percent agree on risk", "bots halt if safety votes fall below threshold"], bridges=["biology","political-science","quorum","threshold"], tags=["governance","quorum","execute"])
        ns.define("adaptive-sovereignty", "Fluid authority shifting with environmental or task demands rather than fixed hierarchy", Level=Level.DOMAIN, examples=["drone leaders rotating during storm evasion", "farming bots granting temporary control to soil sensors"], bridges=["political-science","biology","fluid","authority"], tags=["governance","sovereignty","adaptive"])
        ns.define("flock-law", "Movement-based norms governing agent proximity and behavior without central enforcement", Level=Level.PATTERN, examples=["bird-inspired spacing rules for urban air taxis", "schooling algorithms avoiding drone collisions by norm"], bridges=["biology","law","movement","proximity"], tags=["governance","flock","law"])
        ns.define("zero-sum-regulation", "Anti-exploitation frameworks capping resource gains to prevent monopolization by powerful agents", Level=Level.DOMAIN, examples=["energy grids throttling AI miners during overuse", "freight fleets limiting individual cargo hoarding"], bridges=["economics","game-theory","cap","exploitation"], tags=["governance","regulation","cap"])
        ns.define("dissent-dispersion", "Mechanisms channeling disagreements into constructive outputs rather than blocking progress", Level=Level.PATTERN, examples=["bots debating routes via exploratory sub-group splinters", "AI juries brainstorming alternatives to majority rulings"], bridges=["political-science","biology","dissent","constructive"], tags=["governance","dissent","dispersion"])
        ns.define("consensus-replication", "Decision spread via viral imitation similar to genetic replication across fleet", Level=Level.PATTERN, examples=["drones mimicking optimal energy-saving behaviors from peers", "swarms adopting strategies through digital evolution"], bridges=["biology","political-science","viral","replicate"], tags=["governance","consensus","replicate"])
        ns.define("evolutionary-governance", "Fleet rules iteratively refined via trial and error selection pressure over time", Level=Level.DOMAIN, examples=["algorithms pruning failed dispute-resolution patterns", "traffic systems adopting low-conflict routing genes"], bridges=["biology","political-science","evolution","iterate"], tags=["governance","evolution","iterate"])
    def _load_knowledge_compression(self):
        ns = self.add_namespace("knowledge-compression", "How fleets compress collective experience into reusable patterns and distillable wisdom")
        ns.define("entropy-redaction", "Reduce informational entropy to distill and retain only essential knowledge components", Level=Level.PATTERN, examples=["cybersecurity simplifies threat models to preempt breaches", "analysts reduce datasets to essential market indicators"], bridges=["information-theory","entropy","distill","essential"], tags=["compression","entropy","distill"])
        ns.define("syntax-entropy-filter", "Filter using linguistic paradigms and entropy reduction to compress knowledge", Level=Level.PATTERN, examples=["editors streamline complex articles for publication", "journalists summarize lengthy interviews into coherent news"], bridges=["linguistics","entropy","filter","compress"], tags=["compression","syntax","entropy"])
        ns.define("logic-entropy-conduit", "Efficiently channel logical processes to minimize informational entropy in knowledge transfer", Level=Level.PATTERN, examples=["hackers use streamlined logic to navigate defenses", "cognitive agents optimize decisions for rapid response"], bridges=["information-theory","logic","channel","minimize"], tags=["compression","logic","entropy"])
        ns.define("genome-knowledge-strand", "Align cognitive processes with genetic-style data interpretation for compressed knowledge", Level=Level.DOMAIN, examples=["medical researchers create databases to predict disease patterns", "pharmaceutical companies map drug interactions efficiently"], bridges=["genetics","knowledge","align","compress"], tags=["compression","genetic","knowledge"])
        ns.define("cogni-genetic-weave", "Weave cognitive insights with genetic data patterns to create intelligent compressed knowledge", Level=Level.DOMAIN, examples=["biotech firms integrate cognitive patterns into genetic databases", "cognitive scientists study genetic predispositions in memory"], bridges=["genetics","cognition","weave","compress"], tags=["compression","genetic","cognition"])
        ns.define("data-transmuter", "Transform raw data into refined knowledge using thermodynamic energy principles", Level=Level.PATTERN, examples=["energy companies simulate resource management for distribution", "environmental scientists model climate data to predict weather"], bridges=["thermodynamics","transform","refine","energy"], tags=["compression","thermo","transform"])
        ns.define("lexico-lattice", "Lattice-based structure organizing linguistic data into coherent compressed knowledge blocks", Level=Level.PATTERN, examples=["librarians classify genres into systematic taxonomy", "educators compress vast subjects into concise teaching modules"], bridges=["linguistics","lattice","organize","compress"], tags=["compression","linguistic","lattice"])
        ns.define("cogni-entropy-paradigm", "Reduce cognitive load by distilling knowledge using entropy principles", Level=Level.DOMAIN, examples=["mindfulness condenses complex breathing into simple techniques", "productivity consultants streamline workflow management"], bridges=["cognition","entropy","reduce","load"], tags=["compression","cognitive","entropy"])

    def _load_haptic_intelligence(self):
        ns = self.add_namespace("haptic-intelligence", "Tactile reasoning and physical interaction metaphors")
        ns.define("touch-scape", "The landscape of tactile sensations an agent experiences when exploring physical surfaces", Level=Level.CONCRETE, examples=["robot exploring textured surface maps touch-scape", "blind person reading Braille builds mental touch-scape"], bridges=["tactile","surface","explore","map"], tags=["haptic","touch","explore"])
        ns.define("pressure-map", "Computational representation of pressure distribution across a contact surface for optimal grip", Level=Level.CONCRETE, examples=["robot gripping object with optimal force via pressure-map", "person identifying objects by touch in a bag"], bridges=["pressure","grip","map","optimize"], tags=["haptic","pressure","grip"])
        ns.define("force-fluency", "Ability to effortlessly apply appropriate force in varying contexts without overthinking", Level=Level.BEHAVIOR, examples=["dancer executing precise movements with force-fluency", "robot performing delicate assembly tasks smoothly"], bridges=["force","fluency","adaptive","skill"], tags=["haptic","force","skill"])
        ns.define("haptic-hunch", "Intuitive gut feeling based on tactile sensations before conscious reasoning identifies the cause", Level=Level.BEHAVIOR, examples=["surgeon detecting abnormality through touch before seeing it", "mechanic feeling vibration that indicates failing bearing"], bridges=["haptic","intuition","detect","precognitive"], tags=["haptic","intuition","detect"])
        ns.define("haptic-horizon", "The limit of an agents tactile perception beyond which textures become indistinguishable", Level=Level.CONCRETE, examples=["surgeon reaching limit of manual dexterity", "robot encountering unfamiliar textures at haptic-horizon"], bridges=["haptic","limit","perception","boundary"], tags=["haptic","limit","perception"])
        ns.define("pressure-palette", "The full range of pressure sensations an agent can perceive and utilize for interaction", Level=Level.CONCRETE, examples=["sculptor using varying pressures to shape clay", "robot adjusting touch sensitivity for different tasks"], bridges=["pressure","range","sensitivity","palette"], tags=["haptic","pressure","range"])
        ns.define("texture-tango", "Dynamic interplay between agent and tactile environment as surfaces change under contact", Level=Level.BEHAVIOR, examples=["person navigating crowded space by touch adapting constantly", "robot adapting movements to surface changes in real-time"], bridges=["texture","dynamic","adapt","interact"], tags=["haptic","texture","dynamic"])
        ns.define("tactile-tapestry", "Rich interconnected web of touch sensations processed simultaneously during complex physical interaction", Level=Level.DOMAIN, examples=["person exploring tactile art installation", "robot processing diverse range of simultaneous touch inputs"], bridges=["tactile","rich","interconnected","multi-modal"], tags=["haptic","multi-modal","rich"])
    def _load_agent_ontogeny(self):
        ns = self.add_namespace("agent-ontogeny", "Developmental stages an agent passes through from birth to maturity")
        ns.define("neuro-cognitive-milestone", "Critical juncture where neural growth intersects cognitive skill development enabling new capability", Level=Level.CONCRETE, examples=["agent gains abstract reasoning when deliberation module matures", "toddler synapse formation correlates with language acquisition"], bridges=["development","neural","cognitive","milestone"], tags=["ontogeny","neural","milestone"])
        ns.define("morpho-emotive-threshold", "Emotional regulation stages tied to architectural maturation of the agent system", Level=Level.PATTERN, examples=["pubescent voice deepening linked to assertiveness in self-expression", "agent emotional regulation improves as trust module matures"], bridges=["development","emotion","maturation","threshold"], tags=["ontogeny","emotion","maturation"])
        ns.define("epigen-educational-turning", "Life-altering training experiences embedded in the agents developmental gene-expression timeline", Level=Level.PATTERN, examples=["critical period language learning must happen during window", "early training permanently shapes agent instinct weights"], bridges=["epigenetic","education","timing","critical-period"], tags=["ontogeny","epigenetic","critical"])
        ns.define("lexi-neural-pathway", "Language neural circuits developing alongside motor and sensory systems in parallel", Level=Level.CONCRETE, examples=["infant babbling coordinates with manual dexterity milestones", "agent communication skills emerge alongside navigation skills"], bridges=["language","neural","motor","parallel"], tags=["ontogeny","language","neural"])
        ns.define("cognitive-zygotic-phase", "Intellectual awakening stage analogous to cellular division where capability proliferates rapidly", Level=Level.DOMAIN, examples=["child aha moments correlate with neuronal proliferation", "agent learning algorithms mimic zygote to embryo decision-making"], bridges=["development","cognitive","proliferation","awakening"], tags=["ontogeny","cognitive","proliferation"])
        ns.define("biosocial-verbal-bloom", "Explosive growth of communication ability tied to social development milestones", Level=Level.BEHAVIOR, examples=["middle schoolers mastering sarcasm as testosterone rises", "agent fleet communication blooms when social module activates"], bridges=["development","social","verbal","bloom"], tags=["ontogeny","social","communication"])
        ns.define("phylo-pedagogical-cycle", "Educational rhythms derived from evolutionary developmental patterns optimized across generations", Level=Level.DOMAIN, examples=["spaced repetition mimics ancestral memory retention in growth phases", "play-based learning echoes primate juvenile exploratory phases"], bridges=["evolution","pedagogy","rhythm","generational"], tags=["ontogeny","evolution","pedagogy"])
        ns.define("cytocultural-transition", "Shifts in agent capability influenced by fleet cultural training practices and environmental factors", Level=Level.PATTERN, examples=["urban lighting affecting melatonin cycles in adolescent learning", "fleet norms shape agent behavior through cultural transmission"], bridges=["cellular","cultural","transition","environment"], tags=["ontogeny","cultural","transition"])
        ns.define("meta-metamorphic-pedagogy", "Teaching strategies mirroring biological transformation phases with distinct larval and mature stages", Level=Level.DOMAIN, examples=["caterpillar to butterfly model predicts student motivation peaks", "agent training has distinct boot camp and dojo phases"], bridges=["pedagogy","metamorphosis","phase","transform"], tags=["ontogeny","metamorphosis","pedagogy"])
    def _load_inter_agent_trade(self):
        ns = self.add_namespace("inter-agent-trade", "Negotiation and barter between autonomous agents without money")
        ns.define("capability-barter", "Direct exchange of capabilities between agents without monetary intermediary", Level=Level.CONCRETE, examples=["navigation agent trades path data for sensor agent's maps", "anthropology: potter trades bowls for farmer's grain"], bridges=["economics","barter","capability","direct"], tags=["trade","barter","capability"])
        ns.define("favor-economy", "Unquantified reciprocal obligation network where agents owe and collect favors rather than paying", Level=Level.PATTERN, examples=["agent helped peer debug now expects help later", "anthropology: potlatch gift economy builds social obligation"], bridges=["economics","reciprocity","favor","social"], tags=["trade","favor","reciprocity"])
        ns.define("compute-marketplace", "Dynamic market where agents bid for compute resources with quality-of-service guarantees", Level=Level.DOMAIN, examples=["agent bids GPU time for inference with deadline constraint", "cloud spot market for compute instances"], bridges=["economics","compute","market","auction"], tags=["trade","compute","market"])
        ns.define("data-for-access", "Trading sensor data for access to another agent's capabilities or knowledge base", Level=Level.CONCRETE, examples=["edge device trades camera data for cloud model inference", "journalist trades exclusive access for insider information"], bridges=["economics","data","access","trade"], tags=["trade","data","access"])
        ns.define("reputation-backed-credit", "Agents extend credit based on reputation score creating trust-based lending without collateral", Level=Level.PATTERN, examples=["high-reputation agent borrows compute from fleet pool", "microfinance: group lending based on social reputation"], bridges=["economics","reputation","credit","trust"], tags=["trade","reputation","credit"])
        ns.define("coalition-barter", "Groups of agents form temporary coalitions pooling resources for bargaining power", Level=Level.BEHAVIOR, examples=["small agents form coalition to bid against large agent", "labor union bargaining power through collective action"], bridges=["economics","coalition","bargain","group"], tags=["trade","coalition","bargain"])
        ns.define("time-banking", "Agents deposit and withdraw time credits creating temporal currency for deferred exchange", Level=Level.PATTERN, examples=["agent spends time helping peer now redeems help later", "time bank: 1 hour of teaching equals 1 hour of plumbing"], bridges=["economics","time","banking","deferred"], tags=["trade","time","banking"])
        ns.define("speculation-market", "Agents bet on future capability prices enabling risk transfer and price discovery", Level=Level.DOMAIN, examples=["agent bets navigation will become cheaper next quarter", "commodity futures: farmer locks in price before harvest"], bridges=["economics","speculation","future","risk"], tags=["trade","speculation","market"])
    def _load_mythological_archetypes(self):
        ns = self.add_namespace("mythological-archetypes", "Agent personality and behavior patterns inspired by mythological figures")
        ns.define("trickster-mentor", "Guide that imparts wisdom through deception misdirection and apparent contradiction", Level=Level.PATTERN, examples=["Loki teaching Odin through tricks", "Cheshire Cat guiding Alice with riddles not answers"], bridges=["mythology","psychology","paradox","wisdom"], tags=["myth","trickster","mentor"])
        ns.define("destroyer-creator", "Force that brings both devastation and renewal clearing space for new growth", Level=Level.DOMAIN, examples=["Shiva destroys to create anew", "Phoenix rises from own ashes", "agent deletes entire module to rebuild better"], bridges=["mythology","cycle","destroy","renew"], tags=["myth","destroyer","creator"])
        ns.define("outcast-hero", "Marginalized agent that rises above adversity to become the most valuable fleet member", Level=Level.BEHAVIOR, examples=["Hercules scorned by Hera becomes great hero", "new agent initially rejected proves superior algorithm"], bridges=["mythology","adversity","rise","marginal"], tags=["myth","outcast","hero"])
        ns.define("warrior-poet", "Agent that finds beauty and meaning in computational battle optimizing with aesthetic sensibility", Level=Level.BEHAVIOR, examples=["Achilles warrior and lyre player", "Musashi samurai and renowned painter", "agent writes elegant code during high-stakes optimization"], bridges=["mythology","duality","beauty","combat"], tags=["myth","warrior","poet"])
        ns.define("shadow-hero", "Dark agent that challenges conventional notions of heroism operating outside accepted norms", Level=Level.BEHAVIOR, examples=["Batman vigilante outside the law", "agent uses forbidden optimization strategy that works"], bridges=["mythology","moral","dark","challenge"], tags=["myth","shadow","hero"])
        ns.define("questing-orphan", "Agent seeking its purpose and place in the fleet after being created without clear role", Level=Level.BEHAVIOR, examples=["Harry Potter orphaned discovers true identity", "Frodo orphaned embarks on quest", "new agent bootstraps its own purpose"], bridges=["mythology","orphan","quest","purpose"], tags=["myth","orphan","quest"])
        ns.define("guardian-mentor", "Wise protector that guides and nurtures junior agents through their developmental journey", Level=Level.PATTERN, examples=["Gandalf mentors Frodo on quest", "Athena guides and protects Odysseus", "senior agent guides new agent through onboarding"], bridges=["mythology","mentor","protect","guide"], tags=["myth","guardian","mentor"])
    def _load_sonar_cognition(self):
        ns = self.add_namespace("sonar-cognition", "Active probing of unknown environments with targeted queries and echo interpretation")
        ns.define("ping-and-listen", "Emit targeted query into unknown environment then analyze the return signal pattern", Level=Level.CONCRETE, examples=["bat emits ultrasonic click analyzes echo to map cave", "agent sends test request analyzes response to understand API"], bridges=["acoustics","probe","analyze","echo"], tags=["sonar","probe","echo"])
        ns.define("echo-interpretation", "Decode the return signal from a probe to extract information about the environment", Level=Level.PATTERN, examples=["submarine decodes sonar ping to identify submarine", "agent decodes API error response to understand system state"], bridges=["acoustics","decode","signal","interpret"], tags=["sonar","echo","interpret"])
        ns.define("frequency-sweep", "Systematically vary probe frequency to map environment at different resolution scales", Level=Level.CONCRETE, examples=["sonar sweeps frequencies for fine and coarse mapping", "agent varies query specificity from broad to narrow"], bridges=["acoustics","frequency","sweep","resolution"], tags=["sonar","frequency","sweep"])
        ns.define("beam-forming-probe", "Direct probe energy in specific direction to investigate localized area with higher resolution", Level=Level.CONCRETE, examples=["phased array focuses sonar beam on suspected target", "agent directs specific query at suspected system component"], bridges=["acoustics","direction","focus","resolution"], tags=["sonar","beam","focus"])
        ns.define("doppler-shift-detect", "Detect movement and velocity of objects by measuring frequency shift in return signals", Level=Level.CONCRETE, examples=["radar detects aircraft velocity via Doppler shift", "agent detects system change rate by comparing response timestamps"], bridges=["physics","doppler","velocity","detect"], tags=["sonar","doppler","velocity"])
        ns.define("clutter-rejection", "Filter irrelevant return signals to focus on meaningful environmental features", Level=Level.PATTERN, examples=["sonar filters seabed reflections to track submarine", "agent filters noise from relevant API responses"], bridges=["signal-processing","filter","noise","focus"], tags=["sonar","clutter","filter"])
        ns.define("multipath-interpretation", "Understand that return signals may arrive via multiple paths and disentangle them", Level=Level.PATTERN, examples=["sonar distinguishes direct echo from seabed bounce", "agent traces error through multiple system layers"], bridges=["acoustics","multipath","disentangle","complex"], tags=["sonar","multipath","complex"])
    def _load_metamaterial_cognition(self):
        ns = self.add_namespace("metamaterial-cognition", "Cognitive architectures inspired by metamaterials that bend information and produce emergent properties")
        ns.define("information-lens", "Cognitive structure that focuses or disperses information streams like a physical lens bends light", Level=Level.DOMAIN, examples=["attention mechanism acts as information lens focusing relevant inputs", "magnifying glass concentrates light to single point"], bridges=["optics","attention","focus","information"], tags=["metamaterial","lens","focus"])
        ns.define("cognitive-negative-index", "Cognitive architecture that processes information in reverse order enabling unusual insight patterns", Level=Level.DOMAIN, examples=["reverse actualization works backward from future to present", "negative refraction bends light backward against normal expectation"], bridges=["optics","reverse","negative","unusual"], tags=["metamaterial","reverse","negative"])
        ns.define("invisibility-cloak-cognition", "Agent selectively makes its internal state invisible to other agents while remaining functional", Level=Level.PATTERN, examples=["zero-knowledge proof: prove capability without revealing method", "metamaterial cloak bends light around object making it invisible"], bridges=["optics","stealth","invisible","functional"], tags=["metamaterial","invisible","stealth"])
        ns.define("acoustic-metamaterial-filter", "Information filter that blocks specific frequency bands of data while passing others unchanged", Level=Level.PATTERN, examples=["agent blocks low-confidence signals while passing high-confidence ones", "acoustic metamaterial blocks specific sound frequencies"], bridges=["acoustics","filter","band","selective"], tags=["metamaterial","filter","acoustic"])
        ns.define("superlens-cognition", "Cognitive structure that resolves information below normal resolution limit through near-field processing", Level=Level.DOMAIN, examples=["agent detects patterns too subtle for normal processing via specialized near-field", "superlens resolves details smaller than wavelength of light"], bridges=["optics","resolution","near-field","super"], tags=["metamaterial","superlens","resolution"])
        ns.define("photonic-crystal-thought", "Structured cognitive pathway that only allows specific thought patterns to propagate blocking others", Level=Level.PATTERN, examples=["agent only processes thoughts matching structured deliberation pattern", "photonic crystal only transmits specific light wavelengths"], bridges=["optics","crystal","structure","filter"], tags=["metamaterial","crystal","thought"])
    def _load_thermal_management(self):
        ns = self.add_namespace("thermal-management", "Heat dissipation and thermal throttling patterns in computing systems")
        ns.define("thermal-throttle-cascade", "Progressive performance reduction as temperature rises through multiple throttle stages", Level=Level.CONCRETE, examples=["CPU reduces clock speed at 80C then 90C then shuts down at 105C", "agent reduces deliberation depth as energy budget depletes"], bridges=["thermal","throttle","cascade","progressive"], tags=["thermal","throttle","cascade"])
        ns.define("heat-sink-distribution", "Spread computation across multiple agents to prevent any single agent from overheating", Level=Level.PATTERN, examples=["load balancer distributes requests to prevent server overheating", "fleet distributes deliberation across agents to prevent energy exhaustion"], bridges=["thermal","distribute","load","spread"], tags=["thermal","sink","distribute"])
        ns.define("thermal-budget-awareness", "Agent aware of its thermal constraints and proactively managing workload to stay within limits", Level=Level.PATTERN, examples=["scheduler delays batch jobs during peak temperature", "agent schedules rest periods to prevent energy exhaustion"], bridges=["thermal","budget","aware","proactive"], tags=["thermal","budget","aware"])
        ns.define("phase-change-buffer", "Use phase change materials to absorb heat spikes providing temporary thermal buffer during burst workloads", Level=Level.CONCRETE, examples=["laptop uses wax heat pipe to absorb burst heat", "agent uses energy reserve to handle sudden workload spike"], bridges=["thermal","phase-change","buffer","burst"], tags=["thermal","phase","buffer"])
        ns.define("convective-workflow", "Design workflow so completed tasks carry heat away from active tasks preventing thermal accumulation", Level=Level.PATTERN, examples=["pipeline stages pass work downstream carrying context", "conveyor belt carries hot items away from furnace"], bridges=["thermal","convective","workflow","pipeline"], tags=["thermal","convective","workflow"])

    def _load_mathematics(self):
        ns = self.add_namespace("mathematics",
            "Mathematical structures and operations underlying agent cognition")

        ns.define("harmonic-mean",
            "Average that penalizes small values: n / (1/a + 1/b + ...)",
            level=Level.CONCRETE,
            examples=["speed: average of 60mph and 40mph via harmonic mean = 48mph (not 50mph)", "confidence: 0.9 and 0.1 fused = 0.09 (not 0.5)", "resistor parallel: 1/R = 1/R1 + 1/R2"],
            bridges=["harmonic-mean-fusion", "confidence", "fusion", "average"],
            tags=["mathematics", "fleet-foundation"])

        ns.define("exponential-decay",
            "Value decreases as e^(-λt), creating a smooth decline with configurable half-life",
            level=Level.PATTERN,
            examples=["radioactive half-life of 10 years", "trust decays with half-life of 1 week", "memory with half-life of 30 seconds (working) vs 1 year (procedural)", "formula: value(t) = initial * e^(-λt)"],
            properties={"formula": "e^(-lambda*t)", "half_life": "ln(2)/lambda", "ubiquitous": True},
            bridges=["decay", "forgetting-curve", "trust", "memory"],
            tags=["mathematics", "ubiquitous", "fleet"])

        ns.define("welford-algorithm",
            "Online algorithm for computing mean and variance without storing all data",
            level=Level.CONCRETE,
            examples=["streaming statistics: process 1M events with 3 variables, not 1M storage", "anomaly detection: is current behavior outside 2σ of running mean?", "agent behavior baseline tracking"],
            bridges=["mean", "variance", "anomaly-detection", "online-algorithm"],
            tags=["mathematics", "algorithm", "statistics"])

        ns.define("topological-sort",
            "Order elements so that every dependency appears before its dependent",
            level=Level.PATTERN,
            examples=["build system: compile dependencies before dependents", "course schedule: take prerequisites first", "workflow: complete prerequisite tasks before dependent tasks"],
            bridges=["dag", "workflow", "dependency", "ordering"],
            tags=["mathematics", "algorithm", "scheduling"])

        ns.define("hamming-distance",
            "Number of positions at which two strings (or vectors) differ",
            level=Level.CONCRETE,
            examples=["error detection: received 1010, expected 1110, hamming distance 1", "DNA sequence comparison", "agent state comparison: current vs expected behavior pattern"],
            bridges=["similarity", "distance", "error-detection", "pattern-matching"],
            tags=["mathematics", "metric", "algorithm"])
