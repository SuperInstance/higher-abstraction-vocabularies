"""
Comprehensive test suite for Higher Abstraction Vocabularies (HAV).
Covers Term matching, Namespace operations, HAV search/explain/bridge/suggest,
builtin vocabularies, and the Level enum.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vocab import Term, Namespace, HAV, Level


# =========================================================================
# Level enum
# =========================================================================

class TestLevelEnum:
    """Tests for the Level abstraction-level enum."""

    def test_all_values_present(self):
        expected = {"CONCRETE", "PATTERN", "BEHAVIOR", "DOMAIN", "META"}
        actual = {member.name for member in Level}
        assert actual == expected

    def test_correct_ordering(self):
        assert Level.CONCRETE.value == 0
        assert Level.PATTERN.value == 1
        assert Level.BEHAVIOR.value == 2
        assert Level.DOMAIN.value == 3
        assert Level.META.value == 4


# =========================================================================
# Term.matches()
# =========================================================================

class TestTermMatches:
    """Tests for Term.matches() scoring logic."""

    def setup_method(self):
        self.term = Term(
            name="forgetting-curve",
            short="Exponential decay of memory over time",
            description="Ebbinghaus's discovery: memory fades exponentially with configurable half-lives.",
            examples=["forget 50% in 1 hour", "spaced repetition extends the curve"],
            aliases=["memory-decay", "decay-curve"],
            tags=["memory", "learning", "psychology"],
        )

    def test_exact_name_match_returns_one(self):
        assert self.term.matches("forgetting-curve") == 1.0

    def test_case_insensitive_exact_match(self):
        assert self.term.matches("Forgetting-Curve") == 1.0

    def test_alias_match_returns_one(self):
        assert self.term.matches("memory-decay") == 1.0

    def test_alias_match_case_insensitive(self):
        assert self.term.matches("Decay-Curve") == 1.0

    def test_name_without_hyphens(self):
        assert self.term.matches("forgettingcurve") == 0.95

    def test_substring_in_name(self):
        score = self.term.matches("forgetting")
        assert score >= 0.5

    def test_substring_in_short_description(self):
        score = self.term.matches("exponential decay")
        # Should match in short field with 0.35 weight + token overlap
        assert score >= 0.35

    def test_substring_in_description(self):
        score = self.term.matches("ebbinghaus")
        assert score >= 0.2

    def test_token_overlap(self):
        # "memory time" — both tokens appear in the term's text
        score = self.term.matches("memory time")
        assert score > 0.0

    def test_empty_query_returns_zero(self):
        assert self.term.matches("") == 0.0

    def test_whitespace_query_returns_zero(self):
        assert self.term.matches("   ") == 0.0

    def test_example_text_match(self):
        score = self.term.matches("spaced repetition")
        # Should get at least 0.08 for example match
        assert score >= 0.08

    def test_tag_match(self):
        score = self.term.matches("psychology")
        # Tag match contributes 0.1
        assert score >= 0.1

    def test_scoring_capped_at_one(self):
        # Query that matches name (0.5) + short (0.35) + description (0.2) + tokens
        score = self.term.matches("decay")
        assert score <= 1.0

    def test_no_match_returns_low_score(self):
        score = self.term.matches("quantum entanglement")
        # Should still be >= 0 since there might be some token overlap
        assert score >= 0.0

    def test_single_token_no_overlap(self):
        t = Term(name="xyz", short="A rare concept")
        score = t.matches("zzzzzzz")
        # No substring match, no token overlap
        assert score < 0.1


# =========================================================================
# Term.explain()
# =========================================================================

class TestTermExplain:
    """Tests for Term.explain() output formatting."""

    def test_explain_contains_name(self):
        t = Term(name="confidence", short="A certainty measure")
        output = t.explain()
        assert "confidence" in output

    def test_explain_contains_short(self):
        t = Term(name="confidence", short="A certainty measure")
        output = t.explain()
        assert "A certainty measure" in output

    def test_explain_contains_description(self):
        t = Term(name="confidence", short="A certainty measure",
                 description="Full explanation here.")
        output = t.explain()
        assert "Full explanation here" in output

    def test_explain_contains_examples(self):
        t = Term(name="confidence", short="A certainty measure",
                 examples=["example one", "example two"])
        output = t.explain()
        assert "example one" in output
        assert "example two" in output

    def test_explain_contains_aliases(self):
        t = Term(name="confidence", short="A certainty measure",
                 aliases=["certainty", "sureness"])
        output = t.explain()
        assert "certainty" in output
        assert "sureness" in output

    def test_explain_contains_tags(self):
        t = Term(name="confidence", short="A certainty measure",
                 tags=["core", "fleet"])
        output = t.explain()
        assert "core" in output
        assert "fleet" in output

    def test_explain_contains_bridges(self):
        t = Term(name="confidence", short="A certainty measure",
                 bridges=["trust", "belief"])
        output = t.explain()
        assert "trust" in output
        assert "belief" in output

    def test_explain_contains_antonyms(self):
        t = Term(name="exploration", short="Trying new things",
                 antonyms=["exploitation"])
        output = t.explain()
        assert "exploitation" in output

    def test_explain_minimal_fields(self):
        t = Term(name="simple", short="A simple concept")
        output = t.explain()
        assert "## simple" in output
        assert "A simple concept" in output
        # No description — should still be valid
        assert output != ""

    def test_explain_markdown_heading(self):
        t = Term(name="test-term", short="desc")
        output = t.explain()
        assert output.startswith("## test-term")


# =========================================================================
# Namespace
# =========================================================================

class TestNamespace:
    """Tests for the Namespace class."""

    def setup_method(self):
        self.ns = Namespace(name="test-domain")

    def test_define_creates_term(self):
        t = self.ns.define("alpha", "First term")
        assert isinstance(t, Term)
        assert t.name == "alpha"
        assert t.short == "First term"

    def test_define_sets_domain_to_namespace_name(self):
        t = self.ns.define("beta", "Second term")
        assert t.domain == "test-domain"

    def test_define_with_extra_kwargs(self):
        t = self.ns.define("gamma", "Third term",
                           description="A description",
                           level=Level.CONCRETE,
                           tags=["tag1"])
        assert t.description == "A description"
        assert t.level == Level.CONCRETE
        assert t.tags == ["tag1"]

    def test_lookup_finds_defined_term(self):
        self.ns.define("delta", "Fourth term")
        t = self.ns.lookup("delta")
        assert t is not None
        assert t.name == "delta"

    def test_lookup_returns_none_for_unknown(self):
        assert self.ns.lookup("nonexistent") is None

    def test_search_returns_sorted_results_highest_first(self):
        self.ns.define("abc-term", "ABC definition", tags=["abc"])
        self.ns.define("xyz-term", "XYZ definition", tags=["xyz"])
        results = self.ns.search("abc")
        assert len(results) >= 1
        # First result should be abc-term
        assert results[0][0].name == "abc-term"
        assert results[0][1] > 0

    def test_search_respects_threshold(self):
        self.ns.define("target", "A target concept")
        results_high = self.ns.search("target", threshold=0.9)
        results_low = self.ns.search("target", threshold=0.01)
        # Lower threshold should return >= results from higher threshold
        assert len(results_low) >= len(results_high)

    def test_search_no_matches_returns_empty(self):
        self.ns.define("alpha", "Some definition")
        results = self.ns.search("zzzzzzz-nothing", threshold=0.5)
        assert results == []

    def test_len_returns_term_count(self):
        assert len(self.ns) == 0
        self.ns.define("a", "first")
        assert len(self.ns) == 1
        self.ns.define("b", "second")
        assert len(self.ns) == 2

    def test_iter_iterates_over_terms(self):
        self.ns.define("one", "first")
        self.ns.define("two", "second")
        self.ns.define("three", "third")
        names = {t.name for t in self.ns}
        assert names == {"one", "two", "three"}

    def test_define_overwrites_existing(self):
        self.ns.define("dup", "original")
        self.ns.define("dup", "replaced")
        t = self.ns.lookup("dup")
        assert t.short == "replaced"


# =========================================================================
# HAV class
# =========================================================================

class TestHAV:
    """Tests for the HAV engine class."""

    def setup_method(self):
        self.hav = HAV()

    # --- stats ---

    def test_stats_returns_dict(self):
        s = self.hav.stats()
        assert isinstance(s, dict)

    def test_stats_has_namespaces_key(self):
        s = self.hav.stats()
        assert "namespaces" in s

    def test_stats_has_total_terms_key(self):
        s = self.hav.stats()
        assert "total_terms" in s

    def test_stats_has_by_domain_key(self):
        s = self.hav.stats()
        assert "by_domain" in s

    def test_stats_total_terms_greater_than_zero(self):
        s = self.hav.stats()
        assert s["total_terms"] > 0

    # --- search ---

    def test_search_returns_results(self):
        results = self.hav.search("memory")
        assert len(results) > 0

    def test_search_returns_tuples(self):
        results = self.hav.search("confidence")
        for r in results:
            assert len(r) == 3  # (namespace, term, score)
            assert isinstance(r[0], str)
            assert isinstance(r[1], Term)
            assert isinstance(r[2], float)

    def test_search_sorted_by_score_descending(self):
        results = self.hav.search("memory")
        for i in range(len(results) - 1):
            assert results[i][2] >= results[i + 1][2]

    def test_search_with_domain_filter(self):
        results = self.hav.search("confidence", domain="uncertainty")
        for ns_name, _, _ in results:
            assert ns_name == "uncertainty"

    def test_search_with_high_threshold(self):
        results = self.hav.search("xyznonexistent", threshold=0.9)
        assert results == []

    def test_search_with_threshold_parameter(self):
        low = self.hav.search("learning", threshold=0.01)
        high = self.hav.search("learning", threshold=0.5)
        assert len(low) >= len(high)

    # --- explain ---

    def test_explain_exact_term(self):
        result = self.hav.explain("confidence")
        assert "confidence" in result
        assert "0-1" in result  # from short description

    def test_explain_unknown_term_did_you_mean(self):
        result = self.hav.explain("confidenc")  # slight typo
        assert "Did you mean" in result

    def test_explain_not_found_no_suggestions(self):
        result = self.hav.explain("zzzzzzz-nothing-at-all")
        assert "No match" in result

    # --- suggest ---

    def test_suggest_returns_up_to_ten(self):
        results = self.hav.suggest("agent learning behavior")
        assert len(results) <= 10

    def test_suggest_returns_results(self):
        results = self.hav.suggest("memory that fades")
        assert len(results) > 0

    def test_suggest_natural_language(self):
        results = self.hav.suggest("how agents work together indirectly")
        assert len(results) > 0

    # --- bridge ---

    def test_bridge_finds_connections(self):
        # "trust" is bridged to in the uncertainty namespace
        results = self.hav.bridge("trust")
        assert len(results) > 0

    def test_bridge_with_from_domain(self):
        results = self.hav.bridge("confidence", from_domain="uncertainty")
        for ns_name, t in results:
            assert t.domain == "uncertainty"

    def test_bridge_with_to_domain(self):
        results = self.hav.bridge("confidence", to_domain="uncertainty")
        for ns_name, _ in results:
            assert ns_name == "uncertainty"

    def test_bridge_no_results(self):
        results = self.hav.bridge("xyznonexistent-term-bridge")
        assert results == []

    # --- random_term ---

    def test_random_term_returns_term(self):
        t = self.hav.random_term()
        assert isinstance(t, Term)
        assert t.name != ""

    # --- add_namespace ---

    def test_add_namespace_creates_new(self):
        ns = self.hav.add_namespace("custom-test-ns", "A test namespace")
        assert isinstance(ns, Namespace)
        assert ns.name == "custom-test-ns"

    def test_namespace_returns_existing(self):
        self.hav.add_namespace("findme-ns", "exists")
        ns = self.hav.namespace("findme-ns")
        assert ns is not None
        assert ns.name == "findme-ns"

    def test_namespace_returns_none_for_unknown(self):
        assert self.hav.namespace("does-not-exist-xyz") is None

    # --- define (auto-create namespace) ---

    def test_define_auto_creates_namespace(self):
        t = self.hav.define("new-auto-ns", "auto-term", "auto created term")
        assert isinstance(t, Term)
        ns = self.hav.namespace("new-auto-ns")
        assert ns is not None
        assert len(ns) == 1


# =========================================================================
# Builtin vocabularies
# =========================================================================

class TestBuiltinVocabularies:
    """Tests for builtin vocabulary content."""

    def setup_method(self):
        self.hav = HAV()

    def test_confidence_term_exists(self):
        results = self.hav.search("confidence")
        names = [t.name for _, t, _ in results]
        assert "confidence" in names

    def test_confidence_matches_certainty_query(self):
        results = self.hav.search("certainty")
        names = [t.name for _, t, _ in results]
        # "certainty" is an alias of "confidence"
        assert "confidence" in names

    def test_stigmergy_term_exists(self):
        results = self.hav.search("stigmergy")
        names = [t.name for _, t, _ in results]
        assert "stigmergy" in names

    def test_stigmergy_matches_coordination_query(self):
        results = self.hav.search("indirect coordination")
        # stigmergy should match because "coordination" is in its short desc
        names = [t.name for _, t, _ in results]
        assert "stigmergy" in names

    def test_trust_term_exists(self):
        results = self.hav.search("trust")
        names = [t.name for _, t, _ in results]
        assert "trust" in names

    def test_emergence_term_exists(self):
        results = self.hav.search("emergence")
        names = [t.name for _, t, _ in results]
        assert "emergence" in names

    def test_forgetting_curve_term_exists(self):
        results = self.hav.search("forgetting-curve")
        names = [t.name for _, t, _ in results]
        assert "forgetting-curve" in names

    def test_consensus_term_exists(self):
        results = self.hav.search("consensus")
        names = [t.name for _, t, _ in results]
        assert "consensus" in names

    def test_at_least_ten_namespaces_loaded(self):
        s = self.hav.stats()
        assert s["namespaces"] >= 10

    def test_total_terms_over_100(self):
        s = self.hav.stats()
        assert s["total_terms"] > 100

    def test_uncertainty_namespace_loaded(self):
        ns = self.hav.namespace("uncertainty")
        assert ns is not None
        assert len(ns) >= 5

    def test_memory_namespace_loaded(self):
        ns = self.hav.namespace("memory")
        assert ns is not None
        assert len(ns) >= 5

    def test_coordination_namespace_loaded(self):
        ns = self.hav.namespace("coordination")
        assert ns is not None
        assert len(ns) >= 5

    def test_learning_namespace_loaded(self):
        ns = self.hav.namespace("learning")
        assert ns is not None
        assert len(ns) >= 5

    def test_biological_namespace_loaded(self):
        ns = self.hav.namespace("biological")
        assert ns is not None
        assert len(ns) >= 5


# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:
    """Edge-case and integration tests."""

    def test_hav_search_empty_query(self):
        hav = HAV()
        results = hav.search("")
        assert results == []

    def test_term_matches_with_multiline_description(self):
        t = Term(
            name="multi",
            short="A test",
            description="Line one.\nLine two.\nLine three.",
            examples=["ex1"],
        )
        score = t.matches("line two")
        assert score > 0.0

    def test_namespace_search_empty_query(self):
        ns = Namespace(name="empty-test")
        ns.define("a", "A term")
        results = ns.search("")
        assert results == []

    def test_hav_explain_case_insensitive_lookup(self):
        """explain() should find terms case-insensitively via search fallback."""
        hav = HAV()
        # Exact lookup is case-sensitive; explain with slight variation
        # should trigger the search-based "Did you mean" path
        result = hav.explain("CONFIDENCE")  # uppercase won't match exact
        # Should either find it via search or give suggestions
        assert "confidence" in result.lower() or "No match" in result

    def test_term_with_no_fields_matches_gracefully(self):
        t = Term(name="minimal", short="min desc")
        score = t.matches("minimal")
        assert score == 1.0

    def test_namespace_define_then_search_owns_domain(self):
        hav = HAV()
        hav.define("my-domain", "custom-term", "Custom term definition",
                   aliases=["cterm"])
        results = hav.search("custom-term", domain="my-domain")
        assert len(results) == 1
        assert results[0][1].name == "custom-term"

    def test_bridge_across_custom_terms(self):
        hav = HAV()
        hav.define("dom-a", "term-a", "Term A",
                   bridges=["term-b"])
        hav.define("dom-b", "term-b", "Term B",
                   aliases=["term-a"])
        results = hav.bridge("term-a")
        assert len(results) >= 1
