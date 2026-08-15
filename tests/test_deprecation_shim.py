"""Tests for the pre-split usage-limits deprecation shim.

The shim keeps two surfaces alive for one release cycle: the ``UsageLimits`` class
with its ``request_limit`` spelling, and the ``ReactAgentConfig(usage_limits=...)``
constructor keyword. Either half alone preserves nothing — an old caller writes
``UsageLimits(request_limit=...)`` and fails at *that* line before ever reaching
``ReactAgentConfig``.

The load-bearing assertion throughout is that a deprecated value **arrives at its
destination**. A shim that accepts the old keyword and drops the value is worse than
no shim: the agent silently runs on a budget nobody chose, and nothing fails until
the bill does.
"""

import warnings

import pytest

from akgentic.llm import (
    AgentUsageLimits,
    CompactionConfig,
    ModelConfig,
    ReactAgentConfig,
    RuntimeConfig,
    RunUsageLimits,
    UsageLimits,
)
from akgentic.llm.config import TokenUsageLimits

# What every deprecation warning must communicate about the removal schedule.
#
# Deliberately NOT a version literal. The shim was announced for removal in 2.0.0;
# 2.0.0 shipped as the pydantic-ai v2 bump with the shim still in it, and the old
# assertions -- which pinned the string "2.0.0" -- stayed green while the message
# they guarded had become false. Asserting on the *schedule clause* instead of a
# version number is what makes these tests fail when the message stops being true.
REMOVAL_SCHEDULE = "no removal release is scheduled"


class TestDeprecatedUsageLimitsClass:
    """UsageLimits survives as a deprecated alias of the run tier."""

    def test_pre_split_kwargs_construct_and_map(self):
        """The old spelling still constructs and lands on the new field."""
        with pytest.warns(DeprecationWarning):
            limits = UsageLimits(request_limit=10, tool_calls_limit=5)
        assert limits.run_request_limit == 10
        assert limits.tool_calls_limit == 5

    def test_construction_emits_exactly_one_warning(self):
        """One deprecated construction, one warning — not zero, not a storm."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            UsageLimits(request_limit=10, tool_calls_limit=5)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1

    def test_construction_warning_names_replacement_and_schedule(self):
        """The warning must tell the caller what to write instead, and the schedule."""
        with pytest.warns(DeprecationWarning) as record:
            UsageLimits(request_limit=10)
        message = str(record[0].message)
        assert "RunUsageLimits" in message
        assert "run_request_limit" in message
        assert REMOVAL_SCHEDULE in message

    def test_construction_warning_points_at_the_caller(self):
        """stacklevel must blame the caller's line, not config.py.

        A deprecation warning whose traceback lands inside the library tells the
        user nothing about which of their own lines to fix.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            UsageLimits(request_limit=10)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations[0].filename == __file__

    def test_request_limit_read_returns_underlying_field(self):
        """Reading the old name returns the new field's value, and warns."""
        with pytest.warns(DeprecationWarning):
            limits = UsageLimits(request_limit=10)
        with pytest.warns(DeprecationWarning) as record:
            value = limits.request_limit
        assert value == 10
        message = str(record[0].message)
        assert "run_request_limit" in message
        assert REMOVAL_SCHEDULE in message

    def test_request_limit_read_reflects_new_spelling(self):
        """The read path follows the field, whichever spelling set it."""
        limits = UsageLimits(run_request_limit=7)
        with pytest.warns(DeprecationWarning):
            assert limits.request_limit == 7

    def test_request_limit_is_not_a_field(self):
        """A second storage slot would reintroduce the split-brain the rename removed."""
        assert "request_limit" not in UsageLimits.model_fields
        assert set(UsageLimits.model_fields) == set(RunUsageLimits.model_fields)

    def test_both_spellings_together_rejected(self):
        """Ambiguity is an error, not a silent winner decided by argument order."""
        with pytest.raises(ValueError):
            UsageLimits(request_limit=10, run_request_limit=20)

    def test_both_spellings_together_rejected_when_equal(self):
        """Equal values do not excuse the ambiguity."""
        with pytest.raises(ValueError):
            UsageLimits(request_limit=10, run_request_limit=10)

    def test_both_spellings_raise_before_the_deprecation_warning(self):
        """The ValueError must not be pre-empted by the deprecation warning.

        Downstream projects routinely run with ``-W error::DeprecationWarning`` to hunt
        deprecated usage. If the warning fires first, that turns into the raised
        exception and the caller never sees the message telling them what is actually
        wrong with their call.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with pytest.raises(ValueError):
                UsageLimits(request_limit=10, run_request_limit=20)

    def test_is_a_run_usage_limits(self):
        """Subclassing is what lets an old instance satisfy the new annotation."""
        with pytest.warns(DeprecationWarning):
            limits = UsageLimits()
        assert isinstance(limits, RunUsageLimits)
        assert isinstance(limits, TokenUsageLimits)

    def test_default_run_request_limit_preserved(self):
        """The deprecated alias keeps the run tier's 50-request safety brake."""
        with pytest.warns(DeprecationWarning):
            assert UsageLimits().run_request_limit == 50

    def test_validate_from_instance_skips_the_shim_entirely(self):
        """Revalidating an existing instance neither re-warns nor re-maps.

        Pydantic short-circuits on an instance of the class, so the before-validator
        never runs — which is why the shim cannot corrupt an already-built object.
        """
        with pytest.warns(DeprecationWarning):
            original = UsageLimits(request_limit=10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored = UsageLimits.model_validate(original)
        assert restored is original
        assert restored.run_request_limit == 10
        assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []

    def test_non_mapping_input_rejected_cleanly(self):
        """Non-dict input passes through the shim to Pydantic's own error.

        The guard exists so a bad input raises ValidationError rather than an
        AttributeError from inside the shim.
        """
        with pytest.raises(ValueError):
            UsageLimits.model_validate([("request_limit", 10)])


class TestSerializationKeysAreNotPreserved:
    """The shim covers the constructor keyword and the attribute read — not the wire.

    Pinned deliberately rather than left to chance: a caller that round-trips through
    ``model_dump()`` and keys off the pre-split names is NOT covered, and should find
    that stated here rather than discover it in production.
    """

    def test_dump_emits_the_new_field_name(self):
        """UsageLimits serializes as run_request_limit, not request_limit."""
        with pytest.warns(DeprecationWarning):
            limits = UsageLimits(request_limit=10)
        data = limits.model_dump()
        assert data["run_request_limit"] == 10
        assert "request_limit" not in data

    def test_config_dump_emits_the_new_field_name(self):
        """ReactAgentConfig serializes as run_usage_limits, not usage_limits."""
        with pytest.warns(DeprecationWarning):
            config = ReactAgentConfig(usage_limits=RunUsageLimits(run_request_limit=10))
        data = config.model_dump()
        assert data["run_usage_limits"]["run_request_limit"] == 10
        assert "usage_limits" not in data


class TestDeprecatedUsageLimitsField:
    """ReactAgentConfig(usage_limits=...) survives as a deprecated keyword."""

    def test_keyword_accepted_and_value_reaches_run_usage_limits(self):
        """The destination is the assertion — acceptance alone proves nothing."""
        with pytest.warns(DeprecationWarning):
            config = ReactAgentConfig(usage_limits=RunUsageLimits(run_request_limit=10))
        assert config.run_usage_limits.run_request_limit == 10

    def test_keyword_warning_names_replacement_and_schedule(self):
        """The warning must name the new field and the removal schedule."""
        with pytest.warns(DeprecationWarning) as record:
            ReactAgentConfig(usage_limits=RunUsageLimits(run_request_limit=10))
        message = str(record[0].message)
        assert "run_usage_limits" in message
        assert REMOVAL_SCHEDULE in message

    def test_keyword_warning_points_at_the_caller(self):
        """stacklevel must blame the caller's line, not config.py."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ReactAgentConfig(usage_limits=RunUsageLimits(run_request_limit=10))
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations[0].filename == __file__

    def test_read_accessor_returns_the_run_tier(self):
        """config.usage_limits is a view over the one real field, not a copy."""
        config = ReactAgentConfig(run_usage_limits=RunUsageLimits(run_request_limit=10))
        with pytest.warns(DeprecationWarning) as record:
            value = config.usage_limits
        assert value is config.run_usage_limits
        message = str(record[0].message)
        assert "run_usage_limits" in message
        assert REMOVAL_SCHEDULE in message

    def test_usage_limits_is_not_a_field(self):
        """The read accessor must be a computed view, never a second storage slot."""
        assert "usage_limits" not in ReactAgentConfig.model_fields

    def test_both_names_rejected_old_first(self):
        """Both keywords together raise, regardless of which is written first."""
        with pytest.raises(ValueError):
            ReactAgentConfig(
                usage_limits=RunUsageLimits(run_request_limit=10),
                run_usage_limits=RunUsageLimits(run_request_limit=20),
            )

    def test_both_names_rejected_new_first(self):
        """The mirror-image keyword order raises identically."""
        with pytest.raises(ValueError):
            ReactAgentConfig(
                run_usage_limits=RunUsageLimits(run_request_limit=20),
                usage_limits=RunUsageLimits(run_request_limit=10),
            )

    def test_both_names_rejected_when_values_equal(self):
        """Equal values do not resolve the ambiguity in either name's favour."""
        limits = RunUsageLimits(run_request_limit=10)
        with pytest.raises(ValueError):
            ReactAgentConfig(usage_limits=limits, run_usage_limits=limits)

    def test_deprecated_instance_stored_as_is(self):
        """Pydantic accepts a subclass instance without re-validating it down."""
        with pytest.warns(DeprecationWarning):
            limits = UsageLimits(request_limit=10)
        with pytest.warns(DeprecationWarning):
            config = ReactAgentConfig(usage_limits=limits)
        assert config.run_usage_limits is limits
        assert isinstance(config.run_usage_limits, UsageLimits)

    def test_mapping_value_maps_the_inner_pre_split_spelling(self):
        """A dict under the old keyword keeps its budget too.

        The dict is validated as RunUsageLimits, where ``request_limit`` is an unknown
        key that Pydantic drops in silence. Warning about the outer keyword while losing
        the inner value is the accepted-and-discarded failure with one more layer of
        indirection — the caller is told the shim handled their input, and it did not.
        """
        with pytest.warns(DeprecationWarning):
            config = ReactAgentConfig(usage_limits={"request_limit": 10})
        assert config.run_usage_limits.run_request_limit == 10

    def test_mapping_value_with_both_spellings_rejected(self):
        """Ambiguity one level down is still ambiguity."""
        with pytest.raises(ValueError):
            ReactAgentConfig(usage_limits={"request_limit": 10, "run_request_limit": 20})

    def test_both_names_raise_before_the_deprecation_warning(self):
        """As on the class shim: the error must survive -W error::DeprecationWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with pytest.raises(ValueError):
                ReactAgentConfig(
                    usage_limits=RunUsageLimits(run_request_limit=10),
                    run_usage_limits=RunUsageLimits(run_request_limit=20),
                )

    def test_assignment_through_the_deprecated_name_is_not_shimmed(self):
        """The accessor is read-only, and says so by failing rather than by shadowing.

        Only the constructor keyword and the attribute read are preserved. What matters
        is that a write fails loudly instead of parking the value on a shadow attribute
        that ``run_usage_limits`` would never see.
        """
        config = ReactAgentConfig(run_usage_limits=RunUsageLimits(run_request_limit=10))
        with pytest.raises((AttributeError, ValueError)):
            config.usage_limits = RunUsageLimits(run_request_limit=99)
        assert config.run_usage_limits.run_request_limit == 10


class TestRealConsumerCallShape:
    """The shape an actual downstream caller uses today, reproduced verbatim.

    A sibling package builds its ReactAgentConfig with exactly these four keywords.
    Reproducing the shape here (rather than importing that package, which this one
    may not depend on) is what pins the shim to a real migration path instead of a
    hypothetical one.
    """

    def test_four_keyword_construction_warns_and_lands(self):
        """The consumer's call still works, warns once, and keeps its budget."""
        limits = RunUsageLimits(run_request_limit=7, total_tokens_limit=1234)
        with pytest.warns(DeprecationWarning):
            config = ReactAgentConfig(
                model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
                runtime_cfg=RuntimeConfig(),
                usage_limits=limits,
                compaction_cfg=CompactionConfig(),
            )
        assert config.run_usage_limits.run_request_limit == 7
        assert config.run_usage_limits.total_tokens_limit == 1234
        assert config.model_cfg.model == "gpt-4o"

    def test_four_keyword_construction_with_the_deprecated_class(self):
        """The fully pre-split call — old class and old keyword together."""
        with pytest.warns(DeprecationWarning):
            limits = UsageLimits(request_limit=7, total_tokens_limit=1234)
        with pytest.warns(DeprecationWarning):
            config = ReactAgentConfig(
                model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
                runtime_cfg=RuntimeConfig(),
                usage_limits=limits,
                compaction_cfg=CompactionConfig(),
            )
        assert config.run_usage_limits.run_request_limit == 7
        assert config.run_usage_limits.total_tokens_limit == 1234


class TestPreSplitCallerUnedited:
    """Assertions lifted from the pre-split suite, bodies unchanged.

    The only concession is the pytest.warns wrapper. If any assertion below had to
    be reworded to pass, the surface was not actually preserved.
    """

    def test_specific_limits(self):
        """Test setting specific limits."""
        with pytest.warns(DeprecationWarning):
            limits = UsageLimits(request_limit=10, total_tokens_limit=5000)
            assert limits.request_limit == 10
            assert limits.total_tokens_limit == 5000

    def test_all_limits_set(self):
        """Test all limits can be set."""
        with pytest.warns(DeprecationWarning):
            limits = UsageLimits(
                request_limit=100,
                tool_calls_limit=50,
                input_tokens_limit=2000,
                output_tokens_limit=1000,
                total_tokens_limit=3000,
            )
            assert limits.request_limit == 100
            assert limits.tool_calls_limit == 50
            assert limits.input_tokens_limit == 2000
            assert limits.output_tokens_limit == 1000
            assert limits.total_tokens_limit == 3000

    def test_default_request_limit(self):
        """Test default request_limit is 50."""
        with pytest.warns(DeprecationWarning):
            limits = UsageLimits()
            assert limits.request_limit == 50

    def test_invalid_negative_limit(self):
        """Test negative limits raise error."""
        with pytest.raises(ValueError):
            UsageLimits(request_limit=-1)

    def test_full_config(self):
        """Test complete agent configuration."""
        with pytest.warns(DeprecationWarning):
            config = ReactAgentConfig(
                model_cfg=ModelConfig(provider="openai", model="gpt-4o", temperature=0.7),
                usage_limits=UsageLimits(request_limit=10, total_tokens_limit=5000),
                runtime_cfg=RuntimeConfig(retries=5),
            )
            assert config.usage_limits.request_limit == 10
            assert config.usage_limits.total_tokens_limit == 5000


class TestNewSpellingsAreWarningFree:
    """The migration target must be silent, or the warnings are noise."""

    def _assert_no_deprecation(self, caught):
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations == [], f"unexpected DeprecationWarning: {deprecations}"

    def test_run_usage_limits_silent(self):
        """The run tier constructs without complaint."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RunUsageLimits(run_request_limit=10)
        self._assert_no_deprecation(caught)

    def test_agent_usage_limits_silent(self):
        """The agent tier constructs without complaint."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AgentUsageLimits(agent_request_limit=3)
        self._assert_no_deprecation(caught)

    def test_token_usage_limits_silent(self):
        """The shared base constructs without complaint."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TokenUsageLimits()
        self._assert_no_deprecation(caught)

    def test_react_agent_config_new_fields_silent(self):
        """Both new keywords together produce no deprecation noise."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ReactAgentConfig(
                run_usage_limits=RunUsageLimits(run_request_limit=10),
                agent_usage_limits=AgentUsageLimits(agent_request_limit=3),
            )
        self._assert_no_deprecation(caught)

    def test_bare_react_agent_config_silent(self):
        """Defaults alone must not warn — that would fire on every construction."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ReactAgentConfig()
        self._assert_no_deprecation(caught)

    def test_reading_new_field_silent(self):
        """Only the deprecated accessor warns; the real field is silent."""
        config = ReactAgentConfig(run_usage_limits=RunUsageLimits(run_request_limit=10))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert config.run_usage_limits.run_request_limit == 10
        self._assert_no_deprecation(caught)


class TestWarningsAreObservable:
    """Vacuity guard: the shim tests must not pass on a silenced warning.

    Deliberately does NOT call simplefilter("always"). It asserts the warning is
    visible under the suite's *ambient* filter state, so adding a global
    ``ignore::DeprecationWarning`` to the pytest configuration turns this test red
    instead of quietly hollowing out every warning assertion above.
    """

    def test_construction_warning_observable_under_default_filters(self):
        """A deprecated construction is visible without forcing the filter."""
        with warnings.catch_warnings(record=True) as caught:
            UsageLimits(request_limit=10)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_read_warning_observable_under_default_filters(self):
        """A deprecated read is visible without forcing the filter."""
        config = ReactAgentConfig(run_usage_limits=RunUsageLimits(run_request_limit=10))
        with warnings.catch_warnings(record=True) as caught:
            _ = config.usage_limits
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
