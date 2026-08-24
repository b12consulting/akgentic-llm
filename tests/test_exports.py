"""Test public API exports."""

import pytest

import akgentic.llm


def test_all_exports_importable():
    """All __all__ members must be importable."""
    for name in akgentic.llm.__all__:
        assert hasattr(akgentic.llm, name), f"{name} missing from akgentic.llm"


def test_no_unexpected_exports():
    """Star import should only include __all__ members."""
    # Submodules are always visible via dir() but not in star imports.
    # `loadtest` is an optional-extra subpackage, deliberately absent from __all__.
    submodules = {
        "agent",
        "capabilities",
        "compaction",
        "config",
        "context",
        "event",
        "loadtest",
        "pricing",
        "prompts",
        "providers",
    }
    exported = {name for name in dir(akgentic.llm) if not name.startswith("_")}
    exported_without_submodules = exported - submodules
    expected = set(akgentic.llm.__all__)
    # __version__ might not always be in dir() depending on how it's defined
    assert exported_without_submodules == expected, (
        f"Unexpected exports: {exported_without_submodules - expected}"
    )


def test_key_exports_present():
    """Verify all key classes and functions are exported."""
    # Configuration
    assert hasattr(akgentic.llm, "ModelConfig")
    assert hasattr(akgentic.llm, "UsageLimits")
    assert hasattr(akgentic.llm, "RunUsageLimits")
    assert hasattr(akgentic.llm, "AgentUsageLimits")
    assert hasattr(akgentic.llm, "RuntimeConfig")
    assert hasattr(akgentic.llm, "ReactAgentConfig")

    # Agent
    assert hasattr(akgentic.llm, "ReactAgent")
    assert hasattr(akgentic.llm, "UsageLimitError")
    assert hasattr(akgentic.llm, "RunUsageLimitError")
    assert hasattr(akgentic.llm, "AgentUsageLimitError")

    # Context
    assert hasattr(akgentic.llm, "ContextManager")
    assert hasattr(akgentic.llm, "ContextObserver")
    # Removed in Epic 12 pre-cleanup — must not reappear.
    assert not hasattr(akgentic.llm, "ContextSnapshot")

    # Events
    assert hasattr(akgentic.llm, "LlmMessageEvent")
    # Checkpoint events removed in Epic 12 pre-cleanup — must not reappear.
    assert not hasattr(akgentic.llm, "LlmCheckpointCreatedEvent")
    assert not hasattr(akgentic.llm, "LlmCheckpointRestoredEvent")
    assert hasattr(akgentic.llm, "ToolCallEvent")
    assert hasattr(akgentic.llm, "ToolReturnEvent")

    # Run-loop capabilities
    assert hasattr(akgentic.llm, "LifetimeBudgetCapability")
    assert hasattr(akgentic.llm, "CompactionCapability")
    assert hasattr(akgentic.llm, "EventSourcingCapability")
    assert hasattr(akgentic.llm, "HealingCapability")
    assert hasattr(akgentic.llm, "LimitRecoveryCapability")
    assert hasattr(akgentic.llm, "ConclusionDecision")

    # Prompts
    assert hasattr(akgentic.llm, "PromptTemplate")
    assert hasattr(akgentic.llm, "current_datetime_prompt")
    assert hasattr(akgentic.llm, "json_output_reminder_prompt")

    # Pricing & Aggregation
    assert hasattr(akgentic.llm, "AgentUsageSummary")
    assert hasattr(akgentic.llm, "ModelUsage")
    assert hasattr(akgentic.llm, "RunUsageSummary")
    assert hasattr(akgentic.llm, "aggregate_usage")

    # Providers
    assert hasattr(akgentic.llm, "create_model")
    assert hasattr(akgentic.llm, "create_http_client")
    assert hasattr(akgentic.llm, "create_model_settings")
    assert hasattr(akgentic.llm, "get_output_type")


def test_star_import_works():
    """Test that `from akgentic.llm import *` works correctly."""
    # This simulates star import behavior
    namespace = {}
    exec("from akgentic.llm import *", namespace)

    # Check that all __all__ members are in namespace
    for name in akgentic.llm.__all__:
        assert name in namespace, f"{name} not imported by star import"

    # Check that private names are not imported
    assert "_build_settings" not in namespace
    assert "_create_openai_model" not in namespace


def test_version_exists():
    """Package version should be accessible."""
    assert hasattr(akgentic.llm, "__version__")
    assert isinstance(akgentic.llm.__version__, str)
    assert len(akgentic.llm.__version__) > 0


def test_module_docstring_exists():
    """Module should have a comprehensive docstring."""
    assert akgentic.llm.__doc__ is not None
    assert "Quick Start" in akgentic.llm.__doc__
    assert "Key Concepts" in akgentic.llm.__doc__
    assert "REACT pattern" in akgentic.llm.__doc__


def test_systempromptregistry_not_exported():
    """SystemPromptRegistry should NOT be exported (removed in architecture refactor)."""
    assert "SystemPromptRegistry" not in akgentic.llm.__all__
    assert not hasattr(akgentic.llm, "SystemPromptRegistry")


def test_the_usage_limit_hierarchy_is_importable_from_both_modules():
    """The three exception classes stay importable from ``akgentic.llm.agent``.

    They are defined in ``capabilities/errors.py`` — ``LifetimeBudgetCapability`` raises the
    agent tier and ``agent.py`` imports that package, not the reverse — and ``agent.py``
    re-exports them with the ``X as X`` pattern for callers written against their old home.

    Ruff treats ``X as X`` as an explicit re-export, so dropping one of those lines is silent
    to the toolchain. It is no longer silent to this suite: ``akgentic/llm/__init__.py``
    imports the same three names from ``.agent``, so the package stops importing entirely and
    collection fails. Do not lean on that second-order effect — it is a property of what
    ``__init__.py`` happens to re-export today, and it catches a *deleted* name, never a
    *shadowed* one.

    Asserted on IDENTITY, not on ``hasattr``: two separately-defined classes with the same
    name would satisfy a presence check while breaking every ``except`` written against the
    other module.
    """
    import akgentic.llm.agent as agent_module
    import akgentic.llm.capabilities as capabilities_module

    for name in ("UsageLimitError", "RunUsageLimitError", "AgentUsageLimitError"):
        one_class = getattr(capabilities_module, name)
        assert getattr(agent_module, name) is one_class, f"{name} is not the same class"
        assert getattr(akgentic.llm, name) is one_class, f"{name} is not the same class"


def test_every_public_capability_name_still_resolves_from_the_package():
    """The eleven public names stay importable from ``akgentic.llm.capabilities``.

    ``capabilities`` is a package of one module per capability, so each name now reaches
    callers through a re-export in ``capabilities/__init__.py``. Dropping one of those lines
    breaks every ``from akgentic.llm.capabilities import X`` written outside this repo and
    nothing in the toolchain notices: ruff counts ``__all__`` membership as a use, mypy is
    happy, and the package still imports.

    The names are a **hardcoded literal tuple**, never ``capabilities.__all__``: a test that
    iterates ``__all__`` passes green when a name is dropped from the import and from
    ``__all__`` in the same edit — which is exactly the edit that breaks callers.

    Each name's other homes are named EXPLICITLY, in their own sets, rather than sliced out
    of the tuple. A slice makes the tuple's order load-bearing in a way nothing states, so
    reordering two rows silently moves a name from one home's list to another's and the test
    still passes.

    Asserted on IDENTITY, not on ``hasattr``: two separately-defined classes of the same name
    satisfy a presence check while breaking every ``except`` and every ``isinstance`` written
    against the other module. The failure the identity check exists for is the **shadowed**
    name — a second class of the same name, so every import still resolves and only identity
    changes. A *deleted* re-export is loud here for an unrelated reason (``akgentic/llm/
    __init__.py`` imports several of these, so collection fails), and that is a property of
    what ``__init__.py`` happens to re-export today, not something to lean on.
    """
    import akgentic.llm.agent as agent_module
    import akgentic.llm.capabilities as capabilities_package
    from akgentic.llm.capabilities import (
        budget,
        compaction,
        errors,
        event_sourcing,
        healing,
        limit_recovery,
    )

    one_definition_of = (
        ("RUN_LIMIT_HEALING_MESSAGE", errors),
        ("UsageLimitError", errors),
        ("RunUsageLimitError", errors),
        ("AgentUsageLimitError", errors),
        ("LifetimeBudgetCapability", budget),
        ("CompactionCapability", compaction),
        ("EventSourcingCapability", event_sourcing),
        ("HealingCapability", healing),
        ("LimitRecoveryCapability", limit_recovery),
        ("ConclusionDecision", limit_recovery),
        ("DEFAULT_CONCLUSION_REASON", limit_recovery),
    )

    # Also reached through akgentic.llm: every capability class, plus the exception
    # hierarchy. The two wording constants are deliberately absent — neither ever was there.
    reaches_the_package = {
        "UsageLimitError",
        "RunUsageLimitError",
        "AgentUsageLimitError",
        "LifetimeBudgetCapability",
        "CompactionCapability",
        "EventSourcingCapability",
        "HealingCapability",
        "LimitRecoveryCapability",
        "ConclusionDecision",
    }
    # Also reached through akgentic.llm.agent, for code written against their pre-capability
    # home. The recovery names are NEW rather than moved, so they get no such re-export: the
    # ``X as X`` block in agent.py exists for names that used to live there.
    reaches_the_agent_module = {
        "RUN_LIMIT_HEALING_MESSAGE",
        "UsageLimitError",
        "RunUsageLimitError",
        "AgentUsageLimitError",
    }
    declared = {name for name, _ in one_definition_of}
    assert reaches_the_package <= declared
    assert reaches_the_agent_module <= declared

    for name, sibling in one_definition_of:
        assert hasattr(capabilities_package, name), (
            f"{name} is no longer re-exported from akgentic.llm.capabilities"
        )
        one_object = getattr(sibling, name)
        assert getattr(capabilities_package, name) is one_object, (
            f"akgentic.llm.capabilities.{name} is not the object {sibling.__name__} defines"
        )
        if name in reaches_the_package:
            assert getattr(akgentic.llm, name) is one_object, (
                f"akgentic.llm.{name} is not the object {sibling.__name__} defines"
            )
        if name in reaches_the_agent_module:
            assert getattr(agent_module, name) is one_object, (
                f"akgentic.llm.agent.{name} is not the object {sibling.__name__} defines"
            )


def test_catching_the_base_still_catches_both_tiers():
    """``except UsageLimitError`` written against the old home catches both subclasses.

    The additive guarantee the hierarchy's move must not break: a caller that imported the
    base from ``akgentic.llm.agent`` before the split keeps catching everything it used to.
    """
    from akgentic.llm.agent import (
        AgentUsageLimitError,
        RunUsageLimitError,
        UsageLimitError,
    )

    for tier in (RunUsageLimitError, AgentUsageLimitError):
        assert issubclass(tier, UsageLimitError)
        with pytest.raises(UsageLimitError):
            raise tier("breach")
