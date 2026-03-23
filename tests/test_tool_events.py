"""Tests for ToolCallEvent and ToolReturnEvent emission from ContextManager.

Covers FR-TCE-8: event emission logic for all tool-related message part kinds.

Note: Uses the same importlib loading pattern as test_context.py to avoid
pulling in the providers dependency (pydantic-ai providers require API keys
at import time in some configurations). ToolCallEvent / ToolReturnEvent are
imported directly from akgentic.llm because they have no providers transitive dep.
"""

import importlib.util
from datetime import datetime
from pathlib import Path

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

# Load context module via importlib to avoid providers transitive dependency
_CONTEXT_PATH = Path(__file__).parent.parent / "src" / "akgentic" / "llm" / "context.py"
_spec = importlib.util.spec_from_file_location("context", _CONTEXT_PATH)
_context_module = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_context_module)  # type: ignore[union-attr]

ContextManager = _context_module.ContextManager

# Import event classes directly — no providers dep in event.py
from akgentic.llm.event import LlmMessageEvent, ToolCallEvent, ToolReturnEvent  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class EventCapture:
    """Captures all domain events in emission order."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def notify_event(self, event: object) -> None:
        """Append each received event for ordered assertion."""
        self.events.append(event)


def _make_manager_with_capture() -> tuple[ContextManager, EventCapture]:
    """Return a ContextManager wired to an EventCapture observer."""
    manager = ContextManager()
    capture = EventCapture()
    manager.subscribe(capture)
    return manager, capture


def _tool_call_msg(
    tool_name: str = "web_search_tool",
    tool_call_id: str = "call_abc123",
    args: str = '{"query": "test"}',
) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name=tool_name, tool_call_id=tool_call_id, args=args)],
        timestamp=datetime.now(),
    )


def _tool_return_msg(
    tool_name: str = "web_search_tool",
    tool_call_id: str = "call_abc123",
    content: str = "search results here",
) -> ModelRequest:
    return ModelRequest(
        parts=[ToolReturnPart(tool_name=tool_name, tool_call_id=tool_call_id, content=content)]
    )


def _retry_with_id_msg(
    tool_name: str = "web_search_tool",
    tool_call_id: str = "call_abc123",
    content: str = "Tool raised error",
) -> ModelRequest:
    return ModelRequest(
        parts=[
            RetryPromptPart(tool_name=tool_name, tool_call_id=tool_call_id, content=content)
        ]
    )


def _retry_no_id_msg(content: str = "Please retry in the correct format.") -> ModelRequest:
    return ModelRequest(parts=[RetryPromptPart(content=content)])


def _text_msg(content: str = "Hello world") -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=content)], timestamp=datetime.now())


# ---------------------------------------------------------------------------
# AC-2: single tool-call part
# ---------------------------------------------------------------------------


class TestSingleToolCall:
    """AC-2: single tool-call part → LlmMessageEvent then ToolCallEvent."""

    def test_lmm_message_event_emitted_first(self) -> None:
        """LlmMessageEvent must precede ToolCallEvent."""
        manager, capture = _make_manager_with_capture()
        msg = _tool_call_msg()
        manager.add_message(msg)

        assert len(capture.events) == 2
        assert isinstance(capture.events[0], LlmMessageEvent)
        assert isinstance(capture.events[1], ToolCallEvent)

    def test_llm_message_event_carries_correct_message(self) -> None:
        """LlmMessageEvent.message must be the message passed to add_message()."""
        manager, capture = _make_manager_with_capture()
        msg = _tool_call_msg()
        manager.add_message(msg)

        llm_event = capture.events[0]
        assert isinstance(llm_event, LlmMessageEvent)
        assert llm_event.message is msg

    def test_tool_call_event_fields(self) -> None:
        """ToolCallEvent fields must match the part's attributes."""
        manager, capture = _make_manager_with_capture()
        msg = _tool_call_msg(
            tool_name="my_tool",
            tool_call_id="call_xyz",
            args='{"param": "value"}',
        )
        manager.add_message(msg)

        event = capture.events[1]
        assert isinstance(event, ToolCallEvent)
        assert event.tool_name == "my_tool"
        assert event.tool_call_id == "call_xyz"
        assert event.arguments == '{"param": "value"}'


# ---------------------------------------------------------------------------
# AC-3: two tool-call parts in one response
# ---------------------------------------------------------------------------


class TestParallelToolCalls:
    """AC-3: two tool-call parts in one ModelResponse."""

    def test_two_tool_call_events_emitted_in_part_order(self) -> None:
        """Two ToolCallEvent instances emitted after LlmMessageEvent, in part order."""
        manager, capture = _make_manager_with_capture()
        msg = ModelResponse(
            parts=[
                ToolCallPart(tool_name="tool_a", tool_call_id="call_001", args='{"x": 1}'),
                ToolCallPart(tool_name="tool_b", tool_call_id="call_002", args='{"y": 2}'),
            ],
            timestamp=datetime.now(),
        )
        manager.add_message(msg)

        assert len(capture.events) == 3
        assert isinstance(capture.events[0], LlmMessageEvent)
        assert isinstance(capture.events[1], ToolCallEvent)
        assert isinstance(capture.events[2], ToolCallEvent)

    def test_parallel_tool_events_correct_fields(self) -> None:
        """Each ToolCallEvent carries fields from its own part."""
        manager, capture = _make_manager_with_capture()
        msg = ModelResponse(
            parts=[
                ToolCallPart(tool_name="tool_a", tool_call_id="call_001", args='{"x": 1}'),
                ToolCallPart(tool_name="tool_b", tool_call_id="call_002", args='{"y": 2}'),
            ],
            timestamp=datetime.now(),
        )
        manager.add_message(msg)

        event_a = capture.events[1]
        event_b = capture.events[2]
        assert isinstance(event_a, ToolCallEvent)
        assert isinstance(event_b, ToolCallEvent)
        assert event_a.tool_name == "tool_a"
        assert event_a.tool_call_id == "call_001"
        assert event_a.arguments == '{"x": 1}'
        assert event_b.tool_name == "tool_b"
        assert event_b.tool_call_id == "call_002"
        assert event_b.arguments == '{"y": 2}'


# ---------------------------------------------------------------------------
# AC-4: tool-return part → ToolReturnEvent(success=True)
# ---------------------------------------------------------------------------


class TestToolReturn:
    """AC-4: tool-return part → ToolReturnEvent with success=True."""

    def test_tool_return_event_emitted(self) -> None:
        """ToolReturnEvent emitted after LlmMessageEvent for tool-return part."""
        manager, capture = _make_manager_with_capture()
        msg = _tool_return_msg()
        manager.add_message(msg)

        assert len(capture.events) == 2
        assert isinstance(capture.events[0], LlmMessageEvent)
        assert isinstance(capture.events[1], ToolReturnEvent)

    def test_tool_return_event_fields(self) -> None:
        """ToolReturnEvent carries correct fields and success=True."""
        manager, capture = _make_manager_with_capture()
        msg = _tool_return_msg(tool_name="my_tool", tool_call_id="call_xyz")
        manager.add_message(msg)

        event = capture.events[1]
        assert isinstance(event, ToolReturnEvent)
        assert event.tool_name == "my_tool"
        assert event.tool_call_id == "call_xyz"
        assert event.success is True


# ---------------------------------------------------------------------------
# AC-5: retry-prompt with tool_call_id → ToolReturnEvent(success=False)
# ---------------------------------------------------------------------------


class TestRetryPromptWithId:
    """AC-5: retry-prompt with tool_call_id → ToolReturnEvent(success=False)."""

    def test_retry_prompt_with_id_emits_tool_return_event(self) -> None:
        """ToolReturnEvent(success=False) emitted for retry-prompt with tool_call_id."""
        manager, capture = _make_manager_with_capture()
        msg = _retry_with_id_msg(tool_name="my_tool", tool_call_id="call_xyz")
        manager.add_message(msg)

        assert len(capture.events) == 2
        assert isinstance(capture.events[0], LlmMessageEvent)
        assert isinstance(capture.events[1], ToolReturnEvent)

    def test_retry_prompt_with_id_event_fields(self) -> None:
        """ToolReturnEvent from retry-prompt has correct tool_call_id and success=False."""
        manager, capture = _make_manager_with_capture()
        msg = _retry_with_id_msg(tool_name="err_tool", tool_call_id="call_err")
        manager.add_message(msg)

        event = capture.events[1]
        assert isinstance(event, ToolReturnEvent)
        assert event.tool_name == "err_tool"
        assert event.tool_call_id == "call_err"
        assert event.success is False


# ---------------------------------------------------------------------------
# AC-6: retry-prompt without tool_call_id → no ToolReturnEvent
# ---------------------------------------------------------------------------


class TestRetryPromptWithoutId:
    """AC-6: retry-prompt with tool_call_id=None emits only LlmMessageEvent."""

    def test_retry_prompt_no_id_emits_only_llm_message_event(self) -> None:
        """Only LlmMessageEvent emitted — no ToolReturnEvent for non-tool retry."""
        manager, capture = _make_manager_with_capture()
        msg = _retry_no_id_msg()
        manager.add_message(msg)

        assert len(capture.events) == 1
        assert isinstance(capture.events[0], LlmMessageEvent)

    def test_retry_prompt_no_id_no_tool_return_event(self) -> None:
        """ToolReturnEvent must NOT appear in captured events."""
        manager, capture = _make_manager_with_capture()
        manager.add_message(_retry_no_id_msg())

        tool_return_events = [e for e in capture.events if isinstance(e, ToolReturnEvent)]
        assert len(tool_return_events) == 0


# ---------------------------------------------------------------------------
# AC-7: text-only ModelResponse → no tool events
# ---------------------------------------------------------------------------


class TestTextOnlyResponse:
    """AC-7: text-only ModelResponse emits only LlmMessageEvent."""

    def test_text_only_emits_only_llm_message_event(self) -> None:
        """TextPart response produces exactly one LlmMessageEvent, no tool events."""
        manager, capture = _make_manager_with_capture()
        manager.add_message(_text_msg())

        assert len(capture.events) == 1
        assert isinstance(capture.events[0], LlmMessageEvent)

    def test_text_only_no_tool_events(self) -> None:
        """No ToolCallEvent or ToolReturnEvent emitted for text-only responses."""
        manager, capture = _make_manager_with_capture()
        manager.add_message(_text_msg())

        assert not any(isinstance(e, ToolCallEvent) for e in capture.events)
        assert not any(isinstance(e, ToolReturnEvent) for e in capture.events)


# ---------------------------------------------------------------------------
# AC-9: public API export
# ---------------------------------------------------------------------------


class TestPublicApiExport:
    """AC-9: ToolCallEvent and ToolReturnEvent importable from akgentic.llm."""

    def test_tool_call_event_importable_from_akgentic_llm(self) -> None:
        """ToolCallEvent must be importable from the top-level package."""
        import akgentic.llm as llm

        assert hasattr(llm, "ToolCallEvent")

    def test_tool_return_event_importable_from_akgentic_llm(self) -> None:
        """ToolReturnEvent must be importable from the top-level package."""
        import akgentic.llm as llm

        assert hasattr(llm, "ToolReturnEvent")

    def test_tool_call_event_in_all(self) -> None:
        """ToolCallEvent must appear in __all__."""
        import akgentic.llm as llm

        assert "ToolCallEvent" in llm.__all__

    def test_tool_return_event_in_all(self) -> None:
        """ToolReturnEvent must appear in __all__."""
        import akgentic.llm as llm

        assert "ToolReturnEvent" in llm.__all__

    def test_direct_import(self) -> None:
        """Direct named import from akgentic.llm must succeed."""
        from akgentic.llm import ToolCallEvent as TCE
        from akgentic.llm import ToolReturnEvent as TRE

        assert TCE is ToolCallEvent
        assert TRE is ToolReturnEvent


# ---------------------------------------------------------------------------
# Dataclass properties: frozen=True
# ---------------------------------------------------------------------------


class TestEventDataclassProperties:
    """Verify event dataclasses are frozen (immutable)."""

    def test_tool_call_event_is_frozen(self) -> None:
        """ToolCallEvent must be immutable (frozen=True)."""
        import dataclasses

        event = ToolCallEvent(tool_name="t", tool_call_id="c", arguments="{}")
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            event.tool_name = "other"  # type: ignore[misc]

    def test_tool_return_event_is_frozen(self) -> None:
        """ToolReturnEvent must be immutable (frozen=True)."""
        import dataclasses

        event = ToolReturnEvent(tool_name="t", tool_call_id="c", success=True)
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            event.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# User-prompt message (non-tool ModelRequest) → no tool events
# ---------------------------------------------------------------------------


class TestNonToolModelRequest:
    """A ModelRequest with a UserPromptPart should emit only LlmMessageEvent."""

    def test_user_prompt_emits_only_llm_message_event(self) -> None:
        """UserPromptPart in ModelRequest produces no tool events."""
        manager, capture = _make_manager_with_capture()
        msg = ModelRequest(parts=[UserPromptPart(content="Hello")])
        manager.add_message(msg)

        assert len(capture.events) == 1
        assert isinstance(capture.events[0], LlmMessageEvent)
