"""Tests for prompt template and rendering."""

import pytest

from akgentic.llm.prompts import PromptTemplate, render_prompt


class TestPromptTemplate:
    """Tests for PromptTemplate model."""

    def test_instantiation_with_template_and_params(self):
        """Test creating PromptTemplate with template and params."""
        # Subtask 3.2
        tpl = PromptTemplate(
            template="You are {role}.\n{instructions}",
            params={"role": "Architect", "instructions": "Design systems."},
        )
        assert tpl.template == "You are {role}.\n{instructions}"
        assert tpl.params == {"role": "Architect", "instructions": "Design systems."}

    def test_instantiation_with_default_empty_params(self):
        """Test PromptTemplate defaults to empty params dict."""
        # Subtask 3.3
        tpl = PromptTemplate(template="Simple template with no placeholders")
        assert tpl.template == "Simple template with no placeholders"
        assert tpl.params == {}

    def test_pydantic_serialization_model_dump(self):
        """Test Pydantic model_dump serialization."""
        # Subtask 3.8
        tpl = PromptTemplate(
            template="Hello {name}",
            params={"name": "World"},
        )
        data = tpl.model_dump()
        assert data == {
            "template": "Hello {name}",
            "params": {"name": "World"},
        }

    def test_pydantic_serialization_model_dump_json(self):
        """Test Pydantic model_dump_json serialization."""
        # Subtask 3.8
        tpl = PromptTemplate(
            template="Hello {name}",
            params={"name": "World"},
        )
        json_str = tpl.model_dump_json()
        assert '"template":"Hello {name}"' in json_str
        assert '"params":{"name":"World"}' in json_str

    def test_round_trip_serialization(self):
        """Test round-trip: model_dump → PromptTemplate(**data)."""
        # Subtask 3.9
        original = PromptTemplate(
            template="Role: {role}",
            params={"role": "Engineer"},
        )
        data = original.model_dump()
        restored = PromptTemplate(**data)
        assert restored.template == original.template
        assert restored.params == original.params


class TestRenderPrompt:
    """Tests for render_prompt function."""

    def test_passthrough_for_plain_string(self):
        """Test render_prompt returns plain string as-is."""
        # Subtask 3.4
        result = render_prompt("Simple string prompt")
        assert result == "Simple string prompt"

    def test_interpolation_with_params(self):
        """Test render_prompt interpolates PromptTemplate params."""
        # Subtask 3.5
        tpl = PromptTemplate(
            template="You are {role}.\n{instructions}",
            params={"role": "Architect", "instructions": "Design systems."},
        )
        result = render_prompt(tpl)
        assert result == "You are Architect.\nDesign systems."

    def test_raises_key_error_for_missing_param(self):
        """Test render_prompt raises KeyError when param is missing."""
        # Subtask 3.6
        tpl = PromptTemplate(
            template="You are {role}.\n{instructions}",
            params={"role": "Architect"},  # missing 'instructions'
        )
        with pytest.raises(KeyError):
            render_prompt(tpl)

    def test_empty_params_with_no_placeholders(self):
        """Test render_prompt works with empty params when template has no placeholders."""
        # Subtask 3.7
        tpl = PromptTemplate(
            template="Simple template with no placeholders",
            params={},
        )
        result = render_prompt(tpl)
        assert result == "Simple template with no placeholders"

    def test_empty_params_with_default(self):
        """Test render_prompt works when params not specified (default empty dict)."""
        # Subtask 3.7 extension
        tpl = PromptTemplate(template="No placeholders here")
        result = render_prompt(tpl)
        assert result == "No placeholders here"
