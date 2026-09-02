from akgentic.llm import ReactAgentConfig, ModelConfig, ReactAgent


# ---------------------------------------------------------------------------
# Tool execution with ReactAgent
# ---------------------------------------------------------------------------


def find_case_description(case_id: str) -> str:
    """Returns the case belonging to the given case_id. If case unknown, raise ValueError"""

    if not case_id or case_id != "case_1":
        raise ValueError("Need an existing case id to fetch it.")
    return "The printer on the second floor is not working"


def test_react_agent_find_case_description() -> None:
    """Proves failure when running ReactAgent with find_case_description tool."""
    arg_case_id = "case_1"
    config = ReactAgentConfig(
        # model_cfg=ModelConfig(provider="openai", model="gpt-4o-mini")
        model_cfg=ModelConfig(provider="openai", model="gpt-5.6-luna")
    )
    agent = ReactAgent(config=config, tools=[find_case_description])
    result = agent.run_sync(
        "Determine the priority of the case. You can use the description to determine the priority. "
        + "Use a tool to find the case description. Extract the case_id from the question."
        + "If you cannot find the case_id, tell the user that you cannot extract the case_id."
        + "Use the case description to determine the priority. Return a message with the priority "
        + "and the reason why this priority. "
        + f"The case id is {arg_case_id}."
    )
    assert result is not None
