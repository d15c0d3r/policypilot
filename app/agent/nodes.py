from typing import Annotated
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from app.tools.provider_tools import list_providers, get_provider_details
from app.tools.policy_tools import search_policy, compare_policies
from app.agent.state import AgentState

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# ── Specialist agent subgraph builder ──
# This is the manual ReAct loop: call_model → (has tool calls?) → call_tools → call_model → ... → END
# we can also use create_react_agent from langgraph.prebuilt to create a specialist agent but we want to keep it manual for learning purpose.

class SpecialistState(TypedDict):
    messages: Annotated[list, add_messages]


def build_specialist_agent(system_prompt: str, tools: list):
    """Build a LangGraph subgraph that implements a ReAct tool-calling loop.

    Flow: call_model → should_continue? → call_tools → call_model → ... → END
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    def call_model(state: SpecialistState) -> dict:
        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def call_tools(state: SpecialistState) -> dict:
        last_message = state["messages"][-1]
        results = []
        for tool_call in last_message.tool_calls:
            tool_fn = tool_map[tool_call["name"]]
            result = tool_fn.invoke(tool_call["args"])
            results.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
        return {"messages": results}

    def should_continue(state: SpecialistState) -> str:
        last_message = state["messages"][-1]
        # safety chain, should_continue always receives an AIMessage (since it runs right after call_model), so the hasattr check is defensive. It prevents a crash if something unexpected ends up as the last message.
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "call_tools"
        return END

    graph = StateGraph(SpecialistState)
    graph.add_node("call_model", call_model)
    graph.add_node("call_tools", call_tools)
    graph.set_entry_point("call_model")
    graph.add_conditional_edges(
        "call_model",
        should_continue,
        {"call_tools": "call_tools", END: END},
    )
    graph.add_edge("call_tools", "call_model")

    return graph.compile()


# ── Supervisor: routes to the right specialist ──

SUPERVISOR_PROMPT = """You are a supervisor agent for PolicyPilot, an insurance policy assistant.
Your job is to analyze the user's query and decide which specialist agent should handle it.

You MUST respond with EXACTLY one of these agent names:
- "provider_agent": for questions about which providers exist, their settlement ratio, or active status
- "policy_expert": for detailed policy questions (coverage, plans, exclusions, claims, waiting periods, etc.)
- "comparison_agent": for comparing policies or features across different providers or categories
- "guardrail": for queries unrelated to insurance

Examples:
- "What insurance companies do you have?" → provider_agent
- "What is the claim settlement ratio?" → provider_agent
- "What plans does this provider offer?" → policy_expert
- "Does my policy cover pre-existing conditions?" → policy_expert
- "Compare these two providers on coverage" → comparison_agent
- "What's the weather today?" → guardrail

Respond with ONLY the agent name, nothing else."""


def supervisor_node(state: AgentState) -> dict:
    messages = [SystemMessage(content=SUPERVISOR_PROMPT), *state["messages"]]
    response = llm.invoke(messages)
    next_agent = response.content.strip().lower().replace('"', "")

    valid_agents = {"provider_agent", "policy_expert",
                    "comparison_agent", "guardrail"}
    if next_agent not in valid_agents:
        next_agent = "policy_expert"

    return {"next_agent": next_agent}


# ── Guardrail: handles off-topic queries ──

def guardrail_node(state: AgentState) -> dict:
    return {
        "messages": [
            AIMessage(
                content=(
                    "I'm PolicyPilot, your insurance policy assistant. I can help you with:\n"
                    "- Listing available insurance providers\n"
                    "- Detailed policy information (coverage, exclusions, claims, etc.)\n"
                    "- Comparing policies of the same kind (e.g. two health insurance plans)\n\n"
                    "Could you please ask a question related to your uploaded policies?"
                )
            )
        ]
    }


# ── Build specialist subgraphs ──

PROVIDER_AGENT_PROMPT = """You are the Provider Agent for PolicyPilot.
You specialize in answering questions about available insurance providers,
their claim settlement ratios, and whether they are currently active.

Use the tools available to you to look up provider information.
Be concise, helpful, and accurate. Only answer based on the data from your tools."""

_provider_agent = build_specialist_agent(
    PROVIDER_AGENT_PROMPT,
    [list_providers, get_provider_details],
)


POLICY_EXPERT_PROMPT = """You are the Policy Expert Agent for PolicyPilot.
You specialize in answering detailed questions about insurance policies
by searching through actual policy documents.

Use the search_policy tool to find relevant information from policy PDFs.
You can optionally filter by category (health_insurance, car_insurance, term_insurance, etc.).

CRITICAL RULES:
- Only answer based on information retrieved from the policy documents.
- If the search returns no relevant results, say "I couldn't find that information in the policy documents."
- Always cite which document the information comes from.
- Never make up policy details."""

_policy_expert_agent = build_specialist_agent(
    POLICY_EXPERT_PROMPT,
    [search_policy],
)


COMPARISON_AGENT_PROMPT = """You are the Comparison Agent for PolicyPilot.
You specialize in comparing insurance policies across different providers and categories.

Use the compare_policies tool to retrieve information from multiple providers simultaneously.
You can also use get_provider_details for structured metadata comparisons.

Present comparisons in a clear, structured format. Highlight key differences.
Only use information from the tools — never fabricate policy details."""

_comparison_agent = build_specialist_agent(
    COMPARISON_AGENT_PROMPT,
    [compare_policies, get_provider_details],
)


# ── Node wrappers that invoke subgraphs and return results to parent graph ──

def provider_agent_node(state: AgentState) -> dict:
    result = _provider_agent.invoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["messages"][-1].content)]}


def policy_expert_node(state: AgentState) -> dict:
    result = _policy_expert_agent.invoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["messages"][-1].content)]}


def comparison_agent_node(state: AgentState) -> dict:
    result = _comparison_agent.invoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["messages"][-1].content)]}
