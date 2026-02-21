from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.agent.state import AgentState
from app.agent.nodes import (
    supervisor_node,
    guardrail_node,
    provider_agent_node,
    policy_expert_node,
    comparison_agent_node,
)


def route_to_agent(state: AgentState) -> str:
    return state["next_agent"]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("provider_agent", provider_agent_node)
    graph.add_node("policy_expert", policy_expert_node)
    graph.add_node("comparison_agent", comparison_agent_node)
    graph.add_node("guardrail", guardrail_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "provider_agent": "provider_agent",
            "policy_expert": "policy_expert",
            "comparison_agent": "comparison_agent",
            "guardrail": "guardrail",
        },
    )

    graph.add_edge("provider_agent", END)
    graph.add_edge("policy_expert", END)
    graph.add_edge("comparison_agent", END)
    graph.add_edge("guardrail", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
