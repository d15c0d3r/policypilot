from typing import Annotated
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

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
