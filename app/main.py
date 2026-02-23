import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.agent.graph import build_graph


def main():
    load_dotenv()

    print("=" * 60)
    print("  PolicyPilot - Policy Assistant")
    print("=" * 60)
    print("Initializing...")

    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\nReady! Ask me about your uploaded policy documents.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        ai_message = result["messages"][-1]
        print(f"\nPolicyPilot: {ai_message.content}\n")


if __name__ == "__main__":
    main()
