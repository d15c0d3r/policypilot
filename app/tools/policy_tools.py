from langchain_core.tools import tool
from pydantic import BaseModel, Field
from app.data.ingest import get_vectorstore, PDF_CATEGORIES


class PolicySearchInput(BaseModel):
    query: str = Field(description="The policy-related question to search for.")
    category: str | None = Field(
        default=None,
        description=f"Optional. Filter results to a specific category: {PDF_CATEGORIES}. "
        "Leave empty to search across all uploaded documents.",
    )


@tool(args_schema=PolicySearchInput)
def search_policy(query: str, category: str | None = None) -> str:
    """Search uploaded policy documents for relevant information.
    Can optionally filter by category (health_insurance, car_insurance, term_insurance, etc.).
    Use this for detailed policy questions about coverage, exclusions,
    claims process, waiting periods, etc."""
    vectorstore = get_vectorstore()

    search_kwargs = {"k": 5}
    if category:
        search_kwargs["filter"] = {"category": category}

    docs = vectorstore.similarity_search(query, **search_kwargs)

    if not docs:
        return "No relevant information found in the uploaded policy documents."

    results = []
    for i, doc in enumerate(docs, 1):
        cat = doc.metadata.get("category", "unknown")
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        results.append(
            f"[Source {i}: {cat} / {source} - Page {page}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(results)


class ComparePoliciesInput(BaseModel):
    query: str = Field(
        description="The aspect to compare between categories "
        "(e.g., 'waiting period for pre-existing conditions', 'coverage limits')."
    )
    categories: list[str] = Field(
        description=f"Two or more categories to compare. Allowed: {PDF_CATEGORIES}",
    )


@tool(args_schema=ComparePoliciesInput)
def compare_policies(query: str, categories: list[str] | None = None) -> str:
    """Compare policy information across different categories for a specific aspect.
    Retrieves relevant chunks from each category for side-by-side comparison."""
    vectorstore = get_vectorstore()
    cats = categories or PDF_CATEGORIES

    sections = []
    for cat in cats:
        label = cat.replace("_", " ").title()
        docs = vectorstore.similarity_search(
            query, k=3, filter={"category": cat}
        )
        if docs:
            content = "\n".join(doc.page_content for doc in docs)
            sections.append(f"**{label}:**\n{content}")
        else:
            sections.append(f"**{label}:**\nNo relevant information found.")

    return "\n\n---\n\n".join(sections)
