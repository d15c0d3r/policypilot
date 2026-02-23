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
        description="The aspect to compare (e.g., 'waiting period', 'ambulance coverage', 'claim process')."
    )
    category: str | None = Field(
        default=None,
        description=f"Optional. The insurance category to compare within: {PDF_CATEGORIES}. "
        "Leave empty to auto-detect from uploaded documents.",
    )


@tool(args_schema=ComparePoliciesInput)
def compare_policies(query: str, category: str | None = None) -> str:
    """Compare policy information between different providers.
    Retrieves relevant chunks grouped by source document for side-by-side comparison.
    Optionally filter by category. If results span multiple categories, a warning is returned
    asking the user to specify the insurance type."""
    vectorstore = get_vectorstore()

    search_kwargs = {"k": 8}
    if category:
        if category not in PDF_CATEGORIES:
            return f"Invalid category '{category}'. Allowed categories: {PDF_CATEGORIES}"
        search_kwargs["filter"] = {"category": category}

    docs = vectorstore.similarity_search(query, **search_kwargs)

    if not docs:
        return "No relevant information found in the uploaded policy documents."

    categories_found = set(doc.metadata.get("category", "unknown") for doc in docs)
    if len(categories_found) > 1 and not category:
        cats = ", ".join(sorted(categories_found))
        return (
            f"The results span multiple insurance categories ({cats}). "
            "Policies can only be compared within the same type. "
            "Please specify which insurance type you'd like to compare."
        )

    by_source: dict[str, list[str]] = {}
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        by_source.setdefault(source, []).append(doc.page_content)

    if len(by_source) < 2:
        cat_label = (category or list(categories_found)[0]).replace("_", " ")
        return (
            f"Only found documents from one provider in {cat_label}. "
            "Please upload policies from at least two providers to enable comparison."
        )

    sections = []
    for source, chunks in by_source.items():
        label = source.replace(".pdf", "").replace("-", " ").replace("_", " ").title()
        content = "\n".join(chunks)
        sections.append(f"**{label}:**\n{content}")

    return "\n\n---\n\n".join(sections)
