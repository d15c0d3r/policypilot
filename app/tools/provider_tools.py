import json
from pathlib import Path
from langchain_core.tools import tool
from pydantic import BaseModel, Field

PROVIDERS_FILE = Path(__file__).parent.parent / "data" / "providers.json"


def _load_providers() -> list[dict]:
    with open(PROVIDERS_FILE) as f:
        return json.load(f)["providers"]


@tool
def list_providers() -> str:
    """List all available insurance providers in the system.
    Returns provider names, IDs, and whether they are currently active."""
    providers = _load_providers()
    lines = []
    for p in providers:
        status = "Active" if p["active"] else "Inactive"
        lines.append(
            f"- **{p['name']}** (id: {p['id']}): {p['type']} | "
            f"Claim Settlement Ratio: {p['claim_settlement_ratio']} | "
            f"Status: {status}"
        )
    return "\n".join(lines)


class ProviderDetailsInput(BaseModel):
    provider_id: str = Field(
        description="The provider ID to look up (e.g. 'hdfc', 'icici')."
    )


@tool(args_schema=ProviderDetailsInput)
def get_provider_details(provider_id: str) -> str:
    """Get metadata for a specific insurance provider including
    claim settlement ratio and whether they are currently active."""
    providers = _load_providers()
    provider = next((p for p in providers if p["id"] == provider_id), None)

    if not provider:
        available = [p["id"] for p in providers]
        return f"Provider '{provider_id}' not found. Available providers: {available}"

    status = "Currently active" if provider["active"] else "No longer active"
    return (
        f"**{provider['full_name']}**\n"
        f"Type: {provider['type']}\n"
        f"Claim settlement ratio: {provider['claim_settlement_ratio']}\n"
        f"Status: {status}"
    )
