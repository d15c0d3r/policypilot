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
    """List all available health insurance providers in the system.
    Returns provider names, IDs, and a brief summary of each."""
    providers = _load_providers()
    lines = []
    for p in providers:
        lines.append(
            f"- **{p['name']}** (id: {p['id']}): {p['type']} | "
            f"Plans: {', '.join(p['plans'])} | "
            f"Network Hospitals: {p['network_hospitals']}"
        )
    return "\n".join(lines)


class ProviderDetailsInput(BaseModel):
    provider_id: str = Field(
        description="The provider ID to look up. Must be 'hdfc' or 'icici'."
    )


@tool(args_schema=ProviderDetailsInput)
def get_provider_details(provider_id: str) -> str:
    """Get detailed metadata for a specific insurance provider including plans,
    premiums, coverage, claim settlement ratio, and key features."""
    providers = _load_providers()
    provider = next((p for p in providers if p["id"] == provider_id), None)

    if not provider:
        available = [p["id"] for p in providers]
        return f"Provider '{provider_id}' not found. Available providers: {available}"

    return (
        f"**{provider['full_name']}**\n"
        f"Type: {provider['type']}\n"
        f"Plans offered: {', '.join(provider['plans'])}\n"
        f"Premium range: {provider['premium_range']}\n"
        f"Coverage range: {provider['coverage_range']}\n"
        f"Claim settlement ratio: {provider['claim_settlement_ratio']}\n"
        f"Network hospitals: {provider['network_hospitals']}\n"
        f"Key features:\n" +
        "\n".join(f"  - {f}" for f in provider["key_features"])
    )
