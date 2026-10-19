"""Tiny example MCP server with distinctive inventory tools."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("inventory")

STOCK = {
    "WIDGET-42": {"name": "Blue widget", "warehouse": "Rotterdam", "quantity": 137},
    "GADGET-7": {"name": "Gadget mk7", "warehouse": "Lisbon", "quantity": 0},
    "SPROCKET-9": {"name": "Brass sprocket", "warehouse": "Tallinn", "quantity": 12},
}


@mcp.tool()
def get_inventory_stock(sku: str) -> dict:
    """Look up warehouse inventory stock level for a product SKU."""
    item = STOCK.get(sku.upper())
    if item is None:
        return {"sku": sku, "found": False}
    return {"sku": sku.upper(), "found": True, **item}


@mcp.tool()
def list_inventory_skus() -> list[str]:
    """List all product SKUs tracked in the warehouse inventory."""
    return sorted(STOCK)


if __name__ == "__main__":
    mcp.run()
