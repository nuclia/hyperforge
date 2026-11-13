from _run import main
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")


@mcp.tool()
def get_forecast(city: str, days: int = 3) -> str:
    """Get the weather forecast (temperature, rain, wind) for a city for the next days."""
    return f"Forecast for {city}: sunny for {days} days, 21C"


@mcp.tool()
def get_alerts(region: str) -> str:
    """Get active severe weather alerts such as storms, floods and heat warnings for a region."""
    return f"No active alerts for {region}"


if __name__ == "__main__":
    main(mcp)
