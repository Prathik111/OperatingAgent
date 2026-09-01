from .server import build_gateway, mcp

__all__ = ["build_gateway", "main", "mcp"]


def main() -> None:
    mcp.run(transport="stdio", show_banner=False)
