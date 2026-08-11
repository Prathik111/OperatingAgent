from .server import build_gateway, mcp

__all__ = ["build_gateway", "mcp", "main"]


def main() -> None:
    mcp.run()
