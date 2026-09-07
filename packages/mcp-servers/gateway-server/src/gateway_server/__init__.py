from .server import build_gateway, mcp

__all__ = ["build_gateway", "main", "mcp"]


def main() -> None:
    import os

    workspace = os.environ.get("OPERATING_AGENT_WORKSPACE")
    build_gateway(root=workspace).run(transport="stdio", show_banner=False)
