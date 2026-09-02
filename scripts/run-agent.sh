#!/usr/bin/env bash
# Run the native agent from the OperatingAgent folder.
#
#   ./scripts/run-agent.sh "read README.md and tell me what this project is"
#   ./scripts/run-agent.sh "write a summary to notes.txt" --dir ./scratch --ask
#
# Checks the key and the install first, because the two things most likely to
# bite on a first run are a missing GROQ_API_KEY and a plain `uv sync` (which
# installs nothing here - the workspace root is virtual, so members need
# --all-packages).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [ $# -eq 0 ]; then
    echo "usage: $0 \"your message\" [--dir FOLDER] [--ask] [--model NAME] [extra flags]" >&2
    exit 2
fi
message="$1"; shift

command -v uv >/dev/null 2>&1 || {
    echo "uv is not on PATH. Install it: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
}

# The key can come from the shell or from a .env at the repo root; the agent
# loads .env itself, and an exported variable wins over the file.
if [ -z "${GROQ_API_KEY:-}" ] && [ ! -f "$root/.env" ]; then
    echo "No Groq API key found."
    echo "  This shell only:  export GROQ_API_KEY=gsk_..."
    echo "  Or create $root/.env containing:  GROQ_API_KEY=gsk_..."
    exit 1
fi

# --all-packages is required: the root is a virtual workspace root with no
# dependencies, so a bare `uv sync` leaves agent-native uninstalled.
if [ "${SKIP_SYNC:-}" != "1" ]; then
    echo "==> uv sync --all-packages"
    uv sync --all-packages
fi

# Default the working folder to ./scratch so a first run can't touch the source.
case " $* " in
    *" --dir "*|*" --dir="*) ;;
    *) mkdir -p "$root/scratch"; set -- "$@" --dir "$root/scratch" ;;
esac

echo
exec uv run agent-native -m "$message" "$@"
