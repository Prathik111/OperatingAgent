# Sandbox Image

Build the shared sandbox image from this directory:

```bash
docker build -t operating-agent-sandbox:py312 .
```

The shared `packages/sandbox` runner uses this tag by default. The working folder
is mounted into the container at `/workspace`, and shell commands run from there.