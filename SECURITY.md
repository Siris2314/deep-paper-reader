# Security Policy

Deep Paper Reader processes untrusted PDFs and model output and can call external research or model services. Treat agent instructions as behavior guidance, not a security boundary.

## Trust boundaries

- Local Ollama inference keeps prompts on the configured Ollama host.
- Tavily, OpenAI, Semantic Scholar, and LangSmith receive data only when configured and invoked by their workflows.
- PDF text, web pages, citations, and model output may contain prompt injection or malformed content.
- Tools should receive the minimum context and filesystem access needed for their task.
- Generated code and implementation suggestions are explanatory output and must not be executed automatically.

## Secrets

Keep keys in `.env`, which is ignored by Git. Never place secrets in prompts, PDFs, screenshots, logs, tests, example workspaces, or issues. Rotate a key immediately if it is exposed.

## Supported versions

Until the first stable release, security fixes are applied to the latest `main` branch only.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private vulnerability reporting or Security Advisory flow for the repository. Include reproduction steps, affected paths, impact, and whether an external service or a crafted PDF is required.
