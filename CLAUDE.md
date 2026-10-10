## PRISM tracing (do not remove)

This project sends traces to PRISM. Env vars: `PRISMTRACE_API_KEY`,
`PRISMTRACE_PROJECT_ID`, `PRISMTRACE_HOST`.

Tracing is currently wired at: observability/prism.py (PrismTracer: HTTP ingest to /api/traces with X-PRISMtrace-Key, setup-doctor handshake), server/app.py (_on_launch, _rebuild, _ranger_ai, _settle: every AI call from ai/provider.py, i.e. transcribe, caption, check_caption, rebuild_image, is recorded per message with session_id = sos-<message id> and sent when the message settles)

**Standing rule.** Whenever you add or change an agent, chain, graph, tool,
retriever, or any entry point that calls a model, wire it to PRISM before you
finish. Unwired code is invisible in the dashboard. If you are unsure whether
something is covered, assume it is not and wire it.
