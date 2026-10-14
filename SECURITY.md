# Security policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Report privately via
GitHub Security Advisories ("Report a vulnerability" in the *Security* tab) or by
e-mail to the maintainer listed in `pyproject.toml`.

We aim to acknowledge within 3 business days and to ship a fix or mitigation
within 30 days for confirmed issues.

## Scope and known limitations

AuraLite is an educational framework; the following are **documented, not bugs**:

- `auralite/agent/sandbox.py` is a subprocess sandbox with sanitized environment, resource
  limits and tree-kill — it is **not** an OS-level isolation boundary. A
  determined payload can still open sockets unless the process is containerized.
- Checkpoints: `.pt` files are loaded with `weights_only=True`. The escape hatch
  (`allow_unsafe=True` / `AURALITE_UNSAFE_LOAD=1`) exists for trusted legacy files
  and **does** execute arbitrary code by design.
- The OpenAI-compatible server is single-worker by default; API-key auth is
  optional and disabled unless `AURALITE_API_KEY` is set. Never expose it to the
  internet without a key, a reverse proxy and TLS.
