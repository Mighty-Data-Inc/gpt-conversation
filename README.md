# llm-conversation

`llm-conversation` is a cross-language toolkit for building reliable LLM-powered features.

The project exists to remove repeated integration work from application teams. Instead of re-implementing the same conversation and structured-output patterns in every codebase, this repository provides a shared, production-oriented foundation that stays provider-agnostic in design.

## Why This Project Exists

- LLM features are often simple in concept but fragile in implementation.
- Teams repeatedly rebuild the same reliability and formatting layers.
- Provider APIs evolve quickly, while business workflows need stable behavior.
- Cross-language organizations need comparable semantics, not duplicated design work.

## What This Project Offers

This toolkit provides reusable building blocks for the parts of LLM integration that most often become brittle in production:

- Stateful conversation flows for multi-step prompting and revision workflows.
- Stateless request/response utilities for one-shot tasks.
- Structured JSON output handling, including advisory and schema-constrained patterns.
- Schema shorthand utilities that reduce verbosity when defining structured responses.
- Reliability helpers such as retries, backoff, and optional multi-worker "shotgunning" patterns.
- Cross-language alignment so Python and TypeScript implementations behave consistently.

## Where It Helps Most

- Normalizing external-source data (client, vendor, or user-provided fields) into canonical internal structures.
- Building backend workflows that require predictable machine-readable LLM output.
- Running multi-turn transformations where intermediate context and role management matter.
- Reducing hand-rolled parsing logic and prompt-only output control strategies.

## Scope

This repository focuses on parts of LLM integration that are easy to get wrong repeatedly:

- Managing multi-message conversations over time.
- Working with structured JSON outputs safely.
- Keeping shared semantics aligned across implementations.

This repository is intentionally _not_ an agent framework, orchestration platform, or full application starter.

## Repository Layout

- `packages/python-llm-conversation`: Python package implementation.
- `packages/typescript-llm-conversation`: TypeScript package implementation.

Both packages follow the same product intent and are developed together to maintain consistency.

## Design Principles

- Minimal, composable abstractions.
- Explicit behavior and stable contracts.
- Production-first defaults.
- Parity where shared behavior matters most.
- Provider-agnostic design with clear adaptation points.

## Documentation

The root README describes intent and project-level context.

For package usage, API reference, and language-specific setup, see:

- `packages/python-llm-conversation/README.md`
- `packages/typescript-llm-conversation/README.md`

## Quality and Reliability

The project prioritizes practical correctness over synthetic demos. Package-level test suites are designed to protect real integration behavior, including conversation-state management and structured-output workflows.

## Releases

Python and TypeScript packages are versioned and released from this monorepo using repository automation. Release mechanics are kept with each package and workflow configuration.

## License

MIT. See `LICENSE`.
