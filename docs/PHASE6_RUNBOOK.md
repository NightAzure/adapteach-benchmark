# Phase 6 Runbook

This runbook covers validators and reliability controls.

## 1. Scope

Phase 6 adds:

- JSON schema validation per artifact type
- Deterministic correctness checks
- Unsafe content blocking
- Validation-driven fallback strategy
- Case bundle logging for failed validations
- Sandbox harness and policy

## 2. Artifact schemas

Implemented schemas:

- `src/validation/schemas/parsons.schema.json`
- `src/validation/schemas/tracing.schema.json`
- `src/validation/schemas/mutation.schema.json`
- `src/validation/schemas/flashcard.schema.json`

Validator module:

- `src/validation/validator.py`

## 3. Deterministic checks

Implemented checks:

- parsability (`ast.parse`) for code-bearing artifacts
- tracing step shape and bounds
- mutation is real/localized + fix parsability
- parsons solution ordering parsability + distractor ambiguity checks

Module:

- `src/validation/deterministic_checks.py`

## 4. Unsafe content rules

Blocks content patterns such as:

- URLs
- secret/API-key patterns
- password/secret markers

Module:

- `src/validation/schema_validator.py`

## 5. Fallback behavior

Deterministic fallback rules:

- weak retrieval: clarifying-question fallback
- validator failure: deterministic regeneration attempts
- final validator failure: conservative safe response + case bundle
- graph expansion cap/degrade to base candidates

Modules:

- `src/validation/fallbacks.py`
- `src/validation/case_bundle.py`

## 6. Sandbox harness

Module:

- `src/sandbox/harness.py`

Current policy defaults to static checks only (`allow_execution=false`).

Telemetry:

- `logs/sandbox_telemetry.jsonl`

## 7. Pipeline integration

Validation now runs in `src/pipelines/runner.py` and stores validator outcomes in run logs.

Case bundles on hard validation failure are saved to:

- `logs/case_bundles/<trace_id>.json`

## 8. LLM integration note

LLM generation integration begins in Phase 5 (generation stage across configs A-E).  
Phase 6 wraps that generation path with strict validation and deterministic reliability controls.
