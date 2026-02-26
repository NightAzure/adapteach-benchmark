# Fallback Policy

Deterministic fallback behavior used by the pipeline.

## 1. Retrieval weak fallback

Trigger:

- empty retrieval context, or very low retrieval confidence

Action:

- return safe clarifying question instead of speculative answer

## 2. Validator failure fallback

Trigger:

- schema/unsafe/deterministic checks fail

Action:

1. Deterministic regeneration up to `N=2` attempts
2. If still invalid:
   - return conservative explanation
   - write case bundle for audit/debug

## 3. Graph expansion fallback

Trigger:

- expanded candidate set exceeds safety cap

Action:

- cap expansion size
- degrade to base retrieval candidates (Config C/D style)

## 4. No silent failure rule

All failures must be reflected in logs:

- run log validation stage
- case bundle artifact when hard failure occurs
