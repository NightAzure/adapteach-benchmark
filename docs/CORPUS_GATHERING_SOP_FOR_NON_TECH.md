# Corpus Gathering SOP for Non-Technical Team Members

This guide explains how to collect and prepare learning content for AdapTeach without coding.

Use this SOP when adding materials to:

- `data/corpus_raw/`

Goal:

- build a high-quality, legal, traceable corpus for retrieval and benchmarking

---

## 1. Quick Start (5-minute version)

1. Find a source document you want to include.
2. Check if reuse is legally allowed (license/terms).
3. Copy the content into a raw JSON file format (template in this guide).
4. Fill required metadata fields.
5. Run review checklist (quality + safety + scope).
6. Save file in `data/corpus_raw/`.

If any legal/ethics doubt exists, do **not** include it yet. Mark it for review.

---

## 2. What Content Should Be Collected

Collect content that helps beginner Python learners in target topics:

- variables
- loops
- conditionals
- functions
- common misconceptions and debugging explanations
- beginner-friendly exercises/examples

Prefer:

- clear explanations
- short runnable examples
- concept-focused practice material

Avoid:

- advanced/off-topic material
- harmful code
- copyrighted content without permission
- personally identifiable data

---

## 3. Required File Format

Each raw file should be a JSON document.

Required fields:

- `title` (string)
- `content` (string)
- `type` (string: tutorial/example/misconception/exercise/etc.)
- `concept_tags` (array of strings)

Recommended fields:

- `difficulty` (intro/moderate/hard)
- `provenance` object:
  - `url`
  - `license`
  - `date`
  - `author`
- `ai_generated` (true/false)

Example:

```json
{
  "title": "Python For Loop Basics",
  "content": "A for loop repeats over a sequence...\\n\\n```python\\nfor i in range(3):\\n    print(i)\\n```",
  "type": "tutorial",
  "concept_tags": ["loops", "variables"],
  "difficulty": "intro",
  "provenance": {
    "url": "https://example.com/python-loops",
    "license": "CC-BY-4.0",
    "date": "2026-02-08",
    "author": "Example Author"
  },
  "ai_generated": false
}
```

---

## 4. Licensing and Legal Scenarios (Always Check First)

Use this decision flow:

1. License clearly allows reuse (CC, MIT, public domain, your own content)?
   - Include, with full provenance metadata.
2. License unknown or unclear?
   - Do not include. Mark as `PENDING_LICENSE_REVIEW`.
3. Copyright all rights reserved and no reuse permission?
   - Do not include.
4. Paywalled/proprietary platform terms forbid redistribution?
   - Do not include.
5. Internal organization content?
   - Include only if written permission exists.

Never remove attribution requirements when license requires it.

---

## 5. AI-Generated Content Scenarios

If content is AI-generated:

- set `ai_generated: true`
- keep provenance of prompt source if available
- ensure human review before final benchmark snapshots

Two valid operational policies:

1. Include AI docs in development corpus.
2. Exclude AI docs from final benchmark snapshots.

If uncertain, use policy 1 for development and decide final policy before benchmark freeze.

---

## 6. Safety, Privacy, and Ethics Scenarios

Do not include content containing:

- personal data (emails, phone numbers, student IDs, private names)
- API keys, passwords, tokens
- harmful/malicious coding instructions
- unsafe exploit tutorials unrelated to learning goals

If content contains real user submissions:

- remove identifiers
- anonymize references
- keep only educationally necessary text/code

---

## 7. Source-Type Scenarios and What To Do

### Scenario A: Public tutorial article

- capture relevant sections
- preserve concept context
- include source URL and license

### Scenario B: Documentation page with navigation clutter

- copy only content body
- remove menus, sidebars, footers

### Scenario C: Forum post/Q&A

- include only if license and terms permit reuse
- prefer official docs/tutorials first

### Scenario D: Video lesson/transcript

- include only if transcript reuse rights allow
- cite source and date

### Scenario E: PDF/book excerpt

- include only if explicitly licensed for reuse
- otherwise do not include

### Scenario F: Classroom-created content

- include if your team owns rights
- mark author and version

### Scenario G: AI-generated draft

- set `ai_generated: true`
- perform human quality check

### Scenario H: Content with mixed topics

- keep only relevant sections
- tag only covered concepts

---

## 8. Quality Control Scenarios

Before finalizing each file, confirm:

- explanation is accurate
- code examples are coherent and beginner-appropriate
- concept tags match actual content
- type and difficulty are sensible
- provenance fields are complete

Reject or revise if:

- content is too short to be useful
- content is duplicated or near-duplicated
- concept tags are missing or wrong
- examples are broken/confusing

---

## 9. Tagging Rules (Simple and Consistent)

Use lowercase tags and keep stable vocabulary.

Recommended core tags:

- `variables`
- `loops`
- `conditionals`
- `functions`
- `debugging`
- `misconceptions`

Add secondary tags only when needed.

Do not invent new tag variants unless approved (example: avoid both `for-loop` and `loops`).

---

## 10. File Naming Rules

Use readable, stable filenames:

- `topic_short-title_source.json`

Examples:

- `loops_for-basics_official-docs.json`
- `functions_return-values_manual-authoring.json`

Avoid:

- spaces
- special characters
- generic names like `newfile.json`

---

## 11. Handling Duplicates and Near-Duplicates

If two sources explain the same thing almost identically:

- keep the clearer one
- keep the one with better licensing/provenance
- keep one high-quality version, not many clones

If both are useful but different:

- keep both, but distinguish via tags/type/difficulty

---

## 12. Language and Format Scenarios

If source is non-English:

- include only if project policy allows multilingual corpus
- otherwise translate with caution and mark provenance

If content has code fences:

- preserve Python code blocks if possible

If content includes screenshots/images:

- extract textual explanation/code if legally allowed
- do not include copyrighted images unless reuse allows it

---

## 13. What “Done” Looks Like for One Document

A document is ready when:

- required JSON fields are present
- licensing is clear and acceptable
- provenance is filled
- content is clean and relevant
- concept tags are correct
- no sensitive/private data is present

---

## 14. Team Workflow (No-Code)

1. Collector gathers candidate sources.
2. Verifier checks license and provenance.
3. Reviewer checks educational quality and tags.
4. Finalizer places JSON into `data/corpus_raw/`.
5. Lead approves for corpus build cycle.

If team is small, one person may do multiple roles, but all checks must still be done.

---

## 15. Review Checklist (Copy/Paste)

Use this per file:

- [ ] Required fields present (`title`, `content`, `type`, `concept_tags`)
- [ ] License allows reuse
- [ ] Source URL recorded
- [ ] Author/date recorded (if available)
- [ ] Tags match content
- [ ] Difficulty assigned (recommended)
- [ ] `ai_generated` set correctly
- [ ] No private/sensitive data
- [ ] No harmful/off-scope material
- [ ] File naming rule followed

---

## 16. Common Mistakes and Fixes

Mistake: Missing license info  
Fix: Hold file for review; do not ingest.

Mistake: Copying whole webpage including navigation  
Fix: Keep only educational content body.

Mistake: Random/inconsistent tags  
Fix: Use approved tag list.

Mistake: Duplicate content from many sites  
Fix: Keep best single version.

Mistake: AI content mixed in without label  
Fix: Add `ai_generated: true`.

Mistake: Broken JSON formatting  
Fix: Validate structure before saving.

---

## 17. Escalation Rules (When to Ask the Lead)

Escalate when:

- license is unclear
- source has legal restrictions
- content may include personal/sensitive data
- content quality is questionable
- unsure about tags/type/difficulty

Rule: when in doubt, pause and escalate.

---

## 18. Final Notes

- Corpus quality directly affects retrieval quality and benchmark fairness.
- Consistency and legal safety are more important than collecting content fast.
- It is better to include fewer high-quality, clearly licensed documents than many uncertain ones.
