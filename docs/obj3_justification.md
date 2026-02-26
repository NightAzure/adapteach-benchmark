# Why Objective 3 Belongs in This Paper
### A Structural Justification for the Three-Objective Design

---

## The Concern

Objectives 1 and 2 evaluate the **RAG pipeline** — which retrieval configuration
surfaces the most relevant context, and which generates the most faithful, relevant
responses. Objective 3 evaluates **student learning outcomes** under adaptive vs.
static content delivery. On the surface, these feel like different studies.

This document explains why they are not.

---

## The Argument: A System Needs Both Halves to Work

Cognitive Code is not just a retrieval pipeline. It is a **full adaptive learning
system** with two distinct functional components:

| Component | What it does | Validated by |
|---|---|---|
| RAG pipeline (retrieval + generation) | Produces educationally appropriate artifacts from a structured knowledge base | Obj 1 + Obj 2 |
| Adaptive delivery engine (BKT + ZPD routing) | Sequences those artifacts based on learner mastery state | Obj 3 |

A system with excellent retrieval but poor adaptive sequencing would fail students.
A system with excellent sequencing but poor retrieval would generate garbage artifacts.
**Both halves must be validated for the system to be considered evaluated.**

Obj 3 is not a jump — it is the second half of a complete system evaluation.

---

## The Standard Pattern in Educational Technology Research

This three-phase structure is the established norm in ITS and adaptive learning
research, not an anomaly:

- **VanLehn (2011)** — meta-analysis of 62 ITS studies — structures evaluation
  identically: technical system description → content quality validation →
  learning outcomes.
- **Papakostas et al. (2025)** — BKT-based AR adaptive system — validates
  technical architecture first, then runs n=30 learning efficacy study.
- **Thilakaratne et al. (2025)** — mastery learning adaptive platform —
  technical benchmark followed by n=28 participant study.

The pattern is: *build → validate content → validate impact*. Obj 1+2 are "build
and validate content." Obj 3 is "validate impact." This is the standard.

---

## The Logical Chain

The three objectives are sequentially dependent, not parallel:

```
Obj 1: The pipeline retrieves high-quality, relevant context.
         ↓
       Therefore the artifacts it generates are grounded in correct material.
         ↓
Obj 2: The generated artifacts are faithful and relevant to the query.
         ↓
       Therefore we have high-quality content to deliver to students.
         ↓
Obj 3: Delivering that content adaptively (BKT) produces greater learning
       gains than delivering it randomly (static control).
```

Without Obj 1+2, we cannot trust the artifacts used in Obj 3.
Without Obj 3, we cannot claim the system has educational value.

---

## Why It Is NOT an IT/IS Study

The distinction is in what Obj 3 measures:

- **IT/IS framing** would evaluate system performance metrics —
  uptime, response latency, database throughput, UI usability scores.
- **This study** measures **normalized learning gain** (Hake, 1998) —
  a cognitive science outcome metric used in physics, CS1, and STEM education
  research. It is not a system metric. It is a learning science metric.

Debugging proficiency as a secondary outcome follows the same logic:
it measures a transferable cognitive skill, not a product feature.

---

## The "Huge Jump" Reframed

Yes, Obj 3 requires a different methodology (quasi-experimental, human subjects,
pre/post design). But the research object is the same: **Cognitive Code**.

Obj 1+2 ask: *"Is the system technically sound?"*
Obj 3 asks: *"Does the technically sound system actually help students learn?"*

A thesis that answered only Obj 1+2 would be a strong systems paper but a weak
education paper. A thesis that answered only Obj 3 would have no way to attribute
learning gains to the RAG architecture. Together, they establish that the system
works — and that it matters.

---

## Reassurance: The Committee Agreed

The panel raised concerns about Obj 1 and Obj 2 sounding IT-ish. They did not
flag Obj 3. They also explicitly separated usability evaluation and student
perception (a different type of question) into a second paper — which means they
implicitly validated that Obj 3's scope (learning gains, not usability) is the
right boundary for this paper.

**The jump you're feeling is a methodology shift, not a logic gap. Those are
different things.**

---

*Last updated: 2026-02-27*
