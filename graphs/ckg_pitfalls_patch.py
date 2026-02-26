"""
One-time script: injects pedagogical pitfall annotations into CKG concept nodes.

Run from benchmark/ directory:
    py graphs/ckg_pitfalls_patch.py
"""
import json
from pathlib import Path

# Pitfalls keyed by exact CKG concept node label.
# Covers the 4 core CORE_TOPICS and their close relatives.
PITFALLS: dict[str, list[str]] = {
    # ── Core topics ─────────────────────────────────────────────────────────
    "loops": [
        "Modifying a list while iterating over it causes unexpected skips or double-processing.",
        "Forgetting to update the loop variable in a `while` loop creates an infinite loop.",
        "`range(n)` stops at n-1, not n — a common off-by-one source.",
        "Using `for i in range(len(lst))` is less readable than `for item in lst` or `enumerate(lst)`.",
    ],
    "variables": [
        "Assigning `b = a` for a mutable object (list, dict) shares both names to the same object — mutation via `b` affects `a`.",
        "Using a variable before assignment raises `NameError` at runtime.",
        "Shadowing a built-in name (e.g., `list = [1, 2]`) silently disables that built-in for the rest of the scope.",
        "Augmented assignment (`x += 1`) requires `x` to already be defined; otherwise `UnboundLocalError`.",
    ],
    "functions": [
        "Using a mutable default argument (e.g., `def f(x=[])`) shares state across all calls — the same list persists between invocations.",
        "Forgetting `return` makes the function implicitly return `None`, which surprises callers expecting a value.",
        "Confusing positional and keyword argument order causes `TypeError` at the call site.",
        "Local assignment inside a function makes Python treat the name as purely local — reading it before the assignment raises `UnboundLocalError`.",
    ],
    "conditionals": [
        "Testing `if x == True` instead of `if x` fails for non-boolean truthy values like non-empty strings.",
        "Using `is` for value equality (e.g., `if x is 5`) is unreliable; use `==` for values, `is` only for identity checks like `is None`.",
        "Forgetting that `elif` short-circuits: once a branch matches, the rest are skipped even if also true.",
        "Deeply nested `if/elif/else` can be flattened with early returns (guard clauses) for readability.",
    ],
    # ── Loop sub-concepts ────────────────────────────────────────────────────
    "for": [
        "`for` loop variable leaks into the enclosing scope after the loop finishes.",
        "`range(n)` generates integers 0 to n-1; forgetting this causes off-by-one in index-based loops.",
        "Modifying the iterable inside a `for` loop produces unpredictable results — iterate a copy instead.",
    ],
    "while": [
        "Forgetting to advance the loop condition variable creates an infinite loop.",
        "`while True` loops require an explicit `break` or `return` to terminate — missing one hangs the program.",
        "Using `while len(lst) > 0` is equivalent but less Pythonic than `while lst`.",
    ],
    "infinite-loop": [
        "A `while` loop that never modifies its condition variable runs forever.",
        "Input validation loops (`while True: ... break`) must ensure all error paths eventually reach `break`.",
    ],
    "iteration": [
        "Calling `iter()` on the same object twice gives two independent iterators only for sequences; for file objects or generators a second call reuses the exhausted iterator.",
        "Using `range(len(lst))` when `enumerate(lst)` is cleaner and avoids index errors.",
        "Modifying a collection while iterating it raises `RuntimeError` (for dicts) or silently skips items (for lists).",
    ],
    # ── Variable sub-concepts ────────────────────────────────────────────────
    "scope": [
        "Assigning to a name anywhere inside a function makes Python treat it as local throughout that function — reading it before the assignment raises `UnboundLocalError`, not the global value.",
        "`global x` is needed only when you want to *assign* to a global from inside a function; reading a global requires no declaration.",
        "Nested functions see their enclosing scope at call time, not at definition time (late binding).",
    ],
    "late-binding": [
        "Closures capture variables by reference, not by value — a loop closure using `lambda: i` will use the final value of `i` for all lambdas.",
        "Fix late binding with a default argument: `lambda i=i: i` captures the current value at definition time.",
    ],
    "aliasing": [
        "Two names pointing to the same mutable object means mutations through either name affect both.",
        "Use `copy.copy()` for a shallow copy or `copy.deepcopy()` for a fully independent clone.",
    ],
    "mutability": [
        "Tuple elements that are themselves mutable objects (e.g., a list inside a tuple) can still be mutated.",
        "`b = a` for a list makes `b` an alias, not a copy. Use `b = a[:]` or `list(a)` for a shallow copy.",
    ],
    "mutable-defaults": [
        "Default argument values are evaluated *once* at function definition — a mutable default (list, dict) is shared across all calls.",
        "Canonical fix: use `None` as the default and assign inside the body: `if x is None: x = []`.",
    ],
    # ── Function sub-concepts ────────────────────────────────────────────────
    "closures": [
        "A closure captures the *variable*, not its value at capture time — loop closures all see the final loop value.",
        "Returning an inner function that assigns to an enclosing variable requires the `nonlocal` keyword; without it, the assignment creates a new local.",
    ],
    "parameters": [
        "Positional arguments must come before keyword arguments in a function call.",
        "A parameter with a mutable default (list, dict, set) is shared across all calls where the default is used.",
        "`*args` collects extra positional arguments as a tuple; `**kwargs` collects extra keyword arguments as a dict.",
    ],
    "recursion": [
        "Forgetting a base case causes unbounded recursion and `RecursionError: maximum recursion depth exceeded`.",
        "Python's default recursion limit is 1000; deep recursion should be converted to iteration.",
        "Each recursive call adds a stack frame — recursive solutions to large inputs may be slower and more memory-intensive than iterative ones.",
    ],
    # ── Conditional sub-concepts ─────────────────────────────────────────────
    "truthiness": [
        "Empty containers (`[]`, `{}`, `set()`, `\"\"`) are falsy; a single-element container is truthy.",
        "`0`, `0.0`, `None`, and `False` are falsy; all other numbers and objects are truthy.",
        "Testing `if lst != []` is equivalent but less Pythonic than `if lst`.",
    ],
    "boolean-logic": [
        "`and` / `or` do *not* return `True`/`False` — they return one of their operands (`or` returns the first truthy, `and` returns the first falsy or the last).",
        "Short-circuit evaluation means the right operand of `and` is not evaluated if the left is falsy.",
    ],
    # ── Misc high-value ─────────────────────────────────────────────────────
    "comprehensions": [
        "A list comprehension with a side-effecting expression (e.g., `[lst.append(x) for x in ...]`) returns a list of `None`s and the side effects are the real output — use a plain `for` loop instead.",
        "Nested comprehensions `[x for row in grid for x in row]` read left-to-right, matching the nesting order of equivalent `for` loops.",
    ],
    "generators": [
        "A generator can only be iterated once; once exhausted, re-iterating it yields nothing.",
        "Calling `next()` on an exhausted generator raises `StopIteration`.",
        "Generators are lazy — values are computed on demand; converting to a list forces eager evaluation.",
    ],
    "decorators": [
        "Stacking decorators applies them bottom-up: `@A\\n@B\\ndef f` is equivalent to `f = A(B(f))`.",
        "Forgetting `@functools.wraps(fn)` inside a wrapper hides the original function's `__name__` and `__doc__`.",
    ],
}


def patch(ckg_path: Path = Path("graphs/ckg.json")) -> None:
    graph = json.loads(ckg_path.read_text(encoding="utf-8"))
    patched = 0
    for node in graph["nodes"]:
        if node.get("type") == "concept":
            label = node.get("label", "")
            if label in PITFALLS:
                node["pitfalls"] = PITFALLS[label]
                patched += 1
    ckg_path.write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")
    total_concepts = sum(1 for n in graph["nodes"] if n.get("type") == "concept")
    print(f"Patched {patched} / {total_concepts} concept nodes with pitfall annotations.")
    print("Done.")


if __name__ == "__main__":
    patch()
