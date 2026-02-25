"""Boolean satisfiability (random 3-SAT) tasks.

Generates random k-SAT instances near the phase transition (~4.27 clauses
per variable for 3-SAT) where the problem is hardest. Uses exhaustive
search to determine satisfiability (feasible for the small instances we
train on, n_vars <= 16).

Format: (a|~b|c)&(~a|b|~c)=1  (SAT) or =0 (UNSAT)
Variables are single lowercase letters a-z.
"""

import numpy as np

type Clause = list[tuple[int, bool]]  # list of (variable_index, is_positive)


def sat(rng: np.random.Generator, level: int) -> str:
    """Random 3-SAT instance. Level = number of variables (2-16)."""
    n_vars = max(2, min(level, 16))
    k = min(3, n_vars)  # clause width (3-SAT, or smaller if few vars)

    # Number of clauses near phase transition for interesting difficulty
    n_clauses = max(1, round(4.27 * n_vars + rng.standard_normal() * 0.5))

    clauses: list[Clause] = []
    for _ in range(n_clauses):
        var_idxs = rng.choice(n_vars, size=k, replace=False).tolist()
        signs = rng.choice([True, False], size=k).tolist()
        clauses.append(list(zip(var_idxs, signs)))

    is_sat = _check_sat(clauses, n_vars)
    formula = _format_clauses(clauses)
    return f"{formula}={'1' if is_sat else '0'}"


def _check_sat(clauses: list[Clause], n_vars: int) -> bool:
    """Exhaustive SAT check — try all 2^n_vars assignments."""
    return any(_satisfies(clauses, assignment) for assignment in range(1 << n_vars))


def _satisfies(clauses: list[Clause], assignment: int) -> bool:
    """Check if a single assignment satisfies all clauses."""
    for clause in clauses:
        clause_sat = False
        for var, positive in clause:
            val = bool(assignment & (1 << var))
            if val == positive:
                clause_sat = True
                break
        if not clause_sat:
            return False
    return True


def _format_clauses(clauses: list[Clause]) -> str:
    """Format clauses as (a|~b|c)&(~a|b|~c)."""
    parts = []
    for clause in clauses:
        lits = []
        for var, positive in clause:
            name = chr(ord("a") + var)
            lits.append(name if positive else f"~{name}")
        parts.append(f"({'|'.join(lits)})")
    return "&".join(parts)
