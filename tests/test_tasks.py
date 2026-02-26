"""Tests for procedurally generated reasoning tasks."""

import numpy as np
import pytest

from tasks import (
    BOS,
    CHARS,
    EOS,
    PAD,
    SEP,
    STOI,
    VOCAB_SIZE,
    TaskMix,
    decode,
    encode,
)
from tasks.arithmetic import addition, addition_rev, multiplication, subtraction
from tasks.graph import _bfs, graph_path
from tasks.grid import grid_transform
from tasks.maze import _bfs_rooms, _generate_maze, maze_path
from tasks.sat import _check_sat, sat


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


# --- Vocabulary ---


class TestVocabulary:
    def test_special_tokens(self):
        assert PAD == 0
        assert BOS == 1
        assert EOS == 2

    def test_encode_decode_roundtrip(self):
        text = "123+456=579"
        assert decode(encode(text)) == text

    def test_all_chars_encoded(self):
        for ch in CHARS:
            assert ch in STOI, f"Character {ch!r} missing from STOI"

    def test_no_token_collisions(self):
        ids = list(STOI.values()) + [PAD, BOS, EOS]
        assert len(ids) == len(set(ids))

    def test_vocab_size_covers_all_tokens(self):
        assert max(STOI.values()) < VOCAB_SIZE

    def test_sep_token(self):
        assert STOI["="] == SEP


# --- Arithmetic ---


class TestArithmetic:
    def test_addition_format(self, rng):
        for level in range(1, 6):
            text = addition(rng, level)
            assert "+" in text
            assert "=" in text
            q, a = text.split("=")
            left, right = q.split("+")
            assert int(left) + int(right) == int(a)

    def test_addition_rev_format(self, rng):
        for level in range(1, 6):
            text = addition_rev(rng, level)
            q, a = text.split("=")
            left, right = q.split("+")
            # Reverse back and check
            assert int(left[::-1]) + int(right[::-1]) == int(a[::-1])

    def test_subtraction_non_negative(self, rng):
        for level in range(1, 6):
            text = subtraction(rng, level)
            q, a = text.split("=")
            left, right = q.split("-")
            result = int(a)
            assert result >= 0
            assert int(left) - int(right) == result

    def test_multiplication_format(self, rng):
        for level in range(1, 4):
            text = multiplication(rng, level)
            q, a = text.split("=")
            left, right = q.split("*")
            assert int(left) * int(right) == int(a)

    def test_level_controls_digits(self, rng):
        for level in [1, 3, 5]:
            text = addition(rng, level)
            q, _a = text.split("=")
            left, right = q.split("+")
            if level > 1:
                assert len(left) == level
                assert len(right) == level


# --- SAT ---


class TestSAT:
    def test_format(self, rng):
        text = sat(rng, 3)
        assert "=" in text
        q, a = text.split("=")
        assert a in ("0", "1")
        assert "(" in q and ")" in q

    def test_check_sat_simple(self):
        # (a) — satisfiable (a=True)
        assert _check_sat([[(0, True)]], 1) is True
        # (a) & (~a) — unsatisfiable
        assert _check_sat([[(0, True)], [(0, False)]], 1) is False

    def test_answer_matches_exhaustive_check(self, rng):
        """Verify the SAT label is correct by re-checking."""
        for _ in range(20):
            text = sat(rng, 3)
            # Parse answer
            q, a = text.split("=")
            claimed_sat = a == "1"

            # Re-parse and check: count variables by scanning for letters
            var_letters = sorted({ch for ch in q if ch.isalpha()})
            n_vars = len(var_letters)
            var_map = {ch: i for i, ch in enumerate(var_letters)}

            # Parse clauses
            clauses = []
            for clause_str in q.split("&"):
                clause_str = clause_str.strip("()")
                lits = clause_str.split("|")
                clause = []
                for lit in lits:
                    if lit.startswith("~"):
                        clause.append((var_map[lit[1]], False))
                    else:
                        clause.append((var_map[lit[0]], True))
                clauses.append(clause)

            assert _check_sat(clauses, n_vars) == claimed_sat

    def test_levels_produce_different_sizes(self, rng):
        small = sat(rng, 2)
        large = sat(rng, 6)
        # Larger level should generally produce longer formulas
        assert len(large) > len(small)


# --- Grid ---


class TestGrid:
    def test_format(self, rng):
        text = grid_transform(rng, 3)
        assert "=" in text
        code = text[0]
        assert code in "chvrsq"

    def test_identity_preserves_grid(self):
        """Identity transform should return the same grid."""
        # Force identity by seeding
        rng = np.random.default_rng(0)
        for _ in range(50):
            text = grid_transform(rng, 3)
            if text.startswith("c"):
                q_grid = text[1:].split("=")[0]
                a_grid = text.split("=")[1]
                assert q_grid == a_grid
                return
        pytest.skip("No identity transform generated in 50 tries")

    def test_hflip(self):
        """Check horizontal flip correctness."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            text = grid_transform(rng, 3)
            if text.startswith("h"):
                q_rows = text[1:].split("=")[0].split("\n")
                a_rows = text.split("=")[1].split("\n")
                for q_row, a_row in zip(q_rows, a_rows):
                    assert q_row[::-1] == a_row
                return
        pytest.skip("No hflip generated in 100 tries")

    def test_level_controls_size(self, rng):
        for level in [2, 5]:
            text = grid_transform(rng, level)
            # Grid after the code character, before "="
            grid_str = text[1:].split("=")[0]
            rows = grid_str.split("\n")
            assert len(rows) == level
            assert all(len(r) == level for r in rows)


# --- Graph ---


class TestGraph:
    def test_format(self, rng):
        text = graph_path(rng, 5)
        assert "=" in text
        assert ";" in text
        assert ">" in text

    def test_path_correctness(self, rng):
        """Verify the returned path actually exists in the graph."""
        for _ in range(30):
            text = graph_path(rng, 5)
            parts, answer = text.split("=")
            edge_str, query = parts.split(";")
            src, tgt = query.split(">")

            if answer == "n":
                continue

            # Build adjacency
            adj: dict[str, set[str]] = {}
            for edge in edge_str.split(","):
                a, b = edge[0], edge[1]
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)

            # Verify path
            path = list(answer)
            assert path[0] == src
            assert path[-1] == tgt
            for i in range(len(path) - 1):
                assert path[i + 1] in adj.get(path[i], set()), f"No edge {path[i]}-{path[i + 1]}"

    def test_bfs_finds_shortest(self):
        adj = {"a": ["b", "c"], "b": ["a", "d"], "c": ["a", "d"], "d": ["b", "c"]}
        path = _bfs(adj, "a", "d")
        assert path is not None
        assert len(path) == 3  # a -> b -> d or a -> c -> d

    def test_bfs_no_path(self):
        adj = {"a": ["b"], "b": ["a"], "c": []}
        assert _bfs(adj, "a", "c") is None


# --- Maze ---


class TestMaze:
    def test_format(self, rng):
        text = maze_path(rng, 3)
        assert "=" in text
        q, a = text.split("=")
        assert "s" in q
        assert "e" in q
        assert all(ch in "udlrn" for ch in a)

    def test_maze_always_solvable(self, rng):
        """DFS mazes are perfect — every room reachable."""
        for _ in range(20):
            text = maze_path(rng, 4)
            _q, a = text.split("=")
            assert a != "n", "Perfect maze should always have a path"

    def test_path_reaches_end(self, rng):
        """Simulate the path and verify it reaches the end room."""
        for _ in range(20):
            size = 3
            maze = _generate_maze(rng, size, size)
            maze[1, 1] = "s"
            maze[2 * size - 1, 2 * size - 1] = "e"
            path = _bfs_rooms(maze, size, size, (0, 0), (size - 1, size - 1))
            assert path is not None

            # Simulate
            r, c = 0, 0
            moves = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}
            for d in path:
                dr, dc = moves[d]
                r, c = r + dr, c + dc
            assert (r, c) == (size - 1, size - 1)

    def test_level_controls_size(self, rng):
        for level in [2, 5]:
            text = maze_path(rng, level)
            q = text.split("=")[0]
            rows = q.split("\n")
            expected = 2 * level + 1
            assert len(rows) == expected
            assert all(len(r) == expected for r in rows)


# --- TaskMix ---


class TestTaskMix:
    def test_from_spec_single(self):
        mix = TaskMix.from_spec("addition:3")
        assert len(mix.entries) == 1

    def test_from_spec_multi(self):
        mix = TaskMix.from_spec("addition:1-5 sat:2-4 maze:3")
        assert len(mix.entries) == 3

    def test_from_spec_unknown_task(self):
        with pytest.raises(ValueError, match="Unknown task"):
            TaskMix.from_spec("nonexistent:1")

    def test_get_batch_shapes(self):
        mix = TaskMix.from_spec("addition:1-3")
        x, y = mix.get_batch(batch_size=8, block_size=64)
        assert x.shape == (8, 64)
        assert y.shape == (8, 64)
        assert x.dtype == y.dtype == torch.long

    def test_get_batch_masking(self):
        """Verify question tokens are masked (-1) and answer tokens are not."""
        mix = TaskMix.from_spec("addition:2")
        x, y = mix.get_batch(batch_size=16, block_size=64)

        for b in range(16):
            # Find where actual content ends (before padding)
            content_mask = x[b] != PAD
            if not content_mask.any():
                continue

            # y should have -1 for question portion and PAD portion
            # and actual token ids for the answer portion
            answer_tokens = y[b][y[b] != -1]
            assert len(answer_tokens) > 0, "Should have at least one answer token"

            # Last answer token should be EOS
            assert answer_tokens[-1].item() == EOS

    def test_multi_task_generates_different_tasks(self):
        mix = TaskMix.from_spec("addition:2 sat:3 maze:2")
        texts = set()
        for _ in range(100):
            full, _ = mix.generate_one(256)
            texts.add(tuple(full[:5]))  # first few tokens as fingerprint
        # Should have variety
        assert len(texts) > 3

    def test_get_batch_no_out_of_bounds(self):
        """Ensure all token ids are valid vocab indices."""
        mix = TaskMix.from_spec("addition:1-5 sat:2-4 grid:2-4 graph:3-5 maze:2-4")
        x, y = mix.get_batch(batch_size=32, block_size=128)
        assert x.min() >= 0
        assert x.max() < VOCAB_SIZE
        # y can have -1 (masked) but otherwise valid
        valid_y = y[y != -1]
        assert valid_y.min() >= 0
        assert valid_y.max() < VOCAB_SIZE


import torch  # noqa: E402 (needed for TestTaskMix)
