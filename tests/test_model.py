"""Tests for the looped GPT model."""

import pytest
import torch

from model import GPT, GPTConfig


def _small_config(**overrides) -> GPTConfig:
    """Create a small config for fast tests."""
    defaults = dict(
        block_size=32,
        vocab_size=64,
        n_prelude=0,
        n_block=2,
        n_coda=0,
        n_loop=1,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
        input_injection="passthrough",
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


class TestForwardShapes:
    """Test that forward pass produces correct output shapes."""

    def test_standard_config(self):
        """n_loop=1, passthrough — equivalent to vanilla GPT."""
        config = _small_config()
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (2, 16))
        targets = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss = model(idx, targets)
        assert logits.shape == (2, 16, config.vocab_size)
        assert loss is not None

    def test_inference_last_token_only(self):
        """Without targets, logits should be for last position only."""
        config = _small_config()
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss = model(idx)
        assert logits.shape == (2, 1, config.vocab_size)
        assert loss is None

    def test_full_loop(self):
        """Full loop: all layers recur, no prelude/coda."""
        config = _small_config(n_block=3, n_loop=4, input_injection="inject")
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (2, 16))
        targets = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss = model(idx, targets)
        assert logits.shape == (2, 16, config.vocab_size)
        assert loss is not None

    def test_prelude_recur_coda(self):
        """Prelude/recur/coda architecture."""
        config = _small_config(n_prelude=1, n_block=2, n_coda=1, n_loop=3, input_injection="inject")
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (2, 16))
        targets = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss = model(idx, targets)
        assert logits.shape == (2, 16, config.vocab_size)
        assert loss is not None


class TestInjectionModes:
    """Test all three injection modes."""

    @pytest.mark.parametrize("mode", ["inject", "inject_random", "passthrough"])
    def test_injection_mode_runs(self, mode: str):
        config = _small_config(n_loop=3, input_injection=mode)
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (2, 8))
        targets = torch.randint(0, config.vocab_size, (2, 8))
        logits, loss = model(idx, targets)
        assert logits.shape == (2, 8, config.vocab_size)
        assert loss.isfinite()

    def test_inject_init_identity(self):
        """inject(cat(e, zeros)) should approximate e at initialization."""
        config = _small_config(n_loop=2, input_injection="inject")
        model = GPT(config)
        e = torch.randn(1, 4, config.n_embd)
        zeros = torch.zeros_like(e)
        result = model.inject(torch.cat([e, zeros], dim=-1))
        torch.testing.assert_close(result, e, atol=1e-5, rtol=1e-5)


class TestGradients:
    """Test that gradients flow correctly."""

    def test_gradients_standard(self):
        """n_loop=1 passthrough should produce valid gradients."""
        config = _small_config()
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (2, 8))
        targets = torch.randint(0, config.vocab_size, (2, 8))
        _, loss = model(idx, targets)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_gradients_looped_inject(self):
        """Looped model with inject should produce valid gradients."""
        config = _small_config(n_prelude=1, n_block=2, n_coda=1, n_loop=3, input_injection="inject")
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (2, 8))
        targets = torch.randint(0, config.vocab_size, (2, 8))
        _, loss = model(idx, targets)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_gradients_inject_random(self):
        """inject_random mode should produce valid gradients."""
        config = _small_config(n_prelude=1, n_block=2, n_coda=1, n_loop=3, input_injection="inject_random")
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (2, 8))
        targets = torch.randint(0, config.vocab_size, (2, 8))
        _, loss = model(idx, targets)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


class TestTruncatedBPTT:
    """Test that truncated BPTT correctly detaches gradients."""

    def test_bptt_detaches_early_iterations(self):
        """With bptt_k=1 and n_loop=3, only last iteration should contribute gradients."""
        config = _small_config(n_block=2, n_loop=3, bptt_k=1, input_injection="passthrough")
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (1, 4))
        targets = torch.randint(0, config.vocab_size, (1, 4))
        _, loss = model(idx, targets)
        loss.backward()
        # All recur params should still have gradients (from the last iteration)
        for name, p in model.named_parameters():
            if "recur" in name and p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_bptt_none_full_backprop(self):
        """bptt_k=None should be equivalent to full backprop."""
        torch.manual_seed(42)
        config = _small_config(n_block=2, n_loop=3, bptt_k=None, input_injection="passthrough")
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (1, 4))
        targets = torch.randint(0, config.vocab_size, (1, 4))
        _, loss = model(idx, targets)
        loss.backward()
        # Capture full-backprop gradients
        full_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        # Now compare with bptt_k=n_loop (should be identical)
        model.zero_grad()
        model.config.bptt_k = 3  # same as n_loop
        _, loss = model(idx, targets)
        loss.backward()
        bptt_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        for name in full_grads:
            torch.testing.assert_close(full_grads[name], bptt_grads[name])

    def test_bptt_reduces_gradient_magnitude(self):
        """bptt_k=1 should produce smaller gradients than full backprop for recur params."""
        torch.manual_seed(42)
        config = _small_config(n_block=2, n_loop=4, bptt_k=None, input_injection="passthrough")
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (1, 4))
        targets = torch.randint(0, config.vocab_size, (1, 4))
        _, loss = model(idx, targets)
        loss.backward()
        full_grad_norm = (
            sum(p.grad.norm().item() ** 2 for n, p in model.named_parameters() if "recur" in n and p.grad is not None) ** 0.5
        )

        model.zero_grad()
        model.config.bptt_k = 1
        _, loss = model(idx, targets)
        loss.backward()
        truncated_grad_norm = (
            sum(p.grad.norm().item() ** 2 for n, p in model.named_parameters() if "recur" in n and p.grad is not None) ** 0.5
        )

        assert truncated_grad_norm < full_grad_norm, (
            f"Truncated gradient norm ({truncated_grad_norm:.4f}) should be smaller "
            f"than full gradient norm ({full_grad_norm:.4f})"
        )


class TestGenerate:
    """Test autoregressive generation."""

    def test_generate_produces_tokens(self):
        config = _small_config(n_loop=2, input_injection="inject")
        model = GPT(config)
        model.eval()
        idx = torch.randint(0, config.vocab_size, (1, 4))
        result = model.generate(idx, max_new_tokens=5)
        assert result.shape == (1, 9)  # 4 prompt + 5 generated


class TestUtilities:
    """Test utility methods."""

    def test_get_num_params(self):
        config = _small_config()
        model = GPT(config)
        n_params = model.get_num_params()
        assert n_params > 0

    def test_crop_block_size(self):
        config = _small_config(block_size=32)
        model = GPT(config)
        model.crop_block_size(16)
        assert model.config.block_size == 16

    def test_configure_optimizers(self):
        config = _small_config()
        model = GPT(config)
        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type="cpu")
        assert optimizer is not None
