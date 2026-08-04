"""Checks for the mu scale stage: the rescale identity, and the search.

Two independent things are verified here.

1. y(mu) = y(mu_ref)*(mu_ref/mu) is exact, not approximate. This is the claim
   that lets the scale stage cost one multiply instead of a resynthesis, and it
   would break silently if a future change made the mode grid depend on mu
   other than through T0/mu and D/mu.

2. fit_mu_scale's ternary search lands on the analytic minimizer for the losses
   that have one. Deliberately only for those: log(c*A + eps) is not
   log c + log(A + eps) once eps is comparable to the quiet bins, and log1p is
   not a shift in any regime, so the shift-based closed forms exist only in the
   eps -> 0 limit. test_log_closed_form_needs_zero_eps pins that down rather
   than leaving it as folklore -- it is the reason the production path runs one
   search for every loss instead of four closed forms.

Run:  python -m tests.test_mu_scale
"""

import math

import numpy as np
import torch

from src.cmaes.fit_7param_norm_es import FIXED_PLATE_PARAMS, NU
from src.ddsp.train_encoder import fit_mu_scale
from src.mu_optimization.ternary_mu import COMPOSITE_BOUNDS
from src.plate.SevenParamPlate import BatchedModalPlateTorch as SevenParamPlate

DEV = torch.device("cpu")
DTYPE = torch.float64


def weighted_median(values: torch.Tensor, weights: torch.Tensor) -> float:
    """argmin_c sum_i w_i |v_i - c|, the exact minimizer of a weighted L1."""
    v, order = torch.sort(values)
    w = weights[order]
    cum = torch.cumsum(w, dim=0)
    return float(v[int(torch.searchsorted(cum, 0.5 * cum[-1]))])


def recover_c(loss_fn, target, pred, mu_pred, iters=60):
    """Run the production search and report the scale it chose."""
    mu_fit = fit_mu_scale(loss_fn, target, pred, torch.tensor([mu_pred], dtype=DTYPE), iters)
    return mu_pred / float(mu_fit[0])


def _spectra(seed=0, n=512):
    """Nonnegative 'magnitudes'. The search is loss-agnostic, so feeding it
    magnitude vectors directly tests the search without an STFT in the way."""
    g = torch.Generator().manual_seed(seed)
    p = torch.rand((1, n), generator=g, dtype=DTYPE) + 1e-3
    c_true = 2.7
    t = c_true * p * (1.0 + 0.15 * torch.randn((1, n), generator=g, dtype=DTYPE))
    return t.clamp(min=1e-12), p, c_true


# ---------------------------------------------------------------------------
# 1. the rescale identity
# ---------------------------------------------------------------------------

def test_mu_rescale_is_exact():
    """Scaling E, rho and T0 together holds D/mu and T0/mu and moves only mu."""
    plate = SevenParamPlate(sample_rate=44100, device=DEV, dtype=torch.float32,
                            drop_sub_20hz_modes=False)
    base = {"E": 1.2e11, "rho": 5000.0, "h": 0.003, "Ly": 2.1, "T0": 40.0,
            "op_x": 0.63, "op_y": 0.77}

    def row(p):
        return [FIXED_PLATE_PARAMS["Lx"], p["Ly"], p["h"], p["T0"], p["rho"], p["E"],
                FIXED_PLATE_PARAMS["nu"], FIXED_PLATE_PARAMS["T60_DC"],
                FIXED_PLATE_PARAMS["T60_F1"], FIXED_PLATE_PARAMS["loss_F1"],
                FIXED_PLATE_PARAMS["fp_x"], FIXED_PLATE_PARAMS["fp_y"],
                p["op_x"], p["op_y"]]

    for c in (0.37, 1.0, 4.9):
        scaled = dict(base, E=base["E"] * c, rho=base["rho"] * c, T0=base["T0"] * c)
        mu, mu2 = base["rho"] * base["h"], scaled["rho"] * scaled["h"]
        d = base["E"] * base["h"] ** 3 / (12 * (1 - NU ** 2))
        d2 = scaled["E"] * scaled["h"] ** 3 / (12 * (1 - NU ** 2))
        assert abs(mu2 / mu - c) < 1e-9, "mu did not scale by c"
        assert abs(d2 / mu2 - d / mu) < 1e-6 * (d / mu), "D/mu moved"
        assert abs(scaled["T0"] / mu2 - base["T0"] / mu) < 1e-6 * (base["T0"] / mu), "T0/mu moved"

        params = torch.tensor([row(base), row(scaled)], dtype=torch.float32, device=DEV)
        with torch.no_grad():
            y = plate(params, duration=0.05, normalize=False)
        got, want = y[1], y[0] / c
        err = float((got - want).abs().max() / y[0].abs().max().clamp(min=1e-30))
        assert err < 1e-5, f"rescale identity broke at c={c}: rel err {err:.3e}"
    print("test_mu_rescale_is_exact                 OK")


# ---------------------------------------------------------------------------
# 2. the search against analytic minimizers
# ---------------------------------------------------------------------------

def test_linear_l1_matches_weighted_median():
    t, p, _ = _spectra()

    def loss(a, b):
        return (a - b).abs().mean(dim=-1)

    want = weighted_median(t[0] / p[0], p[0])
    got = recover_c(loss, t, p, mu_pred=10.0)
    assert abs(got - want) / want < 1e-4, f"linear L1: got {got:.6f} want {want:.6f}"
    print(f"test_linear_l1_matches_weighted_median   OK  (c={got:.4f})")


def test_pow_matches_weighted_median_of_powers():
    """(c*A)^g = c^g * A^g, so compression commutes with scaling exactly."""
    g = 0.3
    t, p, _ = _spectra(seed=1)

    def loss(a, b):
        return (a.pow(g) - b.pow(g)).abs().mean(dim=-1)

    d = weighted_median((t[0] / p[0]).pow(g), p[0].pow(g))
    want = d ** (1.0 / g)
    got = recover_c(loss, t, p, mu_pred=10.0)
    assert abs(got - want) / want < 1e-3, f"pow: got {got:.6f} want {want:.6f}"
    print(f"test_pow_matches_weighted_median         OK  (c={got:.4f})")


def test_sc_matches_least_squares():
    """Spectral convergence is a Frobenius norm of an affine function of c."""
    t, p, _ = _spectra(seed=2)

    def loss(a, b):
        return (a - b).pow(2).sum(dim=-1).sqrt() / a.pow(2).sum(dim=-1).sqrt()

    want = float((t[0] * p[0]).sum() / (p[0] * p[0]).sum())
    got = recover_c(loss, t, p, mu_pred=10.0)
    assert abs(got - want) / want < 1e-4, f"SC: got {got:.6f} want {want:.6f}"
    print(f"test_sc_matches_least_squares            OK  (c={got:.4f})")


def test_log_closed_form_needs_zero_eps():
    """The shift identity holds at eps=0 and fails at the production eps.

    losses.py uses log(x + 1e-7). The quiet bins of these IRs sit within a
    factor of a few of that, which is exactly where log(c*A + eps) departs from
    log c + log(A + eps) -- so the median-of-log-ratios closed form is wrong for
    the loss as implemented, while remaining right in the limit.
    """
    t, p, _ = _spectra(seed=3)
    want = float(torch.median(torch.log(t[0]) - torch.log(p[0])).exp())

    def loss_exact(a, b):
        return (torch.log(a) - torch.log(b)).abs().mean(dim=-1)

    got = recover_c(loss_exact, t, p, mu_pred=10.0)
    assert abs(got - want) / want < 1e-3, f"log eps=0: got {got:.6f} want {want:.6f}"

    # Now push the data down so eps=1e-7 sits inside it, as it does for the
    # quiet IRs, and the same closed form should no longer be the minimizer.
    eps = 1e-7
    ts, ps = t * 1e-7, p * 1e-7

    def loss_eps(a, b):
        return (torch.log(a + eps) - torch.log(b + eps)).abs().mean(dim=-1)

    got_eps = recover_c(loss_eps, ts, ps, mu_pred=10.0)
    assert abs(got_eps - want) / want > 1e-2, (
        "log(x+eps) happened to agree with the eps=0 closed form; if this ever "
        "holds, the premise for running a search instead of a closed form is gone"
    )
    print(f"test_log_closed_form_needs_zero_eps      OK  "
          f"(eps=0 {got:.4f}, eps=1e-7 {got_eps:.4f})")


def test_search_respects_mu_bounds():
    """The bracket must not let the fit leave the physical box."""
    t, p, _ = _spectra(seed=4)

    def loss(a, b):
        return (a - b).abs().mean(dim=-1)

    lo, hi = COMPOSITE_BOUNDS["mu"]
    for mu_pred in (lo * 1.001, math.sqrt(lo * hi), hi * 0.999):
        mu_fit = fit_mu_scale(loss, t, p, torch.tensor([mu_pred], dtype=DTYPE))
        v = float(mu_fit[0])
        assert lo - 1e-9 <= v <= hi + 1e-9, f"mu escaped the box: {v:.4f}"
    print("test_search_respects_mu_bounds           OK")


if __name__ == "__main__":
    test_mu_rescale_is_exact()
    test_linear_l1_matches_weighted_median()
    test_pow_matches_weighted_median_of_powers()
    test_sc_matches_least_squares()
    test_log_closed_form_needs_zero_eps()
    test_search_respects_mu_bounds()
    print("\nall OK")
