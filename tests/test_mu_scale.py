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
   eps -> 0 limit. The two log tests pin that down rather than leaving it as
   folklore -- it is the reason the production path runs one search for every
   loss instead of four closed forms.

   Worth knowing while reading them: when the prediction is already close to a
   scalar multiple of the target, every one of these rules minimizes at nearly
   the same c, because far below the knee log(t+eps) - log(cp+eps) ~ (t - cp)/eps,
   which is the linear objective over a constant. The rules diverge where the
   shape fit is poor -- the tail IRs that carry about half the total error.

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


def test_log_shift_identity_fails_with_eps():
    """log(c*A + eps) != log c + log(A + eps). Pure algebra, no optimizer.

    This is the whole reason the production path does not use a closed form for
    the log losses. The identity the closed form rests on is exact only at
    eps = 0, and losses.py uses eps = 1e-7 -- right among the quiet bins.
    """
    eps, c = 1e-7, 3.0
    A = torch.tensor([1e-9, 1e-8, 1e-7, 1e-6, 1e-5], dtype=DTYPE)
    lhs = torch.log(c * A + eps)
    rhs = math.log(c) + torch.log(A + eps)
    dev = (lhs - rhs).abs()
    assert float(dev.max()) > 0.5, "shift identity unexpectedly held"
    # Exact in the limit, and the departure grows as A falls under the knee.
    lhs0 = torch.log(c * A)
    rhs0 = math.log(c) + torch.log(A)
    assert float((lhs0 - rhs0).abs().max()) < 1e-12, "identity should be exact at eps=0"
    assert dev[0] > dev[-1], "departure should be worst for the quietest bin"
    print(f"test_log_shift_identity_fails_with_eps   OK  "
          f"(max departure {float(dev.max()):.3f} nats at A=1e-9)")


def test_log_minimizer_moves_with_eps():
    """And the departure moves the minimizer, not just the objective's value.

    Needs data that is *not* a clean rescale: when the prediction is already
    close to a scalar multiple of the target -- the well-fit case -- every one
    of these objectives minimizes at nearly the same c, since far below the knee
    log(t+eps) - log(cp+eps) ~ (t - cp)/eps, the linear objective over a
    constant. The rules diverge exactly where the shape fit is poor, which is
    the tail this pipeline cares about.

    Here 70% of the bins sit under the knee and disagree with the target by a
    factor of 10, while 30% sit well above it and agree. A pure log weights both
    groups equally and follows the majority; log(x + 1e-7) squashes the group
    below the knee and follows the loud minority instead.
    """
    c_true = 2.7
    quiet_p = torch.logspace(-9, -8, 700, dtype=DTYPE)
    loud_p = torch.logspace(-6, -5, 300, dtype=DTYPE)
    p = torch.cat([quiet_p, loud_p]).unsqueeze(0)
    t = torch.cat([10.0 * c_true * quiet_p, c_true * loud_p]).unsqueeze(0)

    def loss_exact(a, b):
        return (torch.log(a) - torch.log(b)).abs().mean(dim=-1)

    def loss_eps(a, b):
        return (torch.log(a + 1e-7) - torch.log(b + 1e-7)).abs().mean(dim=-1)

    c0 = recover_c(loss_exact, t, p, mu_pred=10.0)
    ce = recover_c(loss_eps, t, p, mu_pred=10.0)
    assert c0 / c_true > 5.0, f"eps=0 log should follow the 70% majority, got {c0:.3f}"
    assert ce / c_true < 2.0, f"eps=1e-7 should follow the loud bins, got {ce:.3f}"
    assert c0 / ce > 3.0, (
        f"minimizers barely moved (eps=0 {c0:.3f}, eps=1e-7 {ce:.3f}); if this "
        f"ever holds, the premise for a search over a closed form is gone"
    )
    print(f"test_log_minimizer_moves_with_eps        OK  "
          f"(eps=0 c={c0:.3f}, eps=1e-7 c={ce:.3f}, true {c_true})")


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
    test_log_shift_identity_fails_with_eps()
    test_log_minimizer_moves_with_eps()
    test_search_respects_mu_bounds()
    print("\nall OK")
