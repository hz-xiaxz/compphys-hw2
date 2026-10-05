"""HW2 Problem 1 (Newman 6.11): relaxation and overrelaxation.

Solve x = 1 - exp(-c x) for c = 2 to an accuracy of 1e-6, first with ordinary
relaxation, x' = f(x), then with overrelaxation, x' = (1 + w) f(x) - w x.

Linearising about the fixed point x* (with the error defined as eps = x* - x)
gives eps' = g eps with g = (1 + w) f'(x*) - w, so the error on the new
estimate is

    eps' = (x - x') / (1 - 1/g),

which is used as the stopping criterion, with f'(x*) replaced by f' at the
current iterate.  The optimum w makes g = 0, i.e. w_opt = f'(x*) / (1 - f'(x*)),
which is negative whenever f'(x*) < 0 (part d).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from plotstyle import use_science

use_science()

TOL = 1e-6
C = 2.0


def f_perc(x, c=C):
    return 1.0 - np.exp(-c * x)


def df_perc(x, c=C):
    return c * np.exp(-c * x)


def overrelax(f, df, x0, omega=0.0, tol=TOL, maxiter=10_000):
    """Overrelaxation (omega = 0 is plain relaxation).

    Returns the solution, the number of iterations, the list of iterates and the
    list of error estimates eps' = (x - x') / (1 - 1/g).
    """
    x = x0
    xs, est = [x0], []
    for n in range(1, maxiter + 1):
        x_new = (1.0 + omega) * f(x) - omega * x
        g = (1.0 + omega) * df(x_new) - omega
        err = abs((x - x_new) / (1.0 - 1.0 / g)) if g != 0 else 0.0
        xs.append(x_new)
        est.append(err)
        x = x_new
        if err < tol:
            return x, n, np.array(xs), np.array(est)
        if not np.isfinite(x) or abs(x) > 1e6:
            break
    return np.nan, maxiter, np.array(xs), np.array(est)


# --- reference solutions --------------------------------------------------
X_STAR = brentq(lambda x: x - f_perc(x), 0.1, 2.0, xtol=1e-15)
GP = df_perc(X_STAR)                       # f'(x*) = c (1 - x*)
W_OPT = GP / (1.0 - GP)

print(f"x = 1 - exp(-{C:g} x):  x* = {X_STAR:.12f},  f'(x*) = {GP:.6f}")
print(f"predicted optimal omega = f'/(1-f') = {W_OPT:.4f}\n")

# (b) plain relaxation
x_r, n_r, xs_r, _ = overrelax(f_perc, df_perc, 1.0, omega=0.0)
print(f"(b) relaxation:     x = {x_r:.8f}  in {n_r:3d} iterations  "
      f"(true error {abs(x_r - X_STAR):.1e})")

# (c) overrelaxation for a few omega, then a fine scan
print("(c) overrelaxation:")
for w in [0.25, 0.5, 0.6, 0.65, 0.7, 0.8, 1.0, 1.5]:
    xw, nw, _, _ = overrelax(f_perc, df_perc, 1.0, omega=w)
    print(f"    omega = {w:4.2f}:  x = {xw:.8f}  in {nw:3d} iterations  "
          f"(true error {abs(xw - X_STAR):.1e})")

omegas = np.round(np.arange(-0.9, 2.31, 0.01), 2)
n_scan = np.array([overrelax(f_perc, df_perc, 1.0, omega=w)[1] for w in omegas])
n_scan_true = []
for w in omegas:                      # iterations until the TRUE error < 1e-6
    _, _, xs, _ = overrelax(f_perc, df_perc, 1.0, omega=w, tol=0.0, maxiter=400)
    ok = np.nonzero(np.abs(xs[1:] - X_STAR) < TOL)[0]
    n_scan_true.append(ok[0] + 1 if len(ok) else np.nan)
n_scan_true = np.array(n_scan_true)
k = int(np.argmin(n_scan))
print(f"    fastest in the scan: omega = {omegas[k]:.2f} with {n_scan[k]} iterations "
      f"(vs {n_r} for plain relaxation)")

# (d) an equation with f'(x*) < 0, where omega < 0 is the right choice
X_COS = brentq(lambda x: x - np.cos(x), 0.0, 1.0, xtol=1e-15)
GP_COS = -np.sin(X_COS)
W_COS = GP_COS / (1.0 - GP_COS)
n_cos = np.array([overrelax(np.cos, lambda x: -np.sin(x), 1.0, omega=w)[1]
                  for w in omegas])
print(f"\n(d) x = cos x:  x* = {X_COS:.10f},  f'(x*) = {GP_COS:.4f},  "
      f"predicted optimal omega = {W_COS:.4f}")
for w in [0.5, 0.2, 0.0, -0.2, -0.3, -0.4, -0.6]:
    xw, nw, _, _ = overrelax(np.cos, lambda x: -np.sin(x), 1.0, omega=w)
    g = (1.0 + w) * GP_COS - w
    if np.isnan(xw):
        print(f"    omega = {w:5.2f}:  diverges (g = {g:+.3f}, |g| > 1)")
    else:
        print(f"    omega = {w:5.2f}:  x = {xw:.8f}  in {nw:3d} iterations  (g = {g:+.3f})")
kc = int(np.argmin(n_cos))
print(f"    fastest in the scan: omega = {omegas[kc]:.2f} with {n_cos[kc]} iterations")

# --- figure -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

ax = axes[0]
for w, col in [(0.0, "C0"), (0.5, "C2"), (round(W_OPT, 2), "C1"), (1.2, "C3")]:
    _, n, xs, est = overrelax(f_perc, df_perc, 1.0, omega=w)
    it = np.arange(1, len(xs))
    lab = "relaxation" if w == 0 else rf"$\omega = {w:g}$"
    ax.semilogy(it, np.abs(xs[1:] - X_STAR), "o-", ms=2.5, lw=1.0, color=col, label=lab)
    ax.semilogy(it, est, "x", ms=3.5, color=col, alpha=0.8)
ax.axhline(TOL, color="0.4", ls=":", lw=1)
ax.set_xlabel("iteration $n$")
ax.set_ylabel(r"error $|x_n - x^*|$")
ax.set_title(r"$x = 1 - e^{-2x}$: convergence", fontsize=8)
ax.set_ylim(1e-10, 1)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.legend(fontsize=6, loc="upper right")
ax.text(0.04, 0.05, r"lines: true error; $\times$: estimate $\varepsilon'$",
        transform=ax.transAxes, fontsize=6, color="0.3")

ax = axes[1]
ax.plot(omegas, n_scan, "-", lw=1.2, color="C0", label=r"estimate $|\varepsilon'| < 10^{-6}$")
ax.plot(omegas, n_scan_true, "--", lw=0.9, color="C3", label=r"true error $< 10^{-6}$")
ax.axvline(W_OPT, color="C1", ls=":", lw=1)
ax.axvline(0.0, color="0.5", lw=0.6)
ax.text(W_OPT + 0.06, 36, rf"$\omega_{{\rm opt}} = {W_OPT:.3f}$", fontsize=7, color="C1")
ax.set_xlabel(r"overrelaxation parameter $\omega$")
ax.set_ylabel("iterations to converge")
ax.set_title(r"$x = 1 - e^{-2x}$, $f'(x^*) = %.3f$" % GP, fontsize=8)
ax.set_ylim(0, 60)
ax.legend(fontsize=6, loc="upper left", framealpha=0.9)

ax = axes[2]
ax.plot(omegas, n_cos, "-", lw=1.2, color="C0")
ax.axvline(W_COS, color="C1", ls=":", lw=1)
ax.axvline(0.0, color="0.5", lw=0.6)
ax.text(W_COS + 0.06, 36, rf"$\omega_{{\rm opt}} = {W_COS:.3f}$", fontsize=7, color="C1")
ax.set_xlabel(r"overrelaxation parameter $\omega$")
ax.set_ylabel("iterations to converge")
ax.set_title(r"$x = \cos x$, $f'(x^*) = %.3f$" % GP_COS, fontsize=8)
ax.set_ylim(0, 60)

fig.savefig("figs/problem1_relaxation.png")
