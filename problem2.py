"""HW2 Problem 2 (Newman 6.13): Wien's displacement constant by binary search.

Maximising Planck's law I(lambda) = 2 pi h c^2 lambda^-5 / (exp(hc/lambda k T) - 1)
with x = hc / (lambda k T) gives 5 exp(-x) + x - 5 = 0.  Its nonzero root,
found here by bisection to an accuracy of 1e-6, gives the displacement constant
b = hc / (k x), and lambda_max = b / T then yields the Sun's surface temperature.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from plotstyle import use_science

use_science()

# CODATA 2018 exact SI values
H = 6.62607015e-34       # J s
C = 299792458.0          # m / s
KB = 1.380649e-23        # J / K

EPS = 1e-6
LAMBDA_SUN = 502e-9      # m


def g(x):
    return 5.0 * np.exp(-x) + x - 5.0


def bisect(f, x1, x2, eps=EPS):
    """Binary search for a root of f in [x1, x2]; f(x1) and f(x2) must differ in sign.

    Stops when the bracket is narrower than eps and returns its midpoint, so the
    error is at most eps/2.  Also returns the history of brackets.
    """
    f1 = f(x1)
    if f1 * f(x2) > 0:
        raise ValueError("the bracket does not contain a sign change")
    hist = [(x1, x2)]
    while abs(x2 - x1) > eps:
        xm = 0.5 * (x1 + x2)
        fm = f(xm)
        if fm * f1 > 0:          # same sign as f(x1): the root is in [xm, x2]
            x1, f1 = xm, fm
        else:
            x2 = xm
        hist.append((x1, x2))
    return 0.5 * (x1 + x2), np.array(hist)


# x = 0 is also a root (lambda -> infinity, where I -> 0); bracket away from it.
# g(1) = 5/e - 4 < 0 and g(10) = 5 + 5 e^-10 > 0.
X1, X2 = 1.0, 10.0
x, hist = bisect(g, X1, X2)
x_ref = brentq(g, X1, X2, xtol=1e-15)

b = H * C / (KB * x)
b_ref = H * C / (KB * x_ref)
db = b * (EPS / 2) / x                      # from the bisection error bound on x
T_sun = b / LAMBDA_SUN

print(f"(b) bisection on [{X1:g}, {X2:g}] with eps = {EPS:g}: {len(hist) - 1} steps")
print(f"    x  = {x:.8f}   (bracket width {hist[-1, 1] - hist[-1, 0]:.1e}, "
      f"true error {abs(x - x_ref):.1e}; reference x = {x_ref:.12f})")
print(f"    expected number of steps: ceil(log2((x2-x1)/eps)) = "
      f"{int(np.ceil(np.log2((X2 - X1) / EPS)))}")
print(f"    b  = hc/(k_B x) = {b:.7e} m K  (+- {db:.0e}; with exact x: {b_ref:.9e} m K)")
print(f"(c) lambda_max = {LAMBDA_SUN * 1e9:.0f} nm  ->  T_sun = b / lambda = {T_sun:.0f} K")

# --- figure -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw=dict(wspace=0.3))

ax = axes[0]
xx = np.linspace(0, 8, 400)
ax.plot(xx, g(xx), lw=1.2)
ax.axhline(0, color="0.5", lw=0.6)
ax.plot([0, x], [0, 0], "o", ms=4, color="C3")
ax.axvspan(X1, X2, color="C1", alpha=0.12, lw=0)
ax.annotate(f"$x = {x:.6f}$", xy=(x, 0), xytext=(4.3, -2.6), fontsize=7, color="C3",
            arrowprops=dict(arrowstyle="->", color="C3", lw=0.8))
ax.text(0.15, 0.4, "trivial root\n$x = 0$", fontsize=6.5, color="C3")
ax.text(5.0, 2.3, "initial bracket\n$[1, 10]$", fontsize=6.5, color="C1")
ax.set_xlim(-0.3, 8)
ax.set_xlabel("$x = hc/\\lambda k_B T$")
ax.set_ylabel("$5e^{-x} + x - 5$")
ax.set_title("The equation for the peak", fontsize=8)

ax = axes[1]
n = np.arange(len(hist))
mid = hist.mean(axis=1)
ax.semilogy(n, hist[:, 1] - hist[:, 0], "o-", ms=2.5, lw=1.0, label="bracket width")
ax.semilogy(n, np.abs(mid - x_ref), "s", ms=2.5, color="C3", label=r"$|x_{\rm mid} - x^*|$")
ax.semilogy(n, (X2 - X1) * 0.5 ** n, "k:", lw=0.8, label=r"$(x_2 - x_1)\,2^{-n}$")
ax.axhline(EPS, color="0.4", ls="--", lw=0.8)
ax.text(0.5, EPS * 1.6, r"$\epsilon = 10^{-6}$", fontsize=7, color="0.3")
ax.set_xlabel("bisection step $n$")
ax.set_ylabel("width / error")
ax.set_title("Convergence of the binary search", fontsize=8)
ax.legend(fontsize=6, loc="upper right")

ax = axes[2]
lam = np.linspace(100e-9, 2500e-9, 600)
for T, col in [(T_sun, "C0"), (5000.0, "C2"), (4000.0, "C1")]:
    I = 2 * np.pi * H * C ** 2 * lam ** -5 / np.expm1(H * C / (lam * KB * T))
    lab = rf"$T = {T:.0f}$ K" + (r" (Sun)" if T == T_sun else "")
    ax.plot(lam * 1e9, I / 1e13, lw=1.2, color=col, label=lab)
    lp = b / T
    Ip = 2 * np.pi * H * C ** 2 * lp ** -5 / np.expm1(H * C / (lp * KB * T))
    ax.plot(lp * 1e9, Ip / 1e13, "o", ms=3.5, color=col)
lp_curve = np.linspace(b / 6000.0, b / 3500.0, 100)
Tp = b / lp_curve
ax.plot(lp_curve * 1e9, 2 * np.pi * H * C ** 2 * lp_curve ** -5
        / np.expm1(H * C / (lp_curve * KB * Tp)) / 1e13, "k--", lw=0.7,
        label=r"peak, $\lambda = b/T$")
ax.axvline(LAMBDA_SUN * 1e9, color="C0", ls=":", lw=0.8)
ax.text(LAMBDA_SUN * 1e9 + 30, 0.4, "502 nm", fontsize=6.5, color="C0")
ax.set_xlabel(r"wavelength $\lambda$ [nm]")
ax.set_ylabel(r"$I(\lambda)$ [$10^{13}$ W m$^{-3}$]")
ax.set_title("Planck spectra and Wien's law", fontsize=8)
ax.legend(fontsize=6, loc="upper right")

fig.savefig("figs/problem2_wien.png")
