"""HW2 Problem 3: gradient descent with numerical derivatives, and a Schechter
fit to the COSMOS galaxy stellar mass function.

The minimiser only ever sees function values: the gradient is a central
difference in every coordinate.  The step is x_{i+1} = x_i - gamma * grad f(x_i)
with a step size gamma that adapts (multiplied by 1.2 after a step that lowers
f, halved and the step retried if f would rise), so the same code works for the
well-scaled test function and for the chi^2 surface.

The Schechter function is fitted in the parameters
p = (log10 phi*, log10 M*, alpha), in which the chi^2 surface is roughly
isotropic; in linear phi* its Hessian is ill-conditioned by a factor ~1e6.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from plotstyle import use_science

use_science()


# --- the minimiser ----------------------------------------------------------
def num_grad(f, x, h=1e-6):
    """Central-difference gradient, O(h^2) truncation error."""
    x = np.asarray(x, dtype=float)
    g = np.empty_like(x)
    for k in range(x.size):
        e = np.zeros_like(x)
        e[k] = h
        g[k] = (f(x + e) - f(x - e)) / (2.0 * h)
    return g


def gradient_descent(f, x0, gamma=0.1, adaptive=True, tol=1e-12, gtol=1e-8,
                     maxiter=100_000, h=1e-6, max_step=None):
    """Minimise f from x0.  Returns the path (n_iter+1, dim) and f along it.

    Stops when the gradient norm falls below gtol or a step lowers f by less
    than tol (relative).  With adaptive=False gamma is held fixed.  max_step
    caps the length of a single step, so that a huge gradient far from the
    minimum cannot throw the walker onto a distant plateau.
    """
    x = np.asarray(x0, dtype=float)
    fx = f(x)
    path, fvals = [x.copy()], [fx]
    for _ in range(maxiter):
        g = num_grad(f, x, h)
        if np.linalg.norm(g) < gtol:
            break
        while True:
            step = gamma * g
            if max_step is not None and np.linalg.norm(step) > max_step:
                step *= max_step / np.linalg.norm(step)
            x_new = x - step
            f_new = f(x_new)
            if not adaptive or f_new < fx:
                break
            gamma *= 0.5                         # overshoot: shorten and retry
            if gamma < 1e-20:
                return np.array(path), np.array(fvals)
        done = abs(fx - f_new) <= tol * max(abs(fx), 1e-300)
        x, fx = x_new, f_new
        path.append(x.copy())
        fvals.append(fx)
        if adaptive:
            gamma *= 1.2
        if done:
            break
    return np.array(path), np.array(fvals)


def num_hessian(f, x, h=1e-4):
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.empty((n, n))
    E = np.eye(n) * h
    for i in range(n):
        for j in range(n):
            H[i, j] = (f(x + E[i] + E[j]) - f(x + E[i] - E[j])
                       - f(x - E[i] + E[j]) + f(x - E[i] - E[j])) / (4 * h * h)
    return H


# --- test on f(x, y) = (x - 2)^2 + (y - 2)^2 --------------------------------
f_test = lambda p: (p[0] - 2.0) ** 2 + (p[1] - 2.0) ** 2
TEST_STARTS = [(-3.0, 4.0), (5.0, 5.0), (4.5, -2.5), (-2.0, -1.0)]
TEST_GAMMA = 0.1                                # fixed step: error * 0.8 per step

print("test function f(x,y) = (x-2)^2 + (y-2)^2, fixed gamma = 0.1")
test_runs = []
for x0 in TEST_STARTS:
    path, fv = gradient_descent(f_test, x0, gamma=TEST_GAMMA, adaptive=False,
                                tol=0.0, gtol=1e-8)
    test_runs.append((x0, path, fv))
    print(f"  start {x0}: ({path[-1][0]:.8f}, {path[-1][1]:.8f}) after "
          f"{len(path) - 1} steps, f = {fv[-1]:.2e}")

# --- the Schechter fit ----------------------------------------------------
logm, n_obs, n_err = np.loadtxt("smf_cosmos.dat", unpack=True)
LN10 = np.log(10.0)


def schechter(logm, p):
    """n(M) = phi* (M/M*)^(alpha+1) exp(-M/M*) ln 10, per dex per (Mpc/h)^3."""
    log_phi, log_ms, alpha = p
    x = 10.0 ** (logm - log_ms)            # M / M*
    return 10.0 ** log_phi * x ** (alpha + 1.0) * np.exp(-x) * LN10


def chi2(p):
    return float(np.sum(((schechter(logm, p) - n_obs) / n_err) ** 2))


STARTS = [(-2.0, 10.5, -0.5), (-3.5, 11.5, -1.5), (-4.0, 10.2, 0.0),
          (-1.5, 11.8, -1.8), (-3.0, 11.0, -0.2), (-2.5, 10.0, -1.3)]
print("\nSchechter fit, p = (log10 phi*, log10 M*, alpha)")
print(f"{'start':>24} {'steps':>6} {'chi2':>10} {'log phi*':>10} {'log M*':>9} {'alpha':>8}")
fits = []
for p0 in STARTS:
    path, cv = gradient_descent(chi2, p0, gamma=1e-6, adaptive=True, max_step=0.2)
    fits.append((p0, path, cv))
    pb = path[-1]
    print(f"{str(p0):>24} {len(path) - 1:6d} {cv[-1]:10.6f} {pb[0]:10.5f} {pb[1]:9.5f} {pb[2]:8.5f}")

finals = np.array([p[-1] for _, p, _ in fits])
best = finals[int(np.argmin([c[-1] for _, _, c in fits]))]
spread = finals.max(axis=0) - finals.min(axis=0)
print(f"\nspread of the end points over {len(STARTS)} starts: "
      f"d(log phi*) = {spread[0]:.1e}, d(log M*) = {spread[1]:.1e}, d(alpha) = {spread[2]:.1e}")

# fixed-gamma run for comparison: the stability limit is gamma < 2/lambda_max
H = num_hessian(chi2, best)
lam = np.linalg.eigvalsh(H)
gamma_fix = 1.0 / lam.max()
path_fix, cv_fix = gradient_descent(chi2, STARTS[0], gamma=gamma_fix, adaptive=False,
                                    tol=1e-12, maxiter=20_000)
print(f"\nHessian eigenvalues at the minimum: {lam[0]:.1f}, {lam[1]:.1f}, {lam[2]:.1f} "
      f"(condition number {lam.max() / lam.min():.0f})")
print(f"fixed gamma = 1/lambda_max = {gamma_fix:.2e} from {STARTS[0]}: chi2 = {cv_fix[-1]:.6f} "
      f"after {len(path_fix) - 1} steps")

# uncertainties from the curvature: cov = 2 H^-1 for chi^2 = -2 ln L
cov = 2.0 * np.linalg.inv(H)
sig = np.sqrt(np.diag(cov))
corr = cov / np.outer(sig, sig)
dof = len(logm) - 3
phi_star = 10 ** best[0]
print(f"\nbest fit: chi2 = {chi2(best):.4f} for {dof} dof (reduced {chi2(best) / dof:.3f})")
print(f"  log10 phi* = {best[0]:.4f} +- {sig[0]:.4f}   "
      f"(phi* = {phi_star:.3e} +- {phi_star * LN10 * sig[0]:.1e} (Mpc/h)^-3)")
print(f"  log10 M*   = {best[1]:.4f} +- {sig[1]:.4f}")
print(f"  alpha      = {best[2]:.4f} +- {sig[2]:.4f}")
print("  correlation matrix:\n" + np.array2string(corr, precision=3))

ref = least_squares(lambda p: (schechter(logm, p) - n_obs) / n_err, STARTS[0], xtol=1e-14,
                    ftol=1e-14, gtol=1e-14)
print(f"\ncheck with scipy least_squares: {ref.x}, chi2 = {chi2(ref.x):.6f}")

# --- figure 1: the test function --------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
ax = axes[0]
gx, gy = np.meshgrid(np.linspace(-4, 6, 300), np.linspace(-4, 6, 300))
cs = ax.contour(gx, gy, f_test((gx, gy)), levels=[0.25, 1, 4, 9, 16, 25, 36, 49],
                colors="0.7", linewidths=0.6)
ax.clabel(cs, fontsize=5, fmt="%g")
for k, (x0, path, _) in enumerate(test_runs):
    ax.plot(path[:, 0], path[:, 1], "o-", ms=2, lw=0.8, color=f"C{k}")
    ax.plot(*x0, "s", ms=4, color=f"C{k}")
ax.plot(2, 2, "k*", ms=9, label="true minimum $(2, 2)$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_aspect("equal")
ax.set_title(r"Paths of gradient descent ($\gamma = 0.1$)", fontsize=8)
ax.legend(fontsize=6, loc="lower right")

ax = axes[1]
for k, (x0, path, fv) in enumerate(test_runs):
    dist = np.hypot(path[:, 0] - 2.0, path[:, 1] - 2.0)
    ax.semilogy(fv, lw=1.2, color=f"C{k}", label=rf"start $({x0[0]:g}, {x0[1]:g})$")
    ax.semilogy(dist, ls="--", lw=0.8, color=f"C{k}")
i = np.arange(0, 90)
ax.semilogy(i, 40 * 0.64 ** i, "k:", lw=0.8)
ax.text(36, 1e-12, r"dotted: $\propto (1-2\gamma)^{2i} = 0.64^{\,i}$", fontsize=7)
ax.set_xlabel("step $i$")
ax.set_ylabel(r"$f(x_i, y_i)$ (solid), distance to $(2,2)$ (dashed)")
ax.set_title("Convergence on the test function", fontsize=8)
ax.set_ylim(1e-17, 1e2)
ax.legend(fontsize=6)
fig.savefig("figs/problem3_test.png")

# --- figure 2: chi^2 vs step and the paths in parameter space ---------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw=dict(wspace=0.32))
ax = axes[0]
for k, (p0, path, cv) in enumerate(fits):
    ax.loglog(np.arange(1, len(cv) + 1), cv, lw=1.1, color=f"C{k % 10}",
              label=rf"$({p0[0]:g},\,{p0[1]:g},\,{p0[2]:g})$")
ax.loglog(np.arange(1, len(cv_fix) + 1), cv_fix, color="0.4", ls="--", lw=0.9,
          label=r"fixed $\gamma = 1/\lambda_{\max}$")
ax.axhline(chi2(best), color="k", ls=":", lw=0.8)
ax.set_xlabel("step $i$ (+1)")
ax.set_ylabel(r"$\chi^2$")
ax.set_title(r"(a) $\chi^2$ as a function of step", fontsize=8)
ax.legend(fontsize=5.5, title=r"start $(\log\phi^*, \log M^*, \alpha)$", title_fontsize=5.5,
          loc="upper right")

# chi^2 contours in two projections, other parameter held at its best value
for ax, (a, b) in zip(axes[1:], [(1, 2), (1, 0)]):
    ga = np.linspace(9.9, 12.0, 220)
    gb = np.linspace(-2.0, 0.2, 220) if b == 2 else np.linspace(-4.2, -1.3, 220)
    Z = np.empty((gb.size, ga.size))
    for ii, vb in enumerate(gb):
        for jj, va in enumerate(ga):
            p = best.copy()
            p[a], p[b] = va, vb
            Z[ii, jj] = chi2(p)
    ax.contour(ga, gb, np.log10(Z), levels=np.arange(0.5, 6.5, 0.5), colors="0.75",
               linewidths=0.5)
    ax.contour(ga, gb, Z - chi2(best), levels=[2.30, 6.18], colors=["C3", "C3"],
               linewidths=[0.9, 0.6], linestyles=["-", "--"])
    for k, (p0, path, _) in enumerate(fits):
        ax.plot(path[:, a], path[:, b], "-", lw=0.8, color=f"C{k % 10}")
        ax.plot(p0[a], p0[b], "s", ms=3.5, color=f"C{k % 10}")
    ax.plot(best[a], best[b], "k*", ms=8)
    ax.set_xlabel(r"$\log_{10} M^*$")
    ax.set_xlim(ga[0], ga[-1])
    ax.set_ylim(gb[0], gb[-1])
axes[1].set_ylabel(r"$\alpha$")
axes[2].set_ylabel(r"$\log_{10}\phi^*$")
axes[1].set_title(r"Paths projected on $(\log M^*, \alpha)$", fontsize=8)
axes[2].set_title(r"Paths projected on $(\log M^*, \log\phi^*)$", fontsize=8)
fig.savefig("figs/problem3_chi2.png")

# --- figure 3: best fit vs data ----------------------------------------------
M = 10.0 ** logm
mm = np.logspace(9.4, 12.0, 400)
fig, (ax, axr) = plt.subplots(2, 1, figsize=(5.2, 4.8), sharex=True,
                              gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06))
ax.errorbar(M, n_obs, yerr=n_err, fmt="o", ms=3, color="k", capsize=1.5, lw=0.8,
            label="COSMOS")
ax.loglog(mm, schechter(np.log10(mm), best), color="C3", lw=1.3,
          label="best-fit Schechter")
ax.axvline(10 ** best[1], color="0.5", ls=":", lw=0.8)
ax.text(10 ** best[1] * 0.93, 3e-6, "$M^*$", fontsize=7, color="0.4", ha="right")
ax.set_ylabel(r"$n(M_{\rm gal})$ [dex$^{-1}$ (Mpc/$h$)$^{-3}$]")
ax.set_ylim(2e-6, 2e-2)
ax.legend(fontsize=7, loc="lower left")
ax.text(0.97, 0.95, "\n".join([
    rf"$\log_{{10}}\phi^* = {best[0]:.3f} \pm {sig[0]:.3f}$",
    rf"$\log_{{10}}M^* = {best[1]:.3f} \pm {sig[1]:.3f}$",
    rf"$\alpha = {best[2]:.3f} \pm {sig[2]:.3f}$",
    rf"$\chi^2 = {chi2(best):.2f}$ for {dof} dof"]),
    transform=ax.transAxes, ha="right", va="top", fontsize=7,
    bbox=dict(fc="white", ec="0.8", pad=2))
ax.set_title("Galaxy stellar mass function: Schechter fit by gradient descent", fontsize=8)
axr.axhline(0, color="0.5", lw=0.6)
axr.errorbar(M, (n_obs - schechter(logm, best)) / n_err, yerr=1, fmt="o", ms=3,
             color="k", capsize=1.5, lw=0.8)
axr.set_xscale("log")
axr.set_ylim(-2.5, 2.5)
axr.set_xlabel(r"stellar mass $M_{\rm gal}$")
axr.set_ylabel(r"$(n - n_{\rm fit})/\sigma$")
fig.savefig("figs/problem3_fit.png")
