"""
1x3 figure: global inter-event-time CCDF for DANA, RENFE 2024, RENFE 2026,
with empirical data + truncated power-law, power-law and lognormal fits.
Larger-font version.
"""
import os
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
import powerlaw

DATA = os.environ.get("DATA_DIR", "../../../Datasets")   # adjust to your Datasets path

def global_iets(df, dmin, dmax):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_localize(None)
    df = df.dropna(subset=['date']).sort_values('date')
    if dmin: df = df[df['date'] >= dmin]
    if dmax: df = df[df['date'] <  dmax]
    w = df['date'].diff().dt.total_seconds().dropna().values
    return w[(w > 0) & np.isfinite(w)]

dana  = pd.read_json(f"{DATA}/voluntariosdanavalencia_old.json")
renfe = pd.read_json(f"{DATA}/vagarenfe.json")
COLS = [("(a) DANA",       global_iets(dana,  None,        "2025-01-01")),
        ("(b) RENFE 2024", global_iets(renfe, "2024-01-01","2025-01-01")),
        ("(c) RENFE 2026", global_iets(renfe, "2026-01-15", None))]

C_EMP="black"; C_TPL="#D62728"; C_PL="#1f77b4"; C_LN="#ff7f0e"
mpl.rcParams.update({'figure.dpi':130,'savefig.dpi':300,'font.size':13,
    'axes.labelsize':16,'axes.titlesize':17,'legend.fontsize':11.5,
    'xtick.labelsize':13,'ytick.labelsize':13,'axes.spines.top':False,'axes.spines.right':False})

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
for ax, (title, w) in zip(axes, COLS):
    fit = powerlaw.Fit(w, discrete=False, verbose=False)
    tpl = fit.truncated_power_law
    a, lam = float(tpl.alpha), float(tpl.Lambda)
    print(f"{title}: alpha={a:.3f} lambda={lam:.3e} n={len(w)}")
    fit.plot_ccdf(ax=ax, color=C_EMP, lw=2.4, label="Empirical")
    tpl.plot_ccdf(ax=ax, color=C_TPL, ls="--", lw=2.4,
                  label=f"Truncated PL\n($\\alpha$={a:.2f}, $\\lambda$={lam:.1e})")
    fit.power_law.plot_ccdf(ax=ax, color=C_PL, ls="-.", lw=2.0, label="Power law")
    fit.lognormal.plot_ccdf(ax=ax, color=C_LN, ls=":", lw=2.0, label="Lognormal")
    ax.set_title(title, loc='left', fontweight='bold')
    ax.set_xlabel("Waiting time (s)")
    ax.legend(frameon=True, facecolor="white", edgecolor="0.7", framealpha=0.9, loc='lower left')
axes[0].set_ylabel(r"$P(T>\tau)$")
plt.tight_layout()
fig.savefig("waiting_time_fit_combined.png", dpi=300, bbox_inches="tight")
fig.savefig("waiting_time_fit_combined.pdf", bbox_inches="tight")
print("SAVED")
