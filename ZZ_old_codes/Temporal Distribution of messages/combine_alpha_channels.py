"""
Combine the three per-channel KM-corrected truncated power-law (TPL) alpha
figures (DANA, RENFE 2024, RENFE 2026) into a single row, with (a)-(c) labels
and ONE shared colormap + colorbar (messages per channel).

The per-channel alpha is recomputed from the raw datasets using the same
pipeline as the original notebooks, so the values match the individual figures.

Run from the "Temporal Distribution of messages" folder.
"""

import numpy as np
import pandas as pd
import textwrap
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
import powerlaw
from lifelines import KaplanMeierFitter
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA = "../../../Datasets"

# Shared x-axis so the three panels are directly comparable.
XLIM = (1.0, 3.05)
CMAP = "viridis"

# ----------------------------------------------------------------------
# Per-channel KM-corrected TPL exponent (same procedure as the notebooks)
# ----------------------------------------------------------------------
def compute_channel_tpl(df, cutoff_date, min_msgs):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "topic_name"])
    if df["date"].dt.tz is not None and getattr(cutoff_date, "tz", None) is None:
        df["date"] = df["date"].dt.tz_localize(None)
    if cutoff_date is not None:
        df = df[df["date"] < cutoff_date]

    sizes = df.groupby("topic_name").size().sort_values(ascending=False)
    eligible = sizes[sizes > min_msgs].index

    rows = []
    for ch_name in eligible:
        ch = df.loc[df["topic_name"] == ch_name, ["date"]].sort_values("date")
        waiting = ch["date"].diff().dt.total_seconds().dropna()
        waiting = waiting[(waiting > 0) & np.isfinite(waiting)]
        if len(waiting) < 100:
            continue

        # Kaplan-Meier correction (uses all IETs + the final censored interval)
        tau_fc = max((cutoff_date - ch["date"].iloc[-1]).total_seconds(), 0.0)
        iet = waiting.values
        km_dur = np.concatenate([iet, iet, [tau_fc]])
        km_obs = np.concatenate([np.ones(len(iet)), np.ones(len(iet)), [0]])
        kmf = KaplanMeierFitter()
        kmf.fit(km_dur, event_observed=km_obs.astype(bool))

        w = waiting.sample(6000, random_state=42).sort_values() if len(waiting) > 6000 else waiting
        try:
            fit = powerlaw.Fit(w.values, discrete=True, verbose=False)
            tpl = fit.truncated_power_law
            alpha = float(getattr(tpl, "alpha", np.nan))
            xmin = float(getattr(tpl, "xmin", np.nan))

            km_t = kmf.survival_function_.index.values
            km_s = kmf.survival_function_["KM_estimate"].values
            mask = (km_t >= xmin) & (km_s > 0) & (km_t > 0)

            a_km, s_km = np.nan, np.nan
            if np.isfinite(xmin) and mask.sum() > 5:
                tt = km_t[mask]
                ss = km_s[mask] / km_s[mask][0]

                def tpl_surv(t, a, lam):
                    return (t / xmin) ** (-(a - 1)) * np.exp(-(t - xmin) * lam)

                popt, pcov = curve_fit(
                    tpl_surv, tt, ss,
                    p0=[max(alpha, 1.01), 1.0 / iet.max()],
                    bounds=([1.001, 0.0], [20.0, np.inf]), maxfev=10000,
                )
                a_km = float(popt[0])
                s_km = float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) else np.nan

            rows.append(dict(channel=ch_name, alpha_tpl_km=a_km,
                             sigma_alpha_km=s_km, n_messages=int(sizes[ch_name])))
        except Exception as exc:
            print(f"skip {ch_name}: {exc}")

    return pd.DataFrame(rows).dropna(subset=["alpha_tpl_km"])


# ----------------------------------------------------------------------
# Compute the three datasets
# ----------------------------------------------------------------------
dana = pd.read_json(f"{DATA}/voluntariosdanavalencia_old.json")
dana_df = compute_channel_tpl(dana, pd.Timestamp("2025-01-01"), min_msgs=1000)

renfe = pd.read_json(f"{DATA}/vagarenfe.json")
renfe["date"] = pd.to_datetime(renfe["date"], errors="coerce")
if renfe["date"].dt.tz is not None:
    renfe["date"] = renfe["date"].dt.tz_localize(None)
r24 = renfe[(renfe["date"] >= "2024-01-01") & (renfe["date"] <= "2024-12-31")]
r26 = renfe[renfe["date"] >= "2026-01-15"]
r24_df = compute_channel_tpl(r24, r24["date"].max(), min_msgs=100)
r26_df = compute_channel_tpl(r26, r26["date"].max(), min_msgs=100)

PANELS = [("(a)", "DANA", dana_df),
          ("(b)", "RENFE 2024", r24_df),
          ("(c)", "RENFE 2026", r26_df)]

# ----------------------------------------------------------------------
# Combined figure with ONE shared colormap + colorbar
# ----------------------------------------------------------------------
vmin = min(d["n_messages"].min() for _, _, d in PANELS)
vmax = max(d["n_messages"].max() for _, _, d in PANELS)
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

mpl.rcParams.update({"figure.dpi": 130, "savefig.dpi": 300})
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for ax, (lab, title, d) in zip(axes, PANELS):
    d = d.sort_values("alpha_tpl_km").reset_index(drop=True)
    d["clabel"] = d["channel"].apply(lambda s: textwrap.shorten(str(s), width=28, placeholder="..."))

    for i, row in d.iterrows():
        ax.hlines(i, XLIM[0], row["alpha_tpl_km"], color="0.8", lw=0.8, zorder=1)

    sizes = np.interp(d["n_messages"], (vmin, vmax), (30, 190))
    ax.scatter(d["alpha_tpl_km"], np.arange(len(d)), s=sizes,
               c=d["n_messages"], cmap=CMAP, norm=norm,
               edgecolor="black", lw=0.4, zorder=3)

    vs = d["sigma_alpha_km"].notna()
    ax.errorbar(d.loc[vs, "alpha_tpl_km"], np.where(vs)[0],
                xerr=d.loc[vs, "sigma_alpha_km"], fmt="none",
                ecolor="0.3", elinewidth=0.8, capsize=2, zorder=2)

    ax.axvline(1.5, color="#B22222", ls="--", lw=1, zorder=0)
    ax.axvline(2.5, color="#1F4E79", ls="-.", lw=1, zorder=0)

    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["clabel"], fontsize=7)
    ax.set_xlim(*XLIM)
    ax.set_xlabel(r"KM-corrected TPL exponent $\alpha$", fontsize=9)
    ax.set_title(title, fontsize=11)
    # Panel label — shift right a bit for RENFE 2024 to leave room for regime label
    lab_x = 0.07 if title == "RENFE 2024" else 0.0
    ax.text(lab_x, 1.05, lab, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")
    if title == "RENFE 2024":
        ax.text(0.0, 1.01, "Overloaded regime (α = 1.5)",
                transform=ax.transAxes, clip_on=False,
                color="#B22222", fontsize=6.5, va="bottom", ha="left",
                bbox=dict(facecolor="white", edgecolor="#B22222", alpha=0.9,
                          boxstyle="round,pad=0.12"))
        ax.text(1.0, 1.01, "Attentive regime (α = 2.5)",
                transform=ax.transAxes, clip_on=False,
                color="#1F4E79", fontsize=6.5, va="bottom", ha="right",
                bbox=dict(facecolor="white", edgecolor="#1F4E79", alpha=0.9,
                          boxstyle="round,pad=0.12"))
    for i, row in d.iterrows():
        ax.annotate(f"{row['alpha_tpl_km']:.2f}", (row["alpha_tpl_km"], i),
                    xytext=(0, 5.75), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5)
    ax.grid(axis="x", alpha=0.25)

axes[0].set_ylabel("Channel", fontsize=9)
fig.subplots_adjust(right=0.9, wspace=0.55)
cax = fig.add_axes([0.92, 0.15, 0.013, 0.7])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
cb.set_label("Messages per channel", fontsize=9)

fig.savefig("combined_alpha_channels.png", dpi=300, bbox_inches="tight")
fig.savefig("combined_alpha_channels.pdf", bbox_inches="tight")
print("Saved combined_alpha_channels.png / .pdf")
