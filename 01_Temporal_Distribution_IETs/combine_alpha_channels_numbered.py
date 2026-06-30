"""
Combine the three per-channel KM-corrected TPL alpha figures (DANA, RENFE 2024,
RENFE 2026) into a single row, with (a)-(c) labels and ONE shared colormap.

Legibility version: channels are NUMBERED on the y-axis, and the number->name
mapping is given in a key below each panel. Run from the
"Temporal Distribution of messages" folder.
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
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(_HERE, "../../Datasets"))
# === [TFM_newcodes] guardar figuras en ./figures/ (junto a este script) ===
_FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(_FIGDIR, exist_ok=True)


XLIM = (1.0, 3.25)
CMAP = "viridis"


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
# Combined figure
# ----------------------------------------------------------------------
vmin = min(d["n_messages"].min() for _, _, d in PANELS)
vmax = max(d["n_messages"].max() for _, _, d in PANELS)
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

mpl.rcParams.update({"figure.dpi": 130, "savefig.dpi": 300})
fig, axes = plt.subplots(1, 3, figsize=(19, 9))
fig.subplots_adjust(left=0.05, right=0.9, bottom=0.46, top=0.88, wspace=0.16)

def shorten(s, w=42):
    return textwrap.shorten(str(s), width=w, placeholder="...")

for ax, (lab, title, d) in zip(axes, PANELS):
    d = d.sort_values("alpha_tpl_km").reset_index(drop=True)
    N = len(d)
    # number 1 = top (highest alpha); row i (y=i) has number N-i
    numbers = [N - i for i in range(N)]

    for i, row in d.iterrows():
        ax.hlines(i, XLIM[0], row["alpha_tpl_km"], color="0.85", lw=0.9, zorder=1)

    sizes = np.interp(d["n_messages"], (vmin, vmax), (45, 230))
    ax.scatter(d["alpha_tpl_km"], np.arange(N), s=sizes,
               c=d["n_messages"], cmap=CMAP, norm=norm,
               edgecolor="black", lw=0.5, zorder=3)

    vs = d["sigma_alpha_km"].notna()
    ax.errorbar(d.loc[vs, "alpha_tpl_km"], np.where(vs)[0],
                xerr=d.loc[vs, "sigma_alpha_km"], fmt="none",
                ecolor="0.3", elinewidth=0.9, capsize=2.5, zorder=2)

    ax.axvline(1.5, color="#B22222", ls="--", lw=1.2, zorder=0)
    ax.axvline(2.5, color="#1F4E79", ls="-.", lw=1.2, zorder=0)

    ax.set_yticks(np.arange(N))
    ax.set_yticklabels([str(n) for n in numbers], fontsize=13)
    ax.set_ylim(-0.7, N - 0.3)
    ax.set_xlim(*XLIM)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel(r"KM-corrected TPL exponent $\alpha$", fontsize=15)
    ax.set_title(title, fontsize=18, pad=22)
    ax.text(0.0, 1.13, lab, transform=ax.transAxes,
            fontsize=20, fontweight="bold", va="center")

    if title == "RENFE 2024":
        ax.text(0.0, 1.01, "Overloaded ($\\alpha=1.5$)",
                transform=ax.transAxes, clip_on=False,
                color="#B22222", fontsize=11, va="bottom", ha="left",
                bbox=dict(facecolor="white", edgecolor="#B22222", alpha=0.9,
                          boxstyle="round,pad=0.18"))
        ax.text(1.0, 1.01, "Attentive ($\\alpha=2.5$)",
                transform=ax.transAxes, clip_on=False,
                color="#1F4E79", fontsize=11, va="bottom", ha="right",
                bbox=dict(facecolor="white", edgecolor="#1F4E79", alpha=0.9,
                          boxstyle="round,pad=0.18"))

    for i, row in d.iterrows():
        sig = row["sigma_alpha_km"] if np.isfinite(row["sigma_alpha_km"]) else 0.0
        ax.annotate(f"{row['alpha_tpl_km']:.2f}",
                    (row["alpha_tpl_km"] + sig, i),
                    xytext=(7, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=11)
    ax.grid(axis="x", alpha=0.25)

    # ---- numbered key below the panel ----
    d_desc = d.sort_values("alpha_tpl_km", ascending=False).reset_index(drop=True)
    key_lines = [f"{k+1}. {shorten(name)}" for k, name in enumerate(d_desc["channel"])]
    pos = ax.get_position()
    fig.text(pos.x0, pos.y0 - 0.09, "\n".join(key_lines),
             fontsize=14, va="top", ha="left", family="monospace",
             linespacing=1.4)

axes[0].set_ylabel("Channel (see key below)", fontsize=15)

cax = fig.add_axes([0.92, 0.50, 0.013, 0.38])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
cb.set_label("Messages per channel", fontsize=14)
cb.ax.tick_params(labelsize=12)

fig.savefig(os.path.join(_FIGDIR, "combined_alpha_channels.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(_FIGDIR, "combined_alpha_channels.pdf"), bbox_inches="tight")
print("Saved combined_alpha_channels.png / .pdf")
