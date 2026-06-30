"""
Regenerate the KM-corrected time-rescaling (TRT) figures with larger fonts.

 - RENFE: 2 panels (2024, 2026) with a SINGLE shared legend (colors are
   consistent across panels, built over the union of channels).
 - DANA: single panel.

Reuses the exact per-channel KM-corrected rescaling pipeline of the notebooks.
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import kstest
from scipy.optimize import curve_fit
import powerlaw

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(_HERE, "../../../Datasets"))
MIN_IETS = 30

# ---------------------------------------------------------------- helpers
def compute_iets(df_subset):
    iets = df_subset['date'].diff().dt.total_seconds().dropna()
    return iets[iets > 0].values

def fit_tpl(iets):
    data = np.asarray(iets, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    fit = powerlaw.Fit(data, discrete=False, verbose=False)
    return fit, fit.truncated_power_law

def km_survival(iets, cutoff_ts, last_msg_ts):
    last_censored = max((pd.Timestamp(cutoff_ts) - pd.Timestamp(last_msg_ts)).total_seconds(), 0.0)
    durations = np.concatenate([iets, [last_censored]])
    events = np.concatenate([np.ones(len(iets), dtype=bool), [False]])
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events)
    return (kmf.survival_function_.index.values.astype(float),
            kmf.survival_function_['KM_estimate'].values.astype(float))

def fit_tpl_to_km(km_t, km_s, xmin):
    mask = (km_t >= xmin) & (km_s > 0)
    if mask.sum() < 5:
        return None
    s_at_xmin = float(np.interp(xmin, km_t, km_s))
    if s_at_xmin <= 0:
        return None
    t_tail, s_norm = km_t[mask], km_s[mask] / s_at_xmin
    def tpl_surv(t, alpha, lam):
        return (t / xmin) ** (-(alpha - 1)) * np.exp(-(t - xmin) * lam)
    try:
        popt, _ = curve_fit(tpl_surv, t_tail, s_norm, p0=[1.5, 1e-6],
                            bounds=([1.001, 0.0], [20.0, np.inf]), maxfev=20000)
        return popt
    except Exception:
        return None

def apply_trt_km(iets, xmin, alpha_km, lam_km):
    tail = iets[iets >= xmin]
    s_vals = (tail / xmin) ** (-(alpha_km - 1)) * np.exp(-(tail - xmin) * lam_km)
    s_vals = np.clip(s_vals, 0.0, 1.0)
    u = np.clip(1.0 - s_vals, 1e-12, 1 - 1e-12)
    z = -np.log(1.0 - u)
    return z

def per_channel_km(df_period, cutoff_ts, min_iets=MIN_IETS):
    results = {}
    counts = df_period['topic_name'].value_counts()
    for ch in sorted(counts[counts >= min_iets].index.tolist()):
        ch_df = df_period[df_period['topic_name'] == ch].sort_values('date')
        iets = compute_iets(ch_df)
        if len(iets) < min_iets:
            continue
        km_t, km_s = km_survival(iets, cutoff_ts, ch_df['date'].iloc[-1])
        fit_raw, _ = fit_tpl(iets)
        xmin = float(fit_raw.truncated_power_law.xmin)
        popt = fit_tpl_to_km(km_t, km_s, xmin)
        if popt is None:
            continue
        z = apply_trt_km(iets, xmin, popt[0], popt[1])
        results[ch] = {'z': z}
    return results

def density(z_ch):
    z_ch = np.asarray(z_ch, float)
    z_ch = z_ch[np.isfinite(z_ch) & (z_ch > 0)]
    if z_ch.size < 10:
        return None
    z_min, z_max = max(np.min(z_ch), 1e-3), np.max(z_ch)
    if z_max <= z_min:
        return None
    bins = np.logspace(np.log10(z_min), np.log10(z_max), 25)
    hist, edges = np.histogram(z_ch, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    m = hist > 0
    return centers[m], hist[m], z_max

# ---------------------------------------------------------------- fonts
mpl.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 18, 'axes.labelsize': 20, 'axes.titlesize': 22,
    'legend.fontsize': 16, 'xtick.labelsize': 17, 'ytick.labelsize': 17,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ================================================================ RENFE
renfe = pd.read_json(f"{DATA}/vagarenfe.json")
renfe['date'] = pd.to_datetime(renfe['date'], utc=True)
renfe = renfe.sort_values('date').reset_index(drop=True)
df_2024 = renfe[(renfe['date'] >= pd.Timestamp('2024-01-01', tz='UTC')) &
                (renfe['date'] <= pd.Timestamp('2024-12-31 23:59:59', tz='UTC'))]
df_2026 = renfe[renfe['date'] >= pd.Timestamp('2026-01-15', tz='UTC')]

res24 = per_channel_km(df_2024, df_2024['date'].max())
res26 = per_channel_km(df_2026, df_2026['date'].max())

all_ch = sorted(set(res24) | set(res26))
tab20 = cm.get_cmap('tab20')
colors = {ch: tab20(i % 20) for i, ch in enumerate(all_ch)}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.32, top=0.93, wspace=0.18)
for ax, res, title in [(axes[0], res24, 'RENFE 2024'), (axes[1], res26, 'RENFE 2026')]:
    zmax_g = 1.0
    for ch, out in res.items():
        d = density(out['z'])
        if d is None:
            continue
        centers, hist, zmax = d
        zmax_g = max(zmax_g, zmax)
        ax.plot(centers, hist, marker='o', linestyle='none', ms=6, alpha=0.8,
                color=colors[ch])
    x = np.logspace(-3, np.log10(max(8, zmax_g)), 400)
    ax.plot(x, stats.expon.pdf(x, scale=1), 'k--', lw=2.2, label='Exp(1) PDF')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('z (log scale)')
    ax.set_title(title, fontweight='bold', pad=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
axes[0].set_ylabel('Density (log scale)')

# shared legend below both panels
handles = [Line2D([0], [0], marker='o', linestyle='none', ms=9, color=colors[ch]) for ch in all_ch]
labels = list(all_ch)
handles.append(Line2D([0], [0], color='k', ls='--', lw=2.2))
labels.append('Exp(1) PDF')
n_cols = min(len(labels), 5)
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
           ncol=n_cols, frameon=True, framealpha=0.9, title='Channel',
           columnspacing=1.0, handletextpad=0.4, fontsize=20, title_fontsize=21)
fig.savefig("paper_km_rescaled_renfe.png", dpi=300, bbox_inches="tight")
fig.savefig("paper_km_rescaled_renfe.pdf", bbox_inches="tight")
print("Saved paper_km_rescaled_renfe")

# ================================================================ DANA
dana = pd.read_json(f"{DATA}/voluntariosdanavalencia_old.json")
dana['date'] = pd.to_datetime(dana['date'], utc=True)
dana = dana.sort_values('date').reset_index(drop=True)
df_dana = dana[dana['date'] < pd.Timestamp('2025-01-01', tz='UTC')]

# Display label -> exact topic name, and fixed color map (from the notebook)
channel_topic_names = {
    'General': 'Discusión general',
    'Bienvenidos': 'Bienvenid@s a la Red de Voluntari@s (leed primero)',
    'Transporte': 'Transporte de personas a Comunidad Valenciana',
    'Grupos de Acción': 'Grupos de Acción Voluntarios en Terreno',
    'Jurídica': 'Asistencia Jurídica/Seguros',
    'Técnica': 'Asistencia Técnica y Maquinaria (mecánica, fontanería, electricidad, agricola)',
    'Animales': 'Asistencia animales compañia y otros',
    'Informática': 'Asistencia Informática/Soporte EAT',
    'Logística': 'Logística Interna Valencia(puntos de recepción)',
    'Ayuda a Empresas': 'Ayuda de Empresas',
    'Investigación': 'Investigación Ciudadana: Responsabilidades Políticas',
    'Asistencia a Voluntarios': 'Asistencia a Voluntarios (Comida y Acogida))',
    'Asistencia a Grupos Vulnerables': 'Asis. Grupos Vulnerables (Maternidad , Infancia, Mayores, etc)',
    'Material': 'Material Video/Fotográfico',
    'Webs': 'Webs Esenciales',
    'Mensajes de Ayuda': 'Mensajes de Ayuda (Ofrezco/Necesito)',
    'Voluntarios de Fuera': 'Voluntarios de fuera de Valencia (Madrid, Cataluña, Alicante, Castellón)',
    'Viviendas': 'Viviendas Edificación , Muebles y Acondicionamiento',
}
channel_colors_dana = {lab: tab20(i % 20) for i, lab in enumerate(channel_topic_names)}

MIN_MESSAGES = 1000
counts_dana = df_dana['topic_name'].value_counts().to_dict()

res_dana = {}
for label, topic in channel_topic_names.items():
    if counts_dana.get(topic, 0) < MIN_MESSAGES:
        continue
    ch_df = df_dana[df_dana['topic_name'] == topic].sort_values('date')
    iets = compute_iets(ch_df)
    if len(iets) < 30:
        continue
    km_t, km_s = km_survival(iets, df_dana['date'].max(), ch_df['date'].iloc[-1])
    fit_raw, _ = fit_tpl(iets)
    xmin = float(fit_raw.truncated_power_law.xmin)
    popt = fit_tpl_to_km(km_t, km_s, xmin)
    if popt is None:
        continue
    res_dana[label] = {'z': apply_trt_km(iets, xmin, popt[0], popt[1])}

dana_ch = list(res_dana)

fig2, ax = plt.subplots(figsize=(10.5, 5.6))
fig2.subplots_adjust(left=0.10, right=0.70, bottom=0.13, top=0.94)
zmax_g = 1.0
for ch in dana_ch:
    d = density(res_dana[ch]['z'])
    if d is None:
        continue
    centers, hist, zmax = d
    zmax_g = max(zmax_g, zmax)
    ax.plot(centers, hist, marker='o', linestyle='none', ms=5, alpha=0.8,
            color=channel_colors_dana[ch])
x = np.logspace(-3, np.log10(max(8, zmax_g)), 400)
ax.plot(x, stats.expon.pdf(x, scale=1), 'k--', lw=2, label='Exp(1) PDF')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('z (log scale)'); ax.set_ylabel('Density (log scale)')
ax.grid(True, alpha=0.25, linewidth=0.5)
handles = [Line2D([0], [0], marker='o', linestyle='none', ms=8, color=channel_colors_dana[ch]) for ch in dana_ch]
labels = list(dana_ch)
handles.append(Line2D([0], [0], color='k', ls='--', lw=2)); labels.append('Exp(1) PDF')
fig2.legend(handles, labels, loc='center left', bbox_to_anchor=(0.71, 0.5),
            frameon=True, framealpha=0.9, title='Channel')
fig2.savefig("paper_km_rescaled_intervals.png", dpi=300, bbox_inches="tight")
fig2.savefig("paper_km_rescaled_intervals.pdf", bbox_inches="tight")
print("Saved paper_km_rescaled_intervals")
print("DANA channels:", dana_ch)
