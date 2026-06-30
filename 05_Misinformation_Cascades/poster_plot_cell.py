# Paste the contents of this file into a NEW notebook code cell (leave the original cell unchanged).
# This produces a poster-ready figure using the existing variables in the notebook:
# `optimal_results_mis`, `optimal_results_inf`, `optimal_results_mixed` (optional)
# and optional `OPTIMAL_MAX_WAITING_TIME_MIS`, `OPTIMAL_MAX_WAITING_TIME_INF`.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Colors: keep orange and green the same as requested
COLOR_MIS = '#ff9900'  # orange (approx)
COLOR_INF = '#2ca02c'  # green (approx)
COLOR_MIX = '#7f7f7f'  # gray

def make_poster_figure(opt_mis, opt_inf, opt_mix=None,
                       opt_mis_val=None, opt_inf_val=None,
                       save_path=None, dpi=300):
    """Create a polished 2x3 poster-ready figure.
    Expects pandas-like objects with columns named at least:
      'max_waiting_time', 'num_cascades_fitted', 'avg_cascade_size',
      'avg_waiting_times_count', 'mean_tpl_alpha', 'mean_ks_pvalue',
      'mean_ks_stat', 'pct_cascades_tpl_better'
    """
    sns.set_style('white')
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.dpi': dpi,
    })

    fig, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)

    # Unpack axes
    ax00, ax01, ax02 = axes[0]
    ax10, ax11, ax12 = axes[1]

    # Top-left: Cascade Count vs Threshold
    ax00.plot(opt_mis['max_waiting_time'], opt_mis['num_cascades_fitted'],
              marker='o', color=COLOR_MIS, lw=2, label='Misinformation')
    ax00.plot(opt_inf['max_waiting_time'], opt_inf['num_cascades_fitted'],
              marker='s', color=COLOR_INF, lw=2, label='Informative')
    if opt_mix is not None:
        ax00.plot(opt_mix['max_waiting_time'], opt_mix['num_cascades_fitted'],
                  marker='D', color=COLOR_MIX, lw=2, label='Mixed')
    ax00.set_title('Number of Fitted Cascades')
    ax00.set_xlabel('Max waiting time (s)')
    ax00.set_ylabel('Cascades fitted')
    ax00.grid(alpha=0.25)
    ax00.legend(frameon=True)

    # Top-middle: Cascade Size vs Threshold
    ax01.plot(opt_mis['max_waiting_time'], opt_mis['avg_cascade_size'],
              marker='o', color=COLOR_MIS, lw=2)
    ax01.plot(opt_inf['max_waiting_time'], opt_inf['avg_cascade_size'],
              marker='s', color=COLOR_INF, lw=2)
    if opt_mix is not None:
        ax01.plot(opt_mix['max_waiting_time'], opt_mix['avg_cascade_size'],
                  marker='D', color=COLOR_MIX, lw=2)
    ax01.set_title('Mean Cascade Size')
    ax01.set_xlabel('Max waiting time (s)')
    ax01.set_ylabel('Average size')
    ax01.grid(alpha=0.25)

    # Top-right: Average waiting time inside cascades
    ax02.plot(opt_mis['max_waiting_time'], opt_mis['avg_waiting_times_count'],
              marker='o', color=COLOR_MIS, lw=2)
    ax02.plot(opt_inf['max_waiting_time'], opt_inf['avg_waiting_times_count'],
              marker='s', color=COLOR_INF, lw=2)
    if opt_mix is not None:
        ax02.plot(opt_mix['max_waiting_time'], opt_mix['avg_waiting_times_count'],
                  marker='D', color=COLOR_MIX, lw=2)
    ax02.set_title('Mean Waiting Time Inside Cascades (s)')
    ax02.set_xlabel('Max waiting time (s)')
    ax02.set_ylabel('Mean waiting time (s)')
    ax02.grid(alpha=0.25)

    # Bottom-left: Power law exponent (alpha)
    if 'mean_tpl_alpha' in opt_mis.columns and 'mean_tpl_alpha' in opt_inf.columns:
        ax10.plot(opt_mis['max_waiting_time'], opt_mis['mean_tpl_alpha'],
                  marker='o', color=COLOR_MIS, lw=2)
        ax10.plot(opt_inf['max_waiting_time'], opt_inf['mean_tpl_alpha'],
                  marker='s', color=COLOR_INF, lw=2)
        if opt_mix is not None:
            ax10.plot(opt_mix['max_waiting_time'], opt_mix['mean_tpl_alpha'],
                      marker='D', color=COLOR_MIX, lw=2)
    ax10.set_title('Mean Power-Law Exponent (alpha)')
    ax10.set_xlabel('Max waiting time (s)')
    ax10.set_ylabel('Alpha')
    ax10.grid(alpha=0.25)

    # Bottom-middle: Goodness of fit (KS p-value)
    if 'mean_ks_pvalue' in opt_mis.columns:
        ax11.plot(opt_mis['max_waiting_time'], opt_mis['mean_ks_pvalue'],
                  marker='o', color=COLOR_MIS, lw=2)
        ax11.plot(opt_inf['max_waiting_time'], opt_inf['mean_ks_pvalue'],
                  marker='s', color=COLOR_INF, lw=2)
        if opt_mix is not None:
            ax11.plot(opt_mix['max_waiting_time'], opt_mix['mean_ks_pvalue'],
                      marker='D', color=COLOR_MIX, lw=2)
    ax11.axhline(0.05, color='0.6', linestyle='--', lw=1)
    ax11.set_title('Mean KS p-value (higher = better)')
    ax11.set_xlabel('Max waiting time (s)')
    ax11.set_ylabel('KS p-value')
    ax11.grid(alpha=0.25)

    # Bottom-right: Fit distance / KS statistic (lower better)
    if 'mean_ks_stat' in opt_mis.columns:
        ax12.plot(opt_mis['max_waiting_time'], opt_mis['mean_ks_stat'],
                  marker='o', color=COLOR_MIS, lw=2)
        ax12.plot(opt_inf['max_waiting_time'], opt_inf['mean_ks_stat'],
                  marker='s', color=COLOR_INF, lw=2)
        if opt_mix is not None:
            ax12.plot(opt_mix['max_waiting_time'], opt_mix['mean_ks_stat'],
                      marker='D', color=COLOR_MIX, lw=2)
    ax12.set_title('Mean KS Statistic (lower = better)')
    ax12.set_xlabel('Max waiting time (s)')
    ax12.set_ylabel('KS statistic')
    ax12.grid(alpha=0.25)

    # Annotate chosen optimal points if provided
    def annotate_opt(ax, xval, label, color):
        if xval is None:
            return
        # find nearest x index
        try:
            idx = (np.abs(opt_mis['max_waiting_time'] - xval)).idxmin()
            y = None
            # choose which y to annotate based on axis
            title = ax.get_title().lower()
            if 'cascade' in title and 'count' in title:
                y = opt_mis.loc[idx, 'num_cascades_fitted']
            elif 'mean cascade size' in title:
                y = opt_mis.loc[idx, 'avg_cascade_size']
            elif 'waiting' in title:
                y = opt_mis.loc[idx, 'avg_waiting_times_count']
            elif 'alpha' in title:
                y = opt_mis.loc[idx, 'mean_tpl_alpha']
            elif 'ks p-value' in title:
                y = opt_mis.loc[idx, 'mean_ks_pvalue']
            elif 'ks statistic' in title:
                y = opt_mis.loc[idx, 'mean_ks_stat']
            else:
                return
            ax.scatter([xval], [y], color=color, s=90, edgecolor='k', zorder=10)
            ax.annotate(label, (xval, y), xytext=(8, 8), textcoords='offset points', fontsize=12,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8), color=color)
        except Exception:
            pass

    annotate_opt(ax00, opt_mis_val, f"Mis opt={int(opt_mis_val)}s" if opt_mis_val else None, COLOR_MIS)
    annotate_opt(ax00, opt_inf_val, f"Inf opt={int(opt_inf_val)}s" if opt_inf_val else None, COLOR_INF)

    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=dpi)
        print(f"Saved poster figure to {save_path}")
    else:
        plt.show()

# If running inside the notebook, call the function with the variables already present.
# Paste below into the cell after the function definition or call it directly in the same cell.
# Example (paste and run):
# make_poster_figure(optimal_results_mis, optimal_results_inf, optimal_results_mixed,
#                    opt_mis_val=OPTIMAL_MAX_WAITING_TIME_MIS,
#                    opt_inf_val=OPTIMAL_MAX_WAITING_TIME_INF,
#                    save_path='poster_figure.png', dpi=300)
