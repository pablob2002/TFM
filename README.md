# Temporal dynamics and information disorders in emergency-driven self-organized Telegram communities

This repository contains the code and analyses for the master's thesis of the same
title (Màster en Física dels Sistemes Complexos i Biofísica, Universitat de Barcelona).
The work characterizes the temporal dynamics of communication in two self-organized
Spanish Telegram communities and the footprint of information disorders on them. It
tests whether classical models of human dynamics, based on circadian and weekly cycles,
can explain the temporal structure of communication, and studies how (mis)information
is reflected in these dynamics at the user and message-cascade levels.

Methodologically, the notebooks fit inter-event-time (IET) distributions and compare
truncated power-law, power-law, lognormal and exponential models; apply a Kaplan–Meier
finite-size correction and the time-rescaling theorem; confront the data with the
cascading non-homogeneous Poisson process of Malmgren et al.; assess stationarity
through a week-reshuffling procedure; and reconstruct and analyze message cascades.

## Data

The analyses use two datasets collected from the public Telegram API for the
communities **@voluntariosdanavalencia** (a volunteer-coordination network created
after the October 2024 DANA flood in Valencia) and **@vagarenfe** (a Rodalies Renfe
commuter-rail community, analyzed both in a routine period in 2024 and during the
emergency following the January 2026 line shutdown).

These datasets are **required to run the code**. They are expected in a `Datasets/`
folder at the project root, alongside this code folder, and the notebooks reference
them through relative paths (`../../Datasets/...`). The main files used are
`voluntariosdanavalencia_old.json`, `vagarenfe.json`, and the derived
`voluntariosdanavalencia_extracted_users.csv` and
`voluntariosdanavalencia_extracted_links.csv`.

## Repository layout

The code is organized by analysis, following the structure of the thesis. Each folder
is self-contained and writes its output to a local `figures/` subfolder. The numbering
of the notebooks within a folder indicates the order in which they are meant to be run.

### `01_Temporal_Distribution_IETs/` — global and per-channel IET distributions
Inter-event-time distributions and truncated power-law fits at the global and
per-channel level.

- `01_global_IET_DANA.ipynb` — global IET distribution, @voluntariosdanavalencia
- `02_global_IET_vagarenfe.ipynb` — global IET distribution, @vagarenfe
- `03_channel_IET_KM_DANA.ipynb` — per-channel exponent α with Kaplan–Meier correction
- `04_channel_IET_vagarenfe.ipynb` — per-channel exponent α, @vagarenfe
- `05_vagarenfe_sections.ipynb` — temporal sections of @vagarenfe
- `waiting_time_fit_combined.py`, `combine_alpha_channels_numbered.py`,
  `combine_alpha_channels.py` — scripts that assemble the combined global-IET and
  per-channel α figures across the three datasets

### `02_Time_Rescaling_TRT/` — time-rescaling theorem
Time-rescaling theorem with Kaplan–Meier correction, used to test the goodness of the
fitted models.

- `01_TRT_KM_DANA.ipynb` — time-rescaling test, @voluntariosdanavalencia
- `02_TRT_vagarenfe.ipynb` — time-rescaling test, @vagarenfe (2024 and 2026)
- `km_rescaled_bigfont.py` — script for the combined time-rescaling figures

### `03_Active_Users/` — user-level and coordinator analysis
Temporal characterization of individual users and of the community coordinators.

- `01_users_all_KM_finite_size.ipynb` — per-user α distribution (truncated power law
  with Kaplan–Meier correction)
- `02_coordinators_ideology_credibility.ipynb` — coordinator analysis, including
  ideology and credibility
- `coordinator_alpha_combined.py` — script for the combined coordinator α figure

### `04_Circadian_CNHPP/` — circadian model and stationarity test
Cascading non-homogeneous Poisson process (Malmgren et al.) and the week-reshuffling
stationarity test. The results use 200 reshuffles, with an additional 300-reshuffle
check; both sets of notebooks are provided.

- `01–03 *_200reshuffles_results.ipynb` — model fits for DANA, @vagarenfe 2024 and 2026
- `04–06 *_300reshuffles_check.ipynb` — the corresponding 300-reshuffle checks
- `07_CNHPP_failure_panel_FIG10.ipynb` — combined panel summarizing the model failure
- `08_reshuffle_stationarity_analysis.ipynb` — analysis of the reshuffling results
- `Data/` — precomputed result CSVs (`circadian_cycles_*.csv`); the notebooks read
  from and write to this folder

### `05_Misinformation_Cascades/` — cascade-level analysis
Reconstruction and characterization of message cascades and information disorders at
the cascade level.

- `01_all_cascades_segmentation_sweep_FIG15.ipynb` — cascade segmentation and sweep
  over the maximum inter-message gap
- `02_cascades_by_link_type.ipynb` — cascades split by link type
- `03_cascade_power_laws_FIG9.ipynb` — per-cascade truncated power-law fits
- `poster_plot_cell.py`, `execute_notebook_and_save.py` — figure and execution helpers

### `06_Information_Disorders_user_level/` — user-level information disorders
Footprint of (mis)information at the user level.

- `01_fake_news_temporal_user_level.ipynb` — temporal analysis of low-credibility and
  politically charged links
- `02_misinformation_bootstrap.ipynb` — bootstrap comparison of misinformation versus
  informative messages

### `tools/`
Utility scripts for working with the notebooks (`check_cells.py`,
`fix_notebook_outputs.py`).

### `TFM_notes.ctb`
Working notes for the project, kept in a [CherryTree](https://www.giuspen.net/cherrytree/)
hierarchical notebook (open with CherryTree). The accompanying `.ctb~` files are
CherryTree's automatic backups.

### `_exploratory_not_in_thesis/`
Additional analyses that are not part of the final thesis figures but are kept for
reference: a Hawkes-process model (`hawkes_process_DANA.ipynb`), a Fano-factor
analysis (`fano_factor.ipynb`), and a Weibull-versus-truncated-power-law comparison
(`active_users_weibull.ipynb`).

### `old_codes/`
Earlier versions and the full development history of the code, kept for reference. It
contains the original working folders (`Active Users/`, `Circadian Cycles/`, `Hawkes/`,
`Misinformation Analysis/`, `Misinformation Cascades/`, `Sentiment Analysis/`,
`Temporal Distribution of messages/`, `tools/`, `Images/`) with all intermediate
iterations of each analysis. The current folders above contain the consolidated,
final version of this work.

## Requirements

The notebooks are written in Python and rely on the standard scientific stack
(`numpy`, `pandas`, `matplotlib`, `scipy`) together with `powerlaw`, `lifelines` and
`seaborn`. A virtual environment is included in the repository.
