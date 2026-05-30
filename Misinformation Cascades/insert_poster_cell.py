import json
from pathlib import Path

nb_path = Path("/Users/pablobarquin/Desktop/Master/TFM/TFM/Misinformation Cascades/260517_Misinformation_cascades_links.ipynb")

new_cell = {
    "cell_type": "code",
    "metadata": {"language": "python"},
    "source": [
        "# Celda para generar figura de poster (pegue como nueva celda y ejecútela)\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from poster_plot_cell import make_poster_figure\n",
        "\n",
        "# Recoge variables desde el notebook si existen\n",
        "opt_mix = globals().get('optimal_results_mixed', None)\n",
        "opt_mis_val = globals().get('OPTIMAL_MAX_WAITING_TIME_MIS', None)\n",
        "opt_inf_val = globals().get('OPTIMAL_MAX_WAITING_TIME_INF', None)\n",
        "\n",
        "# Genera y guarda imagen de alta resolución (600 dpi) para poster\n",
        "make_poster_figure(optimal_results_mis, optimal_results_inf, opt_mix,\n",
        "                   opt_mis_val=opt_mis_val, opt_inf_val=opt_inf_val,\n",
        "                   save_path='poster_figure.png', dpi=600)\n",
        "print('Saved poster_figure.png (600 dpi)')\n"
    ]
}

def main():
    data = json.loads(nb_path.read_text())
    if not isinstance(data.get('cells'), list):
        raise SystemExit('Notebook cells not found or malformed')

    # Append new cell
    data['cells'].append(new_cell)

    # Write back with indentation
    nb_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Appended new cell to {nb_path}")

if __name__ == '__main__':
    main()
