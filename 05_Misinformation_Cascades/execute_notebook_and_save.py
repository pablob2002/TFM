import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

nb_path = Path("/Users/pablobarquin/Desktop/Master/TFM/TFM/Misinformation Cascades/260517_Misinformation_cascades_links.ipynb")

def run_notebook(path, timeout=600):
    nb = nbformat.read(path, as_version=4)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': str(path.parent)}})
    # write executed notebook to a new file
    out_path = path.with_name(path.stem + '_executed.ipynb')
    nbformat.write(nb, out_path)
    print(f"Executed notebook saved to {out_path}")

if __name__ == '__main__':
    run_notebook(nb_path)
