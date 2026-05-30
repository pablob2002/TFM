import nbformat
from nbformat import v4
import sys

nb_path = sys.argv[1]
nb = nbformat.read(nb_path, as_version=4)
changed = False
for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'outputs' not in cell:
            cell['outputs'] = []
            changed = True
        if 'execution_count' not in cell:
            cell['execution_count'] = None
            changed = True

if changed:
    nbformat.write(nb, nb_path)
    print(f"Fixed notebook: added missing outputs/execution_count in code cells for {nb_path}")
else:
    print("No changes needed")
