import nbformat
import sys

nb_path = sys.argv[1]
nb = nbformat.read(nb_path, as_version=4)

for i, cell in enumerate(nb.cells):
    if cell.get('cell_type') == 'code':
        has_outputs = 'outputs' in cell
        has_exec = 'execution_count' in cell
        print(i, cell.get('id'), has_outputs, has_exec, type(cell.get('source')))
