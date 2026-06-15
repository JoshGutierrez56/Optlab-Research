"""Execute notebook cells using IPython's exec without nbconvert/nbclient."""
import json, os, sys, io, contextlib, traceback

sys.stdout.reconfigure(encoding='utf-8')

# Notebook to execute - passed as arg or default
nb_path = r"C:\Temp\nb01.ipynb"

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

print(f"Notebook: {nb_path}")
print(f"Cells: {len(nb['cells'])}")

# Extract cells
code_cells = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell.get('source', []))
        code_cells.append((i, src))

print(f"Code cells: {len(code_cells)}")

# Execute using IPython
import IPython
from IPython.core.interactiveshell import InteractiveShell
shell = InteractiveShell.instance()

# Set up namespace for execution
namespace = {}
results = []  # (cell_index, status, output_text, error_text)

for i, src in code_cells:
    print(f"\n{'='*70}")
    print(f"Cell {i}:")
    
    # Capture stdout
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        try:
            # Execute cell code in shell's namespace
            result = shell.run_cell(src, store_history=False)
            
            # Get output from the shell's display hook results
            # IPython stores formatted outputs in result.success/result.result
            
            stdout_text = stdout_buf.getvalue() + (result.result if hasattr(result, 'result') and result.result else '')
            stderr_text = stderr_buf.getvalue()
            
            if result.error_before_exec is not None:
                err = str(result.error_before_exec)
                print(f"  ❌ BEFORE EXEC: {err}")
                results.append((i, 'ERROR', '', err))
            elif result.error_in_exec is not None:
                tb_text = ''.join(traceback.format_exception(
                    type(result.error_in_exec), 
                    result.error_in_exec, 
                    result.error_in_exec.__traceback__
                ))
                print(f"  ❌ ERROR: {result.error_in_exec}")
                results.append((i, 'ERROR', '', tb_text))
            else:
                output_lines = stdout_text.strip().split('\n')[:3]
                for line in output_lines:
                    if line.strip():
                        print(f"  ✅ {line[:120]}")
                results.append((i, 'PASS', stdout_text, stderr_text))
        
        except Exception as e:
            err_text = f"Exception: {e}\n{traceback.format_exc()}"
            print(f"  ❌ EXCEPTION: {e}")
            results.append((i, 'EXCEPTION', '', err_text))

# Report summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

passed = sum(1 for _, s, _, _ in results if s == 'PASS')
failed = [(i, s, e) for i, s, _, e in results if s != 'PASS']

print(f"Passed: {passed}/{len(results)}")
print(f"Failed/Errors: {len(failed)}")

if failed:
    print("\nErrors:")
    for i, status, _err_text in failed:
        print(f"  Cell {i} [{status}]")

# Write back with outputs (add any cell outputs from IPython execution)
for j, (i, s, stdout, stderr) in enumerate(results):
    # Find the original cell and add ipython3 outputs
    orig_cell_idx = [k for k, c in enumerate(nb['cells']) if ''.join(c.get('source',[]))[:50] == ''.join(code_cells[j][1])[:50]][0] if j < len(nb['cells']) else None
    # Actually we need to map properly
    pass

# Simpler: write back with empty outputs (we can't easily reconstruct them from IPython)
# Just note the results
print("\nNotebook execution complete. Results logged above.")
print("Cell-by-cell output saved to stdout for review.")
