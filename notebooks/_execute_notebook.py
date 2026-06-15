import nbformat, nbclient, json, sys, os, traceback as tb_module

sys.stdout.reconfigure(encoding='utf-8')

# Notebook to execute
nb_path = r"C:\Temp\nb01.ipynb"
print(f"Executing: {nb_path}")

with open(nb_path, encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

code_cells = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == 'code']
print(f"Found {len(code_cells)} code cells")

results = []  # (cell_index, status, error_info)

for i, cell in code_cells:
    src_short = ' '.join(cell.source.split()[:6])[:80]
    print(f"\n--- Cell {i} [CODE]: {src_short}... ---")
    
    try:
        client = nbclient.NotebookClient(
            nb, 
            timeout=300,  # 5 min per cell
            kernel_name='python3',
            resources={'metadata': {'path': os.path.dirname(nb_path) or '.'}}
        )
        client.execute_cell(cell, i)
        
        outputs = cell.get('outputs', [])
        has_error = any(o.get('output_type') == 'error' for o in outputs)
        stdout = ''.join(o.get('text', '') for o in outputs if o.get('output_type') == 'stream')
        
        if has_error:
            error_detail = ""
            for out in outputs:
                if out.get('output_type') == 'error':
                    ename = out.get('ename', '')
                    evalue = out.get('evalue', '')
                    tb_lines = out.get('traceback', [])
                    error_detail = f"{ename}: {evalue}"
                    for tb in reversed(tb_lines):
                        if tb.strip() and 'File' not in tb:
                            error_detail += f"\n    {tb.rstrip()}"
                            break
            print(f"  ❌ FAIL: {error_detail}")
            results.append((i, 'FAIL', error_detail))
        else:
            # Show first meaningful output line
            if stdout.strip():
                for line in stdout.strip().split('\n')[:2]:
                    if line.strip():
                        print(f"  ✅ PASS → {line.strip()[:100]}")
            results.append((i, 'PASS', None))
    
    except nbclient.exceptions.CellExecutionError as e:
        error_msg = str(e).split('\n')[-5] if str(e) else "Cell execution error"
        print(f"  ❌ FAIL: {error_msg}")
        results.append((i, 'FAIL', str(e)))
    
    except nbclient.exceptions.TimeoutError as e:
        print(f"  ⏱️ TIMEOUT (cell took >300s)")
        results.append((i, 'TIMEOUT', f"Timeout after 300s"))
    
    except Exception as e:
        tb_lines = tb_module.format_tb(tb_module.extract_tb(e.__traceback__))
        print(f"  ❌ EXCEPTION: {e}")
        for line in tb_lines[-3:]:
            print(f"     {line.strip()}")
        results.append((i, 'EXCEPTION', str(e)))

print("\n" + "="*70)
print("SUMMARY:")
passed = sum(1 for _, s, _ in results if s == 'PASS')
failed = sum(1 for _, s, _ in results if s != 'PASS')
print(f"  Passed: {passed}/{len(results)}")
print(f"  Failed/Timeout: {failed}")

if failed > 0:
    print("\nErrors:")
    for i, status, detail in results:
        if status != 'PASS':
            print(f"  Cell {i} [{status}]: {detail[:150]}")

# Write back with outputs
with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f"\nNotebook saved with execution results.")
