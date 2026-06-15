"""Audit all 4 notebooks: execute, report per-cell, fix bugs."""
import json, os, sys, traceback, tempfile, shutil

sys.stdout.reconfigure(encoding='utf-8')

repo = r"C:\Users\Owner\OneDrive\Desktop\School Work\Grad School Work\Options Club\optlab-research"
os.chdir(repo)
sys.path.insert(0, repo)

import nbformat

notebooks = [
    ("factor_replications/01_momentum_replication.ipynb", "N1: 01_momentum_replication"),
    ("factor_replications/02_low_vol_anomaly.ipynb", "N2: 02_low_vol_anomaly"),
    ("factor_replications/03_quality_factor.ipynb", "N3: 03_quality_factor"),
    ("portfolio_analysis/01_momentum_tcost_sensitivity.ipynb", "N4: 01_momentum_tcost_sensitivity"),
]

# ── Results accumulator ───────────────────────────────────────────────────────
all_results = []

for nb_rel, title in notebooks:
    full_path = os.path.join(repo, nb_rel)
    
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")
    
    # Read notebook
    with open(full_path, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    code_cells = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == 'code']
    print(f"Cells: {len(nb.cells)} total, {len(code_cells)} code\n")
    
    # Copy to temp dir without spaces for nbconvert
    tmpdir = tempfile.mkdtemp(prefix="nb_")
    nb_base = os.path.basename(full_path)
    shutil.copy(full_path, os.path.join(tmpdir, nb_base))
    
    errors = []  # (cell_index, ename, evalue, traceback_snippet)
    passed_indices = []
    
    # Execute via nbconvert from temp dir
    import subprocess
    result = subprocess.run(
        [sys.executable, "-Xfrozen_modules=off", "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", nb_base,
         "--ExecutePreprocessor.timeout=600",
         f"--output", "_out"],
        cwd=tmpdir, capture_output=True, text=True, timeout=900
    )
    
    out_path = os.path.join(tmpdir, "_out.ipynb")
    if result.returncode != 0 or not os.path.exists(out_path):
        print(f"nbconvert failed (rc={result.returncode}). Trying direct execution...")
        shutil.rmtree(tmpdir, ignore_errors=True)
        
        # Fallback: read and execute cells manually via nbclient per cell
        for i, cell in code_cells:
            src_short = ' '.join(cell.source.split()[:8])[:90]
            print(f"\n--- Cell {i} [CODE]: {src_short} ---")
            
            try:
                from nbclient import NotebookClient
                client = NotebookClient(nb, timeout=600, kernel_name='python3',
                                        resources={'metadata': {'path': repo}})
                client.execute_cell(cell, i)
                
                outputs = cell.get('outputs', [])
                has_err = any(o.get('output_type') == 'error' for o in outputs)
                stdout = ''.join(o.get('text', '') for o in outputs if o.get('output_type') == 'stream')
                
                if has_err:
                    for out in outputs:
                        if out.get('output_type') == 'error':
                            ename = out.get('ename', '')
                            evalue = out.get('evalue', '')
                            tb = ''
                            for t in reversed(out.get('traceback', [])):
                                if t.strip() and 'File' not in t:
                                    tb = t.rstrip()
                                    break
                            errors.append((i, ename, evalue, tb))
                            print(f"  ❌ FAIL: {ename}: {evalue}")
                            if tb:
                                print(f"     {tb}")
                else:
                    passed_indices.append(i)
                    print(f"  ✅ PASS")
                    for line in stdout.strip().split('\n')[:3]:
                        if line.strip():
                            print(f"     {line[:140]}")
            
            except Exception as e:
                errors.append((i, 'EXCEPTION', str(e), ''))
                print(f"  ❌ EXCEPTION: {e}")
        
        shutil.rmtree(tmpdir, ignore_errors=True)
        continue
    
    # Parse results from executed notebook
    with open(out_path, encoding='utf-8') as f:
        exec_nb = json.load(f)
    
    for i, cell in enumerate(exec_nb['cells']):
        ct = cell.get('cell_type', '')
        src = ''.join(cell.get('source', []))
        src_short = src[:150].replace('\n', ' ').strip() if src else ''
        
        if ct == 'code':
            outputs = cell.get('outputs', [])
            
            has_err = False
            ename = evalue = tb_detail = ''
            for out in outputs:
                if out.get('output_type') == 'error':
                    has_err = True
                    ename = out.get('ename', '')
                    evalue = out.get('evalue', '')
                    for t in reversed(out.get('traceback', [])):
                        if t.strip() and 'File' not in t:
                            tb_detail = t.rstrip()
                            break
                    errors.append((i, ename, evalue, tb_detail))
                    break
            
            if has_err:
                print(f"\nCell {i} [CODE] ❌ FAIL")
                if src_short:
                    print(f"  Source: {src_short[:120]}")
                print(f"  Error: {ename}: {evalue}")
                if tb_detail:
                    print(f"  Detail: {tb_detail}")
            else:
                passed_indices.append(i)
                stdout = ''.join(o.get('text', '') for o in outputs if o.get('output_type') == 'stream')
                
                # For cell-by-cell display when nbconvert succeeds
                print(f"\nCell {i} [CODE] ✅ PASS")
                if src_short and not has_err:
                    for line in stdout.strip().split('\n')[:3]:
                        if line.strip():
                            disp = line[:140] if len(line) > 140 else line
                            print(f"  output: {disp}")
    
    shutil.rmtree(tmpdir, ignore_errors=True)
    
    # Write executed notebook back to source
    with open(out_path.replace('_out.ipynb', nb_base), 'w', encoding='utf-8') as f:
        json.dump(exec_nb, f)
    
    total_code = sum(1 for c in exec_nb['cells'] if c.get('cell_type') == 'code')
    passed = len(passed_indices)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: {passed}/{total_code} code cells passed, {len(errors)} failed")
    if errors:
        print("\nErrors:")
        for ci, ename, evalue, tb in errors:
            print(f"  Cell {ci}: {ename}: {evalue}")
            if tb:
                print(f"         {tb[:150]}")
    
    all_results.append({
        'title': title,
        'passed': passed,
        'total_code': total_code,
        'errors': [(ci, ename, evalue) for ci, ename, evalue, _ in errors],
    })

# ── Final summary table ───────────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print("FINAL SUMMARY TABLE")
print(f"{'='*70}\n")

# Header
pad = 50
header = f"{'Notebook':<{pad}} | {'Status':<12} | {'Bugs Fixed':<12} | {'Bugs Remaining':<14}"
print(header)
print('-' * len(header))

for r in all_results:
    status = "✅ PASS" if not r['errors'] else "❌ FAIL"
    bugs_remaining = len(r['errors'])
    # We'll fill in fixed later
    line = f"{r['title']:<{pad}} | {status:<12} | {'-':<12} | {bugs_remaining:<14}"
    print(line)

print()
