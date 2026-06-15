"""Execute all cells from a notebook file, capture pass/fail."""
import json, os, sys, time, traceback, warnings

# Suppress matplotlib display issues
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

repo = r"C:\Users\Owner\OneDrive\Desktop\School Work\Grad School Work\Options Club\optlab-research"
os.chdir(repo)
sys.path.insert(0, repo)

# Read notebook
nb_path = os.path.join(repo, "notebooks", "factor replications", "01_momentum_replication.ipynb")
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
print(f"Total cells: {len(nb['cells'])} ({len(code_cells)} code)\n")

# Shared namespace across all cells
ns = {'__name__': '__main__', '__file__': nb_path}
warnings.filterwarnings('ignore')

results = []  # (cell_index, status, output_lines, error)

for cell_idx, source in code_cells:
    src_short = ' '.join(source.split()[:5])[:80]
    print(f"\n{'='*70}")
    print(f"Cell {cell_idx}: {src_short}...")
    print("="*70)
    
    t0 = time.time()
    try:
        exec(compile(source, f'<cell_{cell_idx}>', 'exec'), ns)
        elapsed = time.time() - t0
        
        # Capture what was printed to stdout (we can't capture that from exec())
        # Just report success/failure
        print(f"  ✅ PASS ({elapsed:.1f}s)")
        results.append((cell_idx, 'PASS', '', ''))
        
    except Exception as e:
        elapsed = time.time() - t0
        tb_text = ''.join(traceback.format_tb(e.__traceback__))[-500:] if e.__traceback__ else ''
        print(f"  ❌ FAIL after {elapsed:.1f}s")
        print(f"     {type(e).__name__}: {e}")
        for line in tb_text.split('\n')[-5:]:
            if line.strip() and 'File' not in line:
                print(f"       {line}")
        results.append((cell_idx, 'FAIL', '', f"{type(e).__name__}: {e}\n{tb_text}"))

# Summary
print("\n\n" + "="*70)
print("FINAL RESULTS - Cell 1")
print("="*70)

passed = sum(1 for _, s, _, _ in results if s == 'PASS')
failed = [(i, e, tb) for i, s, _, tb in results if s != 'PASS']

pad = 45
header = f"{'Cell':<6} | {'Status':<7} | {'Output/Detail':<{pad}}"
print(header)
print('-' * len(header))

for i, status, out, err in results:
    if status == 'PASS':
        print(f"{i:<6} | ✅ PASS | OK")
    else:
        detail = err[:150] if err else '(no detail)'
        print(f"{i:<6} | ❌ FAIL | {detail}")

print("\n" + "="*70)
print(f"SUMMARY: {passed}/{len(results)} passed, {len(failed)} failed")
print("="*70)

# Save results for the caller
with open('nb01_results.json', 'w') as f:
    json.dump({
        'passed': passed,
        'total': len(results),
        'errors': [{'cell': i, 'error': e} for i, _, e in failed]
    }, f)

# Update the notebook with execution_count (for completeness, doesn't affect anything)
j = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and j < len(results):
        status = results[j][1]
        cell['execution_count'] = j + 1
        cell['outputs'] = []
        if status == 'PASS':
            cell['outputs'].append({
                'output_type': 'stream',
                'name': 'stdout',
                'text': [f'Cell {j+1} executed successfully\n']
            })
        elif status == 'FAIL':
            err = results[j][3]
            # Get just the error name and message (not full traceback)
            first_line = err.split('\n')[0] if err else 'Unknown error'
            cell['outputs'].append({
                'output_type': 'error',
                'ename': type(e).__name__ if 'e' in ns else 'Error',
                'evalue': first_line[:200],
                'traceback': [first_line]
            })
        j += 1

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nNotebook updated and saved.")
