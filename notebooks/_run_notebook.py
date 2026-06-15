"""Execute a notebook, report per-cell, fix bugs if needed."""
import json, os, shutil, sys, tempfile

sys.stdout.reconfigure(encoding='utf-8')

repo = r"C:\Users\Owner\OneDrive\Desktop\School Work\Grad School Work\Options Club\optlab-research"
os.chdir(repo)

# Get notebook path from command line or use default
if len(sys.argv) > 1:
    nb_rel = sys.argv[1]
else:
    nb_rel = "notebooks/factor_replications/01_momentum_replication.ipynb"

full_nb_path = os.path.join(repo, nb_rel)
print(f"Source: {full_nb_path}\n")

import nbformat
with open(full_nb_path, encoding='utf-8') as f:
    orig_nb = nbformat.read(f, as_version=4)

print(f"Cells: {len(orig_nb.cells)}")
for i, c in enumerate(orig_nb.cells):
    src = " ".join(c.source.split()[:6]) if c.source else ""
    marker = "[CODE]" if c.cell_type == "code" else "[MD  ]"
    print(f"  Cell {i} {marker}: {src[:80]}")

# Copy to temp dir without spaces to avoid nbconvert issues
tmpdir = tempfile.mkdtemp(prefix="nb_exec_")
nb_basename = os.path.basename(full_nb_path)
shutil.copy(full_nb_path, os.path.join(tmpdir, nb_basename))

print(f"\nTemp dir: {tmpdir}")
print("=" * 70)

# Execute with nbconvert from the temp directory (no spaces in path)
result = __import__('subprocess').run(
    [sys.executable, "-Xfrozen_modules=off", "-m", "jupyter", "nbconvert", "--to", "notebook",
     "--execute", nb_basename,
     "--ExecutePreprocessor.timeout=600",
     f"--output", "_out"],
    cwd=tmpdir, capture_output=True, text=True, timeout=900
)

print(f"Return code: {result.returncode}")

errors_found = []
passed_cells = []

if os.path.exists(os.path.join(tmpdir, "_out.ipynb")):
    with open(os.path.join(tmpdir, "_out.ipynb"), encoding='utf-8') as f:
        exec_nb = json.load(f)
    
    code_count = 0
    for i, cell in enumerate(exec_nb["cells"]):
        ct = cell.get("cell_type", "")
        src = "".join(cell.get("source", []))
        src_short = src[:150].replace('\n', ' ').strip() if src else ""
        
        if ct == "code":
            code_count += 1
            outputs = cell.get("outputs", [])
            
            has_error = False
            ename = evalue = tb_detail = ""
            for out in outputs:
                if out.get("output_type") == "error":
                    has_error = True
                    ename = out.get("ename", "")
                    evalue = out.get("evalue", "")
                    for tb in reversed(out.get("traceback", [])):
                        if tb.strip() and 'File' not in tb:
                            tb_detail = tb.rstrip()
                            break
                    break
            
            status = "FAIL" if has_error else "PASS"
            if has_error:
                errors_found.append((i, ename, evalue, tb_detail))
            
            print(f"\nCell {i} [CODE] {status}")
            if src_short:
                print(f"  Source: {src_short[:120]}")
            if has_error:
                print(f"  Error: {ename}: {evalue}")
                if tb_detail:
                    print(f"  Detail: {tb_detail}")
            else:
                passed_cells.append(i)
                stdout = "".join(o.get("text", "") for o in outputs if o.get("output_type") == "stream")
                if stdout.strip():
                    for line in stdout.strip().split('\n')[:3]:
                        if line.strip():
                            disp = line[:140] if len(line) > 140 else line
                            print(f"  output: {disp}")

shutil.rmtree(tmpdir, ignore_errors=True)

print("\n" + "=" * 70)
total_code = sum(1 for c in exec_nb["cells"] if c.get("cell_type") == "code")
print(f"\nSUMMARY: {len(passed_cells)}/{total_code} code cells passed, {len(errors_found)} failed")

if errors_found:
    print("\nErrors:")
    for ci, ename, evalue, tb in errors_found:
        print(f"  Cell {ci}: {ename}: {evalue}")
        if tb:
            print(f"         {tb}")

# Return data for caller
print("\n__DATA_START__")
print(json.dumps({
    "passed": len(passed_cells),
    "total_code": total_code,
    "errors": [{"cell": ci, "ename": ename, "evalue": evalue, "tb": tb[:200]} for ci, ename, evalue, tb in errors_found],
    "passed_cell_indices": passed_cells
}))
print("__DATA_END__")
