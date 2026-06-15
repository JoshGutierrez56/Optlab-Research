"""Run a single notebook via nbconvert (copy-to-temp to avoid space issues)."""
import json, os, shutil, sys, tempfile

sys.stdout.reconfigure(encoding='utf-8')

repo = r"C:\Users\Owner\OneDrive\Desktop\School Work\Grad School Work\Options Club\optlab-research"
os.chdir(repo)

# Read notebook path from command line or default
if len(sys.argv) > 1:
    nb_rel = sys.argv[1]
else:
    nb_rel = "notebooks/factor_replications/01_momentum_replication.ipynb"

full_nb_path = os.path.join(repo, nb_rel)
print(f"Source: {full_nb_path}")
print(f"Exists: {os.path.exists(full_nb_path)}")

# Read notebook
import nbformat
with open(full_nb_path, encoding='utf-8') as f:
    orig_nb = nbformat.read(f, as_version=4)

print(f"Cells: {len(orig_nb.cells)}")
code_cells_count = sum(1 for c in orig_nb.cells if c.cell_type == "code")
md_cells_count = sum(1 for c in orig_nb.cells if c.cell_type == "markdown")
print(f"  {code_cells_count} code, {md_cells_count} markdown")

# Copy to temp dir without spaces
tmpdir = tempfile.mkdtemp(prefix="nb_exec_")
shutil.copy(full_nb_path, os.path.join(tmpdir, os.path.basename(full_nb_path)))
nb_basename = os.path.basename(full_nb_path)
print(f"\nCopied to: {tmpdir}")

# Execute with nbconvert
result = __import__('subprocess').run(
    [sys.executable, "-Xfrozen_modules=off", "-m", "jupyter", "nbconvert", "--to", "notebook",
     "--execute", nb_basename,
     "--ExecutePreprocessor.timeout=600",
     f"--output", "_out"],
    cwd=tmpdir, capture_output=True, text=True, timeout=900
)

print(f"\nnbconvert return code: {result.returncode}")
if result.returncode != 0:
    print(f"STDERR last lines:\n{result.stderr[-2000:]}")

# Read output
out_path = os.path.join(tmpdir, "_out.ipynb")
errors_found = []
passed_count = 0

if os.path.exists(out_path):
    with open(out_path, encoding='utf-8') as f:
        exec_nb = json.load(f)
    
    print("\n" + "=" * 70)
    print("CELL-BY-CELL REPORT")
    print("=" * 70)
    
    for i, cell in enumerate(exec_nb["cells"]):
        ct = cell.get("cell_type", "")
        src = "".join(cell.get("source", []))
        src_short = src[:150].replace('\n', ' ').strip() if src else ""
        
        if ct == "code":
            outputs = cell.get("outputs", [])
            has_error = False
            for out in outputs:
                if out.get("output_type") == "error":
                    has_error = True
                    ename = out.get("ename", "")
                    evalue = out.get("evalue", "")
                    tb_detail = ""
                    for tb in reversed(out.get("traceback", [])):
                        if tb.strip() and 'File' not in tb:
                            tb_detail = tb.rstrip()
                            break
                    errors_found.append((i, ename, evalue, tb_detail))
                    break
            
            status = "FAIL" if has_error else "PASS"
            if not has_error:
                passed_count += 1
            
            print(f"\nCell {i} [CODE] {status}")
            if src_short:
                print(f"  Source: {src_short[:120]}")
            if has_error:
                print(f"  Error: {ename}: {evalue}")
                if tb_detail:
                    print(f"  Detail: {tb_detail}")
            else:
                stdout = "".join(o.get("text", "") for o in outputs if o.get("output_type") == "stream")
                if stdout.strip():
                    for line in stdout.strip().split('\n')[:3]:
                        if line.strip():
                            disp = line[:140] if len(line) > 140 else line
                            print(f"  output: {disp}")
        elif ct == "markdown":
            src_p = src[:80].replace('\n', ' ').strip()
            print(f"\nCell {i} [MD   ] {src_p[:60]}")

# Clean up temp
shutil.rmtree(tmpdir, ignore_errors=True)

print("\n" + "=" * 70)
total_code = sum(1 for c in exec_nb["cells"] if c.get("cell_type") == "code")
print(f"SUMMARY: {passed_count}/{total_code} code cells passed, {len(errors_found)} failed")
if errors_found:
    print("\nErrors:")
    for ci, ename, evalue, tb in errors_found:
        print(f"  Cell {ci}: {ename}: {evalue}")
