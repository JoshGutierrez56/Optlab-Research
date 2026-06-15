"""Execute notebook 1: momentum_replication and report per-cell."""
import json, sys, subprocess

repo = r"C:\Users\Owner\OneDrive\Desktop\School Work\Grad School Work\Options Club\optlab-research"
nb_path = rf"{repo}\notebooks\factor_replications\01_momentum_replication.ipynb"

# Execute the notebook via nbconvert (handles kernel issues better)
result = subprocess.run(
    [sys.executable, "-Xfrozen_modules=off", "-m", "jupyter", "nbconvert", "--to", "notebook",
     "--execute", nb_path,
     f"--ExecutePreprocessor.timeout=600",
     f"--output", "_exec1_output.ipynb"],
    cwd=repo, capture_output=True, text=True, timeout=900
)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR (truncated):")
stderr_lines = result.stderr.split('\n') if result.stderr else []
for line in stderr_lines[:20]:
    print(line)
if len(stderr_lines) > 20:
    print(f"... ({len(stderr_lines)-20} more lines)")

print(f"\nReturn code: {result.returncode}")

# Now read the output and report per-cell
if result.returncode == 0:
    out_path = rf"{repo}\notebooks\_exec1_output.ipynb"
    with open(out_path, encoding='utf-8') as f:
        nb = json.load(f)
    
    print("\n" + "=" * 70)
    print("CELL-BY-CELL REPORT")
    print("=" * 70)
    
    for i, cell in enumerate(nb["cells"]):
        cell_type = cell.get("cell_type", "")
        source_text = "".join(cell.get("source", []))
        src_short = source_text[:150].replace('\n', ' ').strip() if source_text else ""
        
        if cell_type == "code":
            outputs = cell.get("outputs", [])
            
            has_error = False
            error_msg = ""
            for out in outputs:
                if out.get("output_type") == "error":
                    has_error = True
                    ename = out.get("ename", "")
                    evalue = out.get("evalue", "")
                    error_msg = f"{ename}: {evalue}"
                    break
            
            status = "FAIL" if has_error else "PASS"
            print(f"\nCell {i} [CODE] {status}")
            if src_short:
                print(f"  Source: {src_short[:120]}")
            if has_error:
                # Get the traceback detail
                for out in outputs:
                    if out.get("output_type") == "error":
                        tb = out.get("traceback", [])
                        for tb_line in reversed(tb):
                            if tb_line.strip() and 'File' not in tb_line:
                                print(f"  Error detail: {tb_line.strip()}")
                                break
                print(f"  Error: {error_msg}")
            else:
                stdout = "".join(o.get("text","") for o in outputs if o.get("output_type") == "stream")
                if stdout.strip():
                    for line in stdout.strip().split('\n')[:4]:
                        if line.strip():
                            disp = line[:140] if len(line) > 140 else line
                            print(f"  output: {disp}")
        else:
            src_preview = source_text[:80].replace('\n', ' ').strip()
            print(f"\nCell {i} [MD   ] {src_preview[:60]}")
    
    # Summary
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    errors = sum(1 for c in code_cells if any(o.get("output_type") == "error" for o in c.get("outputs", [])))
    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(code_cells)-errors}/{len(code_cells)} code cells passed, {errors} failed")
else:
    print("\nNotebook execution FAILED (nbconvert returned non-zero)")
    
    # Still try to read if file was partially created
    out_path = rf"{repo}\notebooks\_exec1_output.ipynb"
    import os
    if os.path.exists(out_path):
        with open(out_path, encoding='utf-8') as f:
            nb = json.load(f)
        
        print("\nCells in partially-executed notebook:")
        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        errors = sum(1 for c in code_cells if any(o.get("output_type") == "error" for o in c.get("outputs", [])))
        print(f"{len(code_cells)-errors}/{len(code_cells)} code cells passed, {errors} failed")
