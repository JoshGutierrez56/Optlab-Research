import json, sys, traceback as tb_module

def run_notebook(nb_path, title):
    """Execute notebook and return cell results."""
    print(f"\n{'='*70}")
    print(f"=== {title} ===")
    print(f"{'='*70}")
    
    import nbformat
    with open(nb_path, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    code_cells = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == 'code']
    print(f"Total cells: {len(nb.cells)} ({len(code_cells)} code)")
    
    results = []  # (cell_index, status, error_info, stdout_lines)
    errors = []
    
    # Execute each cell
    for i, cell in code_cells:
        src_short = ' '.join(cell.source.split()[:10])[:90]
        print(f"\n--- Cell {i} [CODE]: {src_short}... ---")
        
        try:
            from nbclient import NotebookClient
            client = NotebookClient(
                nb, 
                timeout=600,
                kernel_name='python3',
                resources={'metadata': {'path': '/'.join(nb_path.split('/')[:-1])}}
            )
            client.execute_cell(cell, i)
            
            outputs = cell.get('outputs', [])
            has_error = any(o.get('output_type') == 'error' for o in outputs)
            stdout = ''.join(o.get('text', '') for o in outputs if o.get('output_type') == 'stream')
            
            if has_error:
                for out in outputs:
                    if out.get('output_type') == 'error':
                        ename = out.get('ename', '')
                        evalue = out.get('evalue', '')
                        errors.append((i, ename, evalue))
                        print(f"  ❌ FAIL: {ename}: {evalue}")
            else:
                results.append(('PASS', i))
                if stdout.strip():
                    for line in stdout.strip().split('\n')[:3]:
                        if line.strip():
                            disp = line[:140] if len(line) > 140 else line
                            print(f"  {disp}")
        
        except Exception as e:
            errors.append((i, 'EXCEPTION', str(e)))
            print(f"  ❌ EXCEPTION: {e}")
            tb_lines = tb_module.format_exc().split('\n')
            for t in tb_lines[-15:]:
                if t.strip():
                    print(f"    {t}")
    
    # Write back executed notebook
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    passed = len([r for r in results if r[0] == 'PASS'])
    total = len(code_cells)
    print(f"\n--- RESULTS: {passed}/{total} cells passed, {len(errors)} failed ---")
    
    return results, errors
