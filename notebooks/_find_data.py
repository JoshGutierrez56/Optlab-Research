"""Find how register_table_view maps source paths."""
import sys, inspect

sys.stdout.reconfigure(encoding='utf-8')

from optlab.db import register_table_view

source = inspect.getsource(register_table_view)
lines = source.split("\n")
for i, line in enumerate(lines[:120]):
    print(f"{i:3d}: {line}")
