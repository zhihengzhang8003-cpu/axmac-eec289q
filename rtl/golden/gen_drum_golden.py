"""Generate drum_int8.csv golden reference for tb_drum_multiplier.sv."""
import sys
import os
import random

# Ensure the project root is on the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

from axmac.approx_mac import drum_multiply
from axmac.exact_mac import INT8

random.seed(42)
rows = []
for _ in range(200):
    a = random.randint(-127, 127)
    b = random.randint(-127, 127)
    p = drum_multiply(a, b, INT8, k=4)
    rows.append((a, b, p))

out_path = os.path.join(script_dir, "drum_int8.csv")
with open(out_path, "w", newline="\n") as f:
    f.write("a,b,expected_product\n")
    for a, b, p in rows:
        f.write(f"{a},{b},{p}\n")

print(f"Written {len(rows)} rows")
print("Sample rows:")
for r in rows[:5]:
    print(r)
