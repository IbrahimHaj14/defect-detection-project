from pathlib import Path

ecf_scratch = Path("data/processed/ecf/1.scratch")

# Show the full directory tree
print("ECF 1.scratch directory structure:")
for p in sorted(ecf_scratch.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(ecf_scratch)}")