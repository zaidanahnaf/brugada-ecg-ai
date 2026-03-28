# Jalankan ini: python scripts/check_config.py
import ast, sys

with open('experiments/ablation.py', 'r') as f:
    content = f.read()

# Cari semua class definitions
tree = ast.parse(content)
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
        print(f"\nClass: {node.name} (line {node.lineno})")
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        print(f"  {target.id} = {ast.unparse(item.value)}")
            elif isinstance(item, ast.AnnAssign):
                print(f"  {ast.unparse(item.target)}: {ast.unparse(item.annotation)}", end="")
                if item.value:
                    print(f" = {ast.unparse(item.value)}", end="")
                print()