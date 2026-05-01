import importlib
import sys
import os

# Ensure project root is on sys.path so package imports resolve when running
# from VS Code / PowerShell where cwd may differ.
sys.path.insert(0, os.getcwd())

mods = [
    'triagerl.reward.components',
    'triagerl.reward.grader',
    'triagerl.reward.shaping',
    'triagerl.reward.path_quality',
]
ok = True
for m in mods:
    try:
        importlib.import_module(m)
        print(m + ' OK')
    except Exception as e:
        print(m + ' ERROR', type(e).__name__, str(e))
        ok = False
sys.exit(0 if ok else 1)
