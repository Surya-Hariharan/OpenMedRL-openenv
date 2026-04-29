from triagerl.tasks.loader import load_all_tasks
cfgs, ids = load_all_tasks()
print('COUNT', len(ids))
print('SAMPLE', ids[:3])
