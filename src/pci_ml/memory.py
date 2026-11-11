import gc
from IPython import get_ipython

def release(*names):
    """Delete the named notebook globals, then collect garbage."""
    ip = get_ipython()
    if ip is None:
        return

    for name in names:
        if name not in ip.user_ns:
            continue
        obj = ip.user_ns[name]
        ip.del_var(name, by_name=True)
        del obj
    _ = gc.collect()