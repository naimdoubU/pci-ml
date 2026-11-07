import gc

from IPython import get_ipython

# Setting, changed through configure() from the notebook
_cfg = {"release": True}


def configure(release=True):
    """release=False makes release() do nothing (useful for comparing runs)."""
    _cfg["release"] = release


def release(*names):
    """Close widgets, delete the named notebook globals, then collect garbage."""
    ip = get_ipython()
    if not _cfg["release"] or ip is None:
        return
    try:
        import ipywidgets as wg
        widget_type = wg.Widget
    except ImportError:
        widget_type = ()

    for name in names:
        if name not in ip.user_ns:
            continue
        obj = ip.user_ns[name]
        if widget_type and isinstance(obj, widget_type):
            obj.close()
        ip.del_var(name, by_name=True)
        del obj
    _ = gc.collect()