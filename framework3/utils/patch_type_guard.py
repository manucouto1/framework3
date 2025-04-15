def patch_inspect_for_notebooks():
    try:
        import dill
        import inspect

        inspect.getsource = dill.source.getsource
        print("✅ Patched inspect.getsource using dill.")
    except ImportError:
        print("⚠️ dill is not installed, skipping inspect patch.")
