import os


def get_data_path(filename: str, subdir: str = "optimizer") -> str:
    """
    Get the path to a data file so it works both from scripts and Jupyter notebooks.

    Parameters
    ----------
    filename : str
        The file name (e.g., "test2.nc").
    subdir : str, optional
        Subdirectory where the file may live (default is "optimizer").

    Returns
    -------
    str
        Full path to the file.
    """
    try:
        # Works when running as a script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Move up one level (utils → WESL root)
        base_dir = os.path.abspath(os.path.join(base_dir, ".."))
    except NameError:
        # Works in Jupyter (__file__ not defined)
        base_dir = os.getcwd()

    # First try in subdir
    path = os.path.join(base_dir, subdir, filename)
    if os.path.exists(path):
        return path

    # Then try directly in base_dir
    path = os.path.join(base_dir, filename)
    if os.path.exists(path):
        return path

    raise FileNotFoundError(f"Could not locate {filename} in {base_dir} or {subdir}/")
