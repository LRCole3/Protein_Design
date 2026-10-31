"""Helper package for the progressive deep-learning PPI notebooks.

The notebooks keep every model definition and training loop visible; this
package only holds the plumbing that would otherwise be copy-pasted between
notebooks (data download/loading, a shared evaluation scorecard, plotting).
"""
from .data import load_ppi, download_data, set_seed, get_device  # noqa: F401
