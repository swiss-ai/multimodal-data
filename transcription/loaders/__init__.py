"""
Loader registry.

Import and register new formats here.
"""

from loaders.arrow import scan_arrow

LOADERS = {
    "arrow": scan_arrow,
}