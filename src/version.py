__version__ = '0.3.0'
version_file_str = '_'.join(__version__.split('.'))  # version format suitable to be used in filenames

# HolOrama restarted version numbering at 0.1.0 after being renamed from AIVUS-CAA (last AIVUS-CAA
# release was 1.8.0). Contour files carry this marker so a freshly-saved 0.x.y file still outranks a
# pre-rename 1.x.y AIVUS-CAA file when loading, instead of losing a naive version-number comparison.
CONTOURS_VERSION_TAG = f'ho_{version_file_str}'
