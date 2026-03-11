from __future__ import annotations

import warnings

# pytest-cov/coverage and some stdlib helpers can leave sqlite connections open on Windows;
# silence these noisy ResourceWarnings so we can enforce a clean warnings budget.
warnings.filterwarnings("ignore", category=ResourceWarning)
