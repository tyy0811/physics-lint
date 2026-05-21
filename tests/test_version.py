"""Lock physics_lint.__version__ to the installed package metadata.

Regression guard: __version__ was once a hand-maintained literal in
__init__.py and silently drifted from pyproject.toml — the v1.0.0 release
shipped a wheel whose __version__ read "0.0.0.dev0". It now derives from
importlib.metadata; this test fails if a future change re-introduces a
hand-maintained literal that diverges from the package metadata.
"""

from importlib.metadata import version

import physics_lint


def test_version_matches_package_metadata():
    assert physics_lint.__version__ == version("physics-lint")
