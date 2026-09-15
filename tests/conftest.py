"""Shared fixtures, including the Zenodo download used by ``test_zenodo.py``.

Every other suite builds its own synthetic channel, which is what makes them
fast and hermetic -- and also what let a file-reading bug ship unnoticed: a
synthetic fixture is written by the same conventions the reader assumes, so a
mismatch between those conventions and the released files cannot show up.
``test_zenodo.py`` closes that gap by running the package against the files a
user actually downloads.

That costs a few hundred megabytes of network, so the suite runs only when
asked for:

* ``pytest --zenodo``,
* ``UWA_CHANNELS_ZENODO=1 pytest``, or
* a cache directory that already holds the files, in which case there is
  nothing to download and the tests run as ordinary ones.

Downloads land in ``$UWA_CHANNELS_CACHE`` if set, else
``~/.cache/uwa-channels``, and are checked against the MD5 digests Zenodo
publishes for this record.  Point ``UWA_CHANNELS_CACHE`` at a directory you
already keep the library in and no download happens at all.

Authors: Zhengnan Li, Claude (Opus 5)
Email  : uwa-channels@ofdm.link
License: MIT

Revision history:
  - Sep. 15, 2026: Initial release.
"""

import hashlib
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# Concept DOI 10.5281/zenodo.21287413 resolves to the newest version; this is
# the version-specific record, so the files and their digests are immutable.
ZENODO_RECORD = "21287414"
ZENODO_DOI = "10.5281/zenodo.21287414"

# The smallest channel in the library (162 MB) and the noise measured on the
# same 8-element array.  MD5 digests as published by the Zenodo REST API,
# https://zenodo.org/api/records/21287414.
CHANNEL_FILE = "black.mat"
NOISE_FILE = "black_noise.mat"
FILES = {
    CHANNEL_FILE: "16307b935cac01d02902e04a265f08cd",
    NOISE_FILE: "75f955908bfa7bf3261b0f855e8b0bbe",
}

_TIMEOUT = 60.0
_CHUNK = 1 << 20


def cache_dir():
    """Directory the library files are kept in."""
    env = os.environ.get("UWA_CHANNELS_CACHE")
    root = Path(env) if env else Path.home() / ".cache" / "uwa-channels"
    return root


def _md5(path):
    # Zenodo publishes MD5 digests, so this is a transfer check against their
    # manifest, not a security one.
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _cached(name):
    """Path to ``name`` if it is present and intact, else None."""
    path = cache_dir() / name
    if path.is_file() and _md5(path) == FILES[name]:
        return path
    return None


def all_files_cached():
    """Whether every file is already on disk, by name only.

    Deliberately cheap: this runs at collection on every ``pytest`` invocation,
    so it does not digest a 162 MB file to decide whether to enable a suite.
    :func:`fetch` checks the contents.
    """
    return all((cache_dir() / name).is_file() for name in FILES)


def fetch(name):
    """Return a local path to the Zenodo file ``name``, downloading it once.

    A download that cannot be reached skips the calling test: an offline
    machine says nothing about the package.  A download that arrives with the
    wrong digest fails it, because that does.
    """
    hit = _cached(name)
    if hit is not None:
        return hit

    root = cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    dest = root / name
    part = dest.with_name(dest.name + ".part")
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{name}/content"

    try:
        # nosec B310: url is the https literal above with a file name appended.
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as response:
            with open(part, "wb") as handle:
                shutil.copyfileobj(response, handle, _CHUNK)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        part.unlink(missing_ok=True)
        pytest.skip(f"cannot download {name} from Zenodo ({exc})")

    got = _md5(part)
    if got != FILES[name]:
        part.unlink(missing_ok=True)
        pytest.fail(
            f"{name} downloaded from Zenodo has MD5 {got}, expected "
            f"{FILES[name]}; the record may have been revised, or the "
            f"transfer was truncated."
        )
    part.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# Opting in
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--zenodo",
        action="store_true",
        default=False,
        help="run the end-to-end tests against the released files on Zenodo "
        "(downloads ~162 MB on the first run)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "zenodo: end-to-end test against the released files on Zenodo; "
        "needs --zenodo, UWA_CHANNELS_ZENODO=1, or a populated cache",
    )


def pytest_collection_modifyitems(config, items):
    enabled = (
        config.getoption("--zenodo")
        or os.environ.get("UWA_CHANNELS_ZENODO") == "1"
        or all_files_cached()
    )
    if enabled:
        return
    skip = pytest.mark.skip(
        reason=f"needs the library files from {ZENODO_DOI}; run with --zenodo, "
        f"set UWA_CHANNELS_ZENODO=1, or put them in {cache_dir()}"
    )
    for item in items:
        if "zenodo" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# The files themselves
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def channel_path():
    return fetch(CHANNEL_FILE)


@pytest.fixture(scope="session")
def noise_path():
    return fetch(NOISE_FILE)

# [EOF]
