# AGENTS.md

Notes for coding agents working in this repository.  Read `README.md` for what
the package does and `CONTRIBUTING.md` for the channel and noise file format.

## Layout

- `src/uwa_channels/` holds the package: `replay`, `unpack`, `noisegen`, and
  the loaders `load_channel`, `load_noise`, and `is_hdf5` in `io.py`.
- `examples/` holds runnable scripts that use them.
- `tests/` holds the pytest suite.

## Tests

```bash
pip install -r requirements.txt pytest
PYTHONPATH=src pytest
```

`tests/test_zenodo.py` runs the package on the released Zenodo files.  It is
skipped unless you pass `--zenodo` or set `UWA_CHANNELS_ZENODO=1`.  It
downloads the files to `$UWA_CHANNELS_CACHE`, or to `~/.cache/uwa-channels`.

```bash
PYTHONPATH=src pytest tests/test_zenodo.py --zenodo
```

CI (`.github/workflows/ci.yaml`) runs the unit tests on Python 3.10 to 3.13
and the Zenodo tests on 3.13.  Keep the code working on Python 3.10.

## Parity with MATLAB

`uwa-channels/matlab` implements the same functions, and
`uwa-channels/calibrations` checks every week that both give the same output on
the released Zenodo files.  A change to the behavior of a function here needs
the same change in the MATLAB toolbox.  Do not change argument order, index
conventions (`array_index` is 0-based here and 1-based in MATLAB; `start`), or
output length in one implementation only.

## File format

The `.mat` files on Zenodo are fixed.  Code must read them as they are;
`CONTRIBUTING.md` is the specification.  Do not change the format to suit the
code.  MATLAB v7.3 files are HDF5, and h5py reads their axes in reverse order.

## Conventions

- Format and lint with the hooks in `.pre-commit-config.yaml` (black, ruff, and
  flake8 at 88 columns).
- Public functions have numpy-style docstrings with Parameters and Returns.
- Test modules start with a docstring that says what the tests assert and why,
  with Authors, Email, License, and a dated Revision history.
- Commit messages are one imperative sentence that says what changed and why,
  for example "Read MAT v5 files, and fix the beta axis order from h5py".

## Releases

Do not change `version` in `pyproject.toml` unless you are asked to make a
release.  A push to `main` with a new version creates the tag and publishes the
package to TestPyPI and PyPI.  When you change the version, change `version` in
`CITATION.cff` to the same value.

## Citation

`CITATION.cff` cites the arXiv paper (arXiv:2609.03207) and the Zenodo data
(doi:10.5281/zenodo.21287414).  Use these when a README or a docstring needs a
citation.
