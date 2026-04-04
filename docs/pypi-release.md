# Publishing To PyPI

The Python package is published as:

- package name: `forestfire-ml`
- import name: `forestfire`

## Local release flow

Build and validate:

```bash
task python-package-check
```

Upload to TestPyPI:

```bash
task python-package-publish-testpypi
```

Upload to PyPI:

```bash
task python-package-publish
```

## Trusted publishing

This repository also includes GitHub Actions trusted publishing for PyPI in:

- `.github/workflows/publish-python.yaml`

That workflow builds wheels and an sdist, then publishes them through PyPI trusted publishing.

The docs site is deployed separately through `.github/workflows/docs.yaml` to GitHub Pages.

## Before releasing

Make sure you:

- bump the version in the workspace `pyproject.toml`
- bump the version in `bindings/python/pyproject.toml`
- bump the version in `bindings/python/Cargo.toml`
- verify the mixed package metadata still includes:
  - `module-name = "forestfire._core"`
  - `python-source = "python"`
  - `python/forestfire/_core.pyi`
  - `python/forestfire/py.typed`
- verify the PyPI project name is correct
- confirm that the `forestfire-ml` project name is the intended PyPI target
- run the package check task before tagging
- ensure the GitHub trusted publisher is configured for the correct project name

## Local tasks

- `task python-package-build`
- `task python-package-check`
- `task python-package-publish-testpypi`
- `task python-package-publish`
