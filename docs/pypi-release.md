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

## Automated publishing

This repository includes GitHub Actions trusted publishing for PyPI in:

- `.github/workflows/publish-python.yaml`

That workflow runs after every push to `main`, including merged pull requests.
It calculates the next patch version from the existing `v*` release tags, builds
wheels and an sdist with that version, publishes them through PyPI trusted
publishing, then tags the released commit as `vX.Y.Z`.

If a workflow is re-run for a commit that already has a `vX.Y.Z` tag, the
publish job is skipped so the same commit is not released twice.

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
- ensure the workflow has `contents: write` permission so it can create tags

## Local tasks

- `task python-package-build`
- `task python-package-check`
- `task python-package-publish-testpypi`
- `task python-package-publish`
