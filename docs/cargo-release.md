# Publishing To crates.io

## Before publishing

Make sure:

- your crates.io email address is verified
- crate versions are bumped consistently
- internal dependency versions match the crate versions being released

## Package check

```bash
task cargo-package-check
```

Notes:

- `forestfire-data` is the only crate that can be fully packaged and verified
  before release.
- `forestfire-core` cannot be packaged against crates.io until
  `forestfire-data 0.3.0` has already been published and indexed, because the
  packaged crate resolves `forestfire-data = "^0.3.0"` from crates.io rather
  than from the workspace path dependency.
- after publishing `forestfire-data`, run `cargo publish -p forestfire-core --dry-run`
  and then publish `forestfire-core`.

## Publish

```bash
cargo publish -p forestfire-data
```

Wait for crates.io indexing, then:

```bash
cargo publish -p forestfire-core --dry-run
cargo publish -p forestfire-core
```

Publish order is:

1. `forestfire-data`
2. `forestfire-core`

`forestfire-inference` and the example crate stay workspace-local and are not published.

If crates.io has not indexed the previous crate yet, wait a minute and retry the remaining publish step.
