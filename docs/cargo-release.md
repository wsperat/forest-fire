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

## Publish

```bash
task cargo-publish
```

Publish order is:

1. `forestfire-data`
2. `forestfire-core`
3. `forestfire-inference`

If crates.io has not indexed the previous crate yet, wait a minute and retry the remaining publish step.
