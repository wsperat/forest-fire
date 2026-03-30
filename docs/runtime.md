# Runtime, Serialization, And Optimization

## Why runtime design matters in tree systems

Tree libraries often optimize only the trainer and treat prediction as “walk the same structure later”.

ForestFire does not do that.

The project assumes training and inference have different performance needs:

- training tolerates richer bookkeeping
- inference cares about compact layouts, regular access patterns, and batched execution

So the runtime story is designed explicitly rather than inherited accidentally from the trainer.

The core idea is that ForestFire keeps two views of the same learned model:

- a semantic view for correctness, serialization, and introspection
- a runtime view for speed

The semantic view answers:

- what split is this node performing?
- what do the leaf values mean?
- how should outputs be postprocessed?
- what preprocessing assumptions must be preserved?

The runtime view answers:

- how should the CPU touch memory while scoring?
- which columns should be materialized first?
- which structures should be traversed row-by-row vs batch-by-batch?
- which information is hot-path critical, and which is only useful for export or inspection?

That split is what lets the project improve runtime behavior aggressively without turning serialization and introspection into moving targets.

## Training structure vs runtime structure

The trained model carries information that matters for understanding and exporting it:

- impurity
- gain
- sample counts
- class counts
- variances
- branch metadata

Those fields are valuable semantically but costly operationally.

On the hot prediction path they increase:

- object size
- cache pressure
- pointer chasing
- general-case branching

That is why `optimize_inference(...)` exists. It lowers the semantic model into a runtime-specialized representation.

That lowering step is intentionally one-way from the user’s point of view:

- training produces a semantic `Model`
- optimized inference derives an `OptimizedModel`
- serialization of the semantic model still uses the semantic IR
- compiled artifacts optionally snapshot the lowered runtime as an execution cache

This means ForestFire can answer “what does this model mean?” and “how should this model run?” separately.

## Reading the snippets below

The snippets in this page are intentionally representative rather than exact copies of every line in the implementation.

They are here to show:

- the naïve shape of the problem
- the lowered shape ForestFire uses instead
- why the lowered shape is friendlier to CPU execution

The real code lives in `crates/core/src`, but these snippets are meant to make the design readable without forcing you to reverse-engineer the whole crate first.

## End-to-end optimized prediction lifecycle

When you call `optimize_inference(...)`, the runtime currently goes through these stages:

1. inspect the semantic model
2. compute the set of features the model actually uses
3. build a global-to-local feature remapping
4. lower each tree or ensemble member into a runtime-oriented layout
5. optionally reorder ensemble members for better locality
6. keep the semantic model around for export and inspection

When you later call `predict(...)` on the optimized model, the runtime does:

1. accept full user-facing inference input
2. validate it against the semantic preprocessing contract
3. project the input down to the used feature subset
4. bin or normalize only that projected subset
5. choose scalar or batch-oriented scoring
6. aggregate the results back into semantic predictions

That flow is the important architectural point: optimization is not just about faster tree traversal. It starts before traversal, at input materialization.

## CART-style fallthrough layout

Binary trees are a natural fit for compact control-flow layouts.

ForestFire’s compiled CART-style runtime keeps the more common child as a fallthrough path and stores only the less common branch as an explicit jump target.

Why this helps:

- fewer explicit jumps on the hot path
- smaller runtime node representation
- less branch-heavy traversal logic

The important point is not that the model changes. It does not. The traversal strategy changes.

The specific idea is:

- every binary branch has two children
- one child is designated as the fallthrough child
- the other child becomes an explicit jump target

ForestFire chooses the fallthrough child using training-time sample counts:

- the more common child is laid out immediately after the parent
- the less common child gets stored as the explicit jump

Why use sample counts here:

- they are already part of the semantic tree
- they are a reasonable proxy for the hot path during inference
- choosing the common child as fallthrough increases the chance that the common path avoids an extra jump

This is a runtime-only choice. The semantic tree still has left and right children exactly as before.

### Naïve binary traversal

```rust
struct NaiveNode {
    feature_index: usize,
    threshold_bin: u16,
    left: usize,
    right: usize,
    is_leaf: bool,
    value: f64,
}

fn predict_naive(nodes: &[NaiveNode], row: &[u16]) -> f64 {
    let mut node_index = 0usize;
    loop {
        let node = &nodes[node_index];
        if node.is_leaf {
            return node.value;
        }

        if row[node.feature_index] > node.threshold_bin {
            node_index = node.right;
        } else {
            node_index = node.left;
        }
    }
}
```

This works, but it keeps both child references live on every branch and makes the control flow depend on a full left/right decision at every node.

### ForestFire fallthrough layout

```rust
enum OptimizedBinaryRegressorNode {
    Leaf(f64),
    Branch {
        feature_index: usize,
        threshold_bin: u16,
        jump_index: usize,
        jump_if_greater: bool,
    },
}

fn predict_fallthrough(nodes: &[OptimizedBinaryRegressorNode], row: &[u16]) -> f64 {
    let mut node_index = 0usize;
    loop {
        match &nodes[node_index] {
            OptimizedBinaryRegressorNode::Leaf(value) => return *value,
            OptimizedBinaryRegressorNode::Branch {
                feature_index,
                threshold_bin,
                jump_index,
                jump_if_greater,
            } => {
                let go_right = row[*feature_index] > *threshold_bin;
                node_index = if go_right == *jump_if_greater {
                    *jump_index
                } else {
                    node_index + 1
                };
            }
        }
    }
}
```

Why the lowered version helps:

- the common path becomes `node_index + 1`
- only the uncommon path needs an explicit jump target
- the node payload is smaller and more regular
- the traversal loop is simpler for the CPU to execute repeatedly

## Dense lookup for multiway nodes

Classifier trees with multiway branching can be expensive if every node does:

1. scan the branch list
2. compare until a matching bin is found

ForestFire lowers those into dense lookup tables indexed by bin id.

Why this helps:

- work becomes one indexed read instead of a small search
- execution becomes more regular
- the runtime cost depends less on branch-list length

This is a good example of using the regularity created by binning to simplify scoring.

This matters most for `id3` and `c45`, where the semantic structure naturally wants to branch on discrete bins rather than collapsing everything into a binary threshold tree.

At the semantic level, a multiway node still says:

- feature `f`
- branch on observed bins
- use a fallback leaf if the observed bin does not match one of the trained branches

At the runtime level, the cost becomes:

- read one projected bin id
- bounds-check it against the lookup table
- either jump to the matching child or return the fallback leaf payload

That is substantially more regular than scanning a list of branches one by one.

### Naïve multiway branch search

```rust
fn predict_multiway_naive(bin: u16, branches: &[(u16, usize)], fallback: &[f64]) -> &[f64] {
    for (branch_bin, child_index) in branches {
        if *branch_bin == bin {
            return CHILD_PROBABILITIES[*child_index].as_slice();
        }
    }
    fallback
}
```

The cost of this approach grows with branch count, and the access pattern is a tiny linear search on every row.

### ForestFire dense lookup

```rust
struct OptimizedMultiwayNode {
    feature_index: usize,
    child_lookup: Vec<usize>,
    max_bin_index: usize,
    fallback_probabilities: Vec<f64>,
}

fn predict_multiway_dense<'a>(
    node: &'a OptimizedMultiwayNode,
    projected_row: &[u16],
) -> (&'a [f64], Option<usize>) {
    let bin = usize::from(projected_row[node.feature_index]);
    if bin > node.max_bin_index {
        return (node.fallback_probabilities.as_slice(), None);
    }

    let child_index = node.child_lookup[bin];
    if child_index == usize::MAX {
        (node.fallback_probabilities.as_slice(), None)
    } else {
        (&[], Some(child_index))
    }
}
```

Why this helps:

- branch lookup becomes one indexed access
- the runtime cost depends on bin value, not branch list length
- the fallback path is explicit and cheap

## Oblivious-tree runtime design

Oblivious trees are structurally different enough that they deserve their own runtime strategy.

Because each depth shares one split, scoring can be expressed as:

- a compact array of feature indices
- a compact array of thresholds
- a loop that accumulates a leaf index

Why this matters:

- the runtime becomes array-driven rather than pointer-driven
- prediction is more regular
- SIMD- and batch-oriented execution becomes more natural

This is why oblivious trees are such a good fit for optimized runtime lowering even though they are more constrained during training.

In practical terms, an oblivious tree is close to:

- one feature index per depth
- one threshold per depth
- one bit contribution per depth
- one leaf array indexed by the accumulated bit pattern

That regularity is what makes the current SIMD path feasible there but not yet equally profitable for arbitrary CART-style trees. Arbitrary trees diverge in shape; oblivious trees do not.

### Naïve oblivious evaluation as repeated tree traversal

```rust
fn predict_oblivious_naive(levels: &[Level], row: &[u16], leaves: &[f64]) -> f64 {
    let mut leaf_index = 0usize;
    for level in levels {
        let go_right = usize::from(row[level.feature_index] > level.threshold_bin);
        leaf_index = (leaf_index << 1) | go_right;
    }
    leaves[leaf_index]
}
```

Even this naïve version is already fairly regular, which is exactly why oblivious trees are attractive for fast inference.

### ForestFire optimized oblivious shape

```rust
fn predict_oblivious_optimized(
    feature_indices: &[usize],
    threshold_bins: &[u16],
    leaves: &[f64],
    projected_row: &[u16],
) -> f64 {
    let mut leaf_index = 0usize;
    for (&feature_index, &threshold_bin) in feature_indices.iter().zip(threshold_bins) {
        let bit = usize::from(projected_row[feature_index] > threshold_bin);
        leaf_index = (leaf_index << 1) | bit;
    }
    leaves[leaf_index]
}
```

This is the structure the batch and SIMD paths build on:

- feature indices are already projected
- thresholds are tightly packed
- the loop is regular enough to vectorize across rows

## Feature projection

Optimized models do not eagerly preprocess every feature from the semantic input schema.

Instead, the runtime:

- scans the semantic model for the feature indices that actually appear in splits
- computes the sorted union of those indices across the whole model or ensemble
- remaps the lowered runtime into that compact local feature space
- preprocesses only the projected columns during optimized scoring

Why this helps:

- wide inputs often carry many columns the model never touches
- skipping unused columns reduces binning and packing work before traversal starts
- the benefit compounds in forests and boosted ensembles because the same projected columns are reused many times

This is a runtime optimization only. The semantic model still expects the full input schema, and both semantic and optimized models expose the same prediction behavior.

Two details matter here:

### 1. Projection is computed from model semantics, not runtime guesses

ForestFire does not infer “used features” from observed inference traffic.

It computes them from the semantic model itself:

- scan all trees
- collect every feature index that appears in a split
- sort and deduplicate them

That means the projection is deterministic and exact with respect to the trained model.

### 2. Projection changes runtime feature indices, not the public schema

Suppose the semantic model uses features `[0, 4, 9]`.

The optimized runtime remaps them into local indices:

- semantic `0` -> runtime `0`
- semantic `4` -> runtime `1`
- semantic `9` -> runtime `2`

So runtime nodes refer to a dense local feature space, but callers still provide the original full feature set. This is important because it keeps the optimized model interoperable with the same user-facing API as the base model.

The user-facing metadata:

- `used_feature_indices`
- `used_feature_count`

exists so you can inspect that projected feature set directly.

### Naïve inference preprocessing

```rust
fn preprocess_all_features(rows: &[Vec<f64>], preprocessing: &[FeaturePreprocessing]) -> InferenceTable {
    let mut columns = Vec::new();
    for feature_index in 0..preprocessing.len() {
        columns.push(materialize_and_bin_feature(rows, feature_index, &preprocessing[feature_index]));
    }
    InferenceTable::new(columns)
}
```

This is simple, but it pays conversion and binning cost for every feature whether the model uses it or not.

### ForestFire projected preprocessing

```rust
fn preprocess_projected_features(
    rows: &[Vec<f64>],
    preprocessing: &[FeaturePreprocessing],
    projection: &[usize],
) -> InferenceTable {
    let mut columns = Vec::new();
    for &feature_index in projection {
        columns.push(materialize_and_bin_feature(
            rows,
            feature_index,
            &preprocessing[feature_index],
        ));
    }
    InferenceTable::new(columns)
}
```

Why this helps:

- preprocessing cost scales with used features, not semantic feature count
- projected feature indices are then dense and local
- the downstream runtime can use smaller batch matrices

## Batched preprocessing

Prediction cost is not just model traversal. It also includes turning user input into the internal binned representation.

ForestFire therefore preprocesses batches of rows together when possible.

Why:

- per-row conversion overhead dominates small or shallow-model workloads
- batching lets the runtime reuse feature metadata and conversion logic
- compact batch formats can then be fed directly into compiled runtimes

This is also why `polars.LazyFrame` prediction is batched in chunks instead of being materialized and traversed one row at a time.

The important implementation distinction is:

- single-row or tiny workloads prefer low overhead
- multi-row workloads benefit from a shared preprocessing step

ForestFire therefore keeps separate paths:

- a scalar row-oriented path
- a batch-oriented path for optimized runtimes

For `LazyFrame` specifically, the runtime does not try to collect the whole frame eagerly unless it is already small. Instead it slices the frame into chunks of about `10_000` rows, preprocesses each chunk, predicts it, and appends the results. That keeps memory use bounded while still letting the optimized runtime operate on batches.

### Naïve per-row scoring loop

```rust
fn predict_rows_naive(model: &Model, rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter()
        .map(|row| {
            let table = preprocess_projected_features(
                std::slice::from_ref(row),
                model.feature_preprocessing(),
                &(0..model.num_features()).collect::<Vec<_>>(),
            );
            model.predict_table(&table)[0]
        })
        .collect()
}
```

This repeats allocation, binning, and dispatch setup per row.

### ForestFire batch-oriented path

```rust
fn predict_rows_optimized(
    runtime: &OptimizedRuntime,
    rows: Vec<Vec<f64>>,
    preprocessing: &[FeaturePreprocessing],
    projection: &[usize],
    executor: &InferenceExecutor,
) -> Vec<f64> {
    let table = InferenceTable::from_rows_projected(rows, preprocessing, projection).unwrap();
    let matrix = table.to_column_major_binned_matrix();
    runtime.predict_column_major_matrix(&matrix, executor)
}
```

Why this helps:

- preprocessing happens once per batch
- the same projected columns are reused across many rows
- the runtime can pick chunked and SIMD-friendly execution from there

## Ensemble locality ordering

Forests and boosted ensembles also get a lightweight runtime-only ordering pass before lowering.

Today the ordering key is intentionally simple:

- primary/root split feature first
- then used-feature count
- then the full used-feature set as a stable tiebreaker

Why do this at all:

- nearby trees are more likely to touch the same projected feature columns
- top-level feature metadata and hot batch columns have a better chance of staying warm in cache
- the runtime can improve locality without changing the semantic model, IR, or predictions

This is not a semantic reorder. It only affects the lowered optimized runtime and compiled artifacts.

Today the ordering heuristic is intentionally conservative:

- root or primary split feature first
- then used-feature count
- then the full used-feature set as a stable tiebreaker

The reason to keep it simple is that it needs to be:

- deterministic
- cheap to compute
- obviously semantics-preserving

This is not trying to be a learned scheduling policy. It is a locality hint that is easy to reason about and easy to snapshot in the compiled artifact.

### Naïve ensemble lowering

```rust
fn lower_forest_naive(trees: &[Model], feature_map: &[usize]) -> Vec<OptimizedRuntime> {
    trees.iter()
        .map(|tree| OptimizedRuntime::from_model(tree, feature_map))
        .collect()
}
```

This preserves semantic order, but it does nothing to improve locality.

### ForestFire locality-aware lowering

```rust
fn lower_forest_locality_aware(trees: &[Model], feature_map: &[usize]) -> Vec<OptimizedRuntime> {
    let order = ordered_ensemble_indices(trees);
    order
        .into_iter()
        .map(|tree_index| OptimizedRuntime::from_model(&trees[tree_index], feature_map))
        .collect()
}
```

Why this helps:

- trees that touch similar top-level features end up near one another
- projected batch columns have a better chance of being reused while still hot
- the optimization is deterministic and semantics-preserving

## Compact `u8` / `u16` batches

The runtime stores batched bin ids as:

- `u8` when the realized feature bin domain fits
- `u16` only when it has to

This is a small design detail with large practical consequences:

- less memory bandwidth
- better cache density
- more rows processed per cache line

Because training already commits to bounded bins, the runtime can aggressively exploit that compactness.

Feature projection makes this more effective:

- if the model only uses a narrow subset of columns, the compact batch matrix only stores those columns
- smaller projected matrices improve both bandwidth and cache residency
- multiway lookup tables can also be right-sized to the actual realized bin domain

This is closely tied to the current auto-binning strategy:

- bins are chosen as powers of two
- the count is capped
- each realized bin must contain at least two rows

That makes the realized bin domain small enough that compact integer storage is useful not just as a theoretical option but as the default fast path.

### Naïve fixed-width batch storage

```rust
struct NaiveBatchMatrix {
    n_rows: usize,
    columns: Vec<Vec<u16>>,
}
```

This works, but it pays the wider storage cost even for binary or very small bin domains.

### ForestFire compact batch storage

```rust
enum CompactBinnedColumn {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

struct ColumnMajorBinnedMatrix {
    n_rows: usize,
    columns: Vec<CompactBinnedColumn>,
}
```

Why this helps:

- binary and small-cardinality projected columns stay narrow
- more rows fit in cache
- SIMD loads are cheaper for the narrow case

## Parallelism and SIMD

The optimized runtime currently uses CPU parallelism and SIMD where the structure supports it.

The main choices are:

- row-parallel chunking for large enough batches
- scalar fast paths for small workloads
- SIMD-oriented oblivious execution where the structure is regular enough

Why not apply the exact same SIMD strategy everywhere:

- arbitrary CART-style trees diverge quickly across rows
- oblivious trees keep the same split shape at every depth
- SIMD works best when many rows can execute the same instruction pattern

So the runtime is selective:

- exploit SIMD where the structure is regular
- exploit batching and locality where full SIMD is not yet worthwhile
- keep a cheap scalar path when the batch is too small to amortize setup costs

This matters because “optimized inference” is not one trick. It is the combination of:

- better layouts
- less preprocessing
- better memory density
- batching
- parallel chunking
- selective SIMD

### Naïve “always scalar” mindset

```rust
fn predict_batch_scalar(nodes: &[Node], matrix: &ColumnMajorBinnedMatrix) -> Vec<f64> {
    (0..matrix.n_rows)
        .map(|row_index| predict_one_scalar(nodes, matrix, row_index))
        .collect()
}
```

This is easy to reason about, but it misses the fact that some model families are regular enough to benefit from chunking or SIMD.

### ForestFire selective execution strategy

```rust
match runtime {
    OptimizedRuntime::ObliviousRegressor { .. } => {
        predict_oblivious_column_major_matrix(feature_indices, threshold_bins, leaves, matrix, executor)
    }
    OptimizedRuntime::BinaryRegressor { .. } => {
        predict_binary_regressor_column_major_matrix(nodes, matrix, executor)
    }
    _ => scalar_or_batch_fallback(...)
}
```

Why this helps:

- regular structures get specialized execution
- irregular structures keep cheaper generic paths
- the runtime does not force one strategy onto every tree family

## Compiled artifacts

The compiled artifact format exists for one reason: runtime lowering can itself be meaningful work.

If you repeatedly load the same optimized model, you do not want to:

- deserialize the semantic model
- recompute feature projection
- rebuild the lowered runtime layout
- reorder ensemble members again

every time.

So the compiled artifact stores:

- the semantic IR
- the projected feature set
- the lowered runtime layout

That gives you two useful guarantees:

- the semantic meaning is still preserved and inspectable
- reload can skip the expensive lowering step

This is why the compiled artifact is not a replacement for the IR. It is a cache of execution-oriented lowering built on top of the same semantic contract.

### Naïve reload strategy

```rust
let model = Model::deserialize(json)?;
let optimized = model.optimize_inference(Some(1))?;
```

This is fine when optimize time is negligible, but it repeats lowering work on every reload.

### ForestFire compiled artifact flow

```rust
let bytes = optimized.serialize_compiled()?;
let restored = OptimizedModel::deserialize_compiled(&bytes, Some(1))?;
```

Why this helps:

- semantic meaning is still available through the embedded IR
- the lowered runtime does not need to be rebuilt
- projection and locality-ordering decisions are preserved across reload

## Why the IR sits in the middle

The IR is the semantic source of truth for:

- serialization
- runtime lowering
- introspection

This has two major benefits.

First, it prevents divergence:

- optimized runtime code has to lower from an explicit semantic structure
- serialization has to preserve the same explicit structure
- introspection sees the same structure instead of hidden trainer internals

Second, it makes portability realistic:

- bin boundaries are explicit
- leaf payload meaning is explicit
- model family and structure are explicit

That is what makes the project’s “train once, introspect, serialize, optimize, and score” story coherent.

It also explains why optimized and non-optimized models deliberately export the same IR:

- optimization changes execution strategy
- it does not change the meaning of the model
- therefore the canonical semantic export should stay identical

That choice prevents a common failure mode in inference systems where the “fast path format” quietly becomes the only real model definition.

## Why introspection belongs next to serialization

At first glance, introspection looks like a developer convenience while serialization looks like a deployment feature.

In practice, they need the same thing:

- a stable, explicit account of model structure

That is why ForestFire can expose:

- `tree_structure(...)`
- `tree_node(...)`
- `tree_level(...)`
- `tree_leaf(...)`
- `to_dataframe(...)`

without inventing a separate debug-only representation.

The same semantic structure used for export is what powers the model-inspection features.

## Performance impact in practical terms

The point of these runtime design decisions is not theoretical purity. It is to change where time and memory go.

Typical wins come from:

- fewer allocations on the prediction path
- more compact working sets
- more regular access patterns
- better amortization of preprocessing work across batches
- less preprocessing spent on columns the model never uses
- better feature-column reuse across ensemble members

The largest benefits generally show up when:

- the same model serves many predictions
- batch sizes are moderate or large
- trees are deep enough that traversal overhead matters
- forests or boosted ensembles amplify any per-tree inefficiency

The smallest gains generally show up when:

- batches are tiny
- models are shallow
- input conversion dominates traversal cost

That is why the project exposes optimized inference explicitly: it is a useful specialization, but not a magic speedup in every workload.

The most important practical takeaway is this:

- if your workload is “score one tiny row once”, the semantic model is often good enough
- if your workload is “score many rows repeatedly with the same model”, the optimized runtime has much more room to pay for itself

That is also why the library keeps both views instead of pretending one runtime strategy is always best.
