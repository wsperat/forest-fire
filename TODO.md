# RF implementation plan

## 1. Add a forest trainer shell

Create a `RandomForestTrainer` layer above the existing tree trainers.

Why:

* RF mostly needs orchestration around trees
* keeps your current tree-growing code reusable
* gives one place for bootstrap, `n_trees`, `max_features`, OOB, seeds, etc.

Include at least:

* `n_trees`
* `max_features`
* `sample_rate` or classic full-size bootstrap
* `min_samples_leaf`, `min_samples_split`, `max_depth`
* `seed`
* optional `compute_oob`

---

## 2. Keep the current shared binned `Table`

Do not change the preprocessing model: bin once, train many trees from the same immutable structure.

Why:

* this already fits RF very well
* avoids repeated per-tree binning work
* keeps memory and implementation simpler

The forest should own or borrow:

* the shared binned `Table`
* target vector
* metadata about feature count / task type

---

## 3. Add bootstrap sampling per tree

Creeate a new `BootstrapSampler` object. For each tree, generate a sampled row-index vector.

Start with the simple version:

* sample `n_rows` indices with replacement
* store them in a `Vec<RowIdx>`

Why:

* minimal change to your current tree code
* duplicates naturally implement bootstrap
* easy to reason about
* easy to derive OOB rows later

Do not optimize this prematurely into weighted counts.

---

## 4. Represent nodes as ranges into a tree-local row-index buffer

Have each tree own one row-index array, then split nodes by partitioning that array in place.

Why:

* very efficient and standard for tree growth
* avoids allocating a fresh row list per node
* works naturally with bootstrap duplicates
* compatible with histogram/split search over row subsets

Each node can just be:

* `start`
* `end`

over the tree-local sampled row buffer.

---

## 5. Change split search to accept a feature subset

This is the key RF algorithm change.

Instead of always evaluating all columns at a node:

* sample `mtry` candidate features
* only search those features for the best split

Why:

* this is what makes RF “random forest” rather than bagged trees
* reduces tree correlation
* often reduces split-search cost too

Suggested defaults:

* classification: `sqrt(n_features)`
* regression: `n_features / 3`
* also allow exact integer and “all features”

---

## 6. Make training parallelism primarily tree-level

Shift the default parallelization strategy from “inside one tree across columns” to “across trees.”

Why:

* trees are independent
* much simpler scheduling
* avoids nested oversubscription with your current per-tree parallel split search
* usually gives better scaling for forests

Practical policy:

* parallelize across trees by default
* disable or strongly limit intra-tree column parallelism in RF mode
* only re-enable intra-tree parallelism later if profiling justifies a hybrid strategy

---

## 7. Add deterministic RNG plumbing

Use one forest seed, then derive a reproducible seed per tree.

Why:

* RF uses randomness everywhere
* parallel training should still be reproducible
* avoids schedule-dependent results

Use RNG for:

* bootstrap sampling
* feature subsampling at each node

Do not share one mutable RNG across threads.

---

## 8. Reuse existing tree training logic with RF-specific defaults

Keep the core split/grow logic the same, but use forest-appropriate defaults.

Why:

* RF wants deeper, noisier trees than standalone trees
* averaging handles variance later

Good initial defaults:

* no canaries
* no pruning
* deep or unlimited depth by default
* add `min_samples_leaf` and `min_samples_split` as the main safeguards

Think “grow strong and diverse trees,” not “grow one carefully regularized tree.”

---

## 9. Store the forest as a list of compiled prediction trees

After training each tree, lower it into your optimized prediction layout exactly as you already do for single trees.

Why:

* your inference stack is already strong
* RF prediction mainly needs aggregation on top
* no need to invent a separate tree runtime first

The forest model can just be:

* forest metadata
* list of compiled trees
* task-specific output metadata

---

## 10. Add forest-level prediction aggregation

Build a thin runtime that:

* preprocesses the batch once
* evaluates all trees
* combines outputs

Why:

* RF prediction is mostly “many trees + reduction”
* your current compiled runtimes already solve the hard part per tree

Aggregation:

* regression: average tree predictions
* classification: averaged class probabilities
* multiclass: accumulate per-class totals, then normalize/select argmax

---

## 11. Keep batch preprocessing once per input batch

Do not preprocess or bin per tree during prediction.

Why:

* all trees consume the same feature bins
* this is one of the biggest RF inference wins
* matches your current optimized multi-row inference design

Desired flow:

* input batch
* preprocess/bin once into compact column-major batch matrix
* score all trees against that matrix
* aggregate final outputs

---

## 12. Parallelize prediction across row blocks, not across trees

Keep prediction parallelism row-oriented.

Why:

* avoids partial-result reductions across threads
* fits your existing batch runtime design
* each worker can score all trees for its own row chunk and finalize locally

Tree-parallel inference is usually more awkward because it creates shared accumulation problems.

---

## 13. Add optional OOB bookkeeping

Once the basic RF works, add out-of-bag tracking.

Per tree, keep enough information to know which rows were not sampled.

Why:

* OOB score is one of the nicest RF features
* gives validation-like feedback “for free”
* useful later for feature importance and diagnostics

Start simple:

* build an in-bag count or OOB bitset per tree
* accumulate OOB predictions after training

This can be phase 2 if you want the first version smaller.

---

## 14. Only then optimize hotspots

After correctness is done, profile before adding complexity.

Likely future optimizations:

* hybrid tree/node parallelism
* histogram subtraction
* scratch-buffer reuse per worker
* tighter forest-level runtime loops
* better memory compaction for large forests

Why:

* RF is easy to over-engineer too early
* your current infrastructure is already strong enough for a good first implementation

---

## Recommended build order

If you want the shortest path to a usable RF:

1. forest params + trainer shell
2. bootstrap row sampling
3. `mtry` feature subsampling in split search
4. tree-level parallel training
5. compile each trained tree into existing inference layout
6. forest prediction aggregation
7. OOB support
8. profiling and optimization

That gets you to a practical, classical random forest without fighting your current design.
