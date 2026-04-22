use super::*;

pub(crate) const PARALLEL_INFERENCE_ROW_THRESHOLD: usize = 256;
pub(crate) const PARALLEL_INFERENCE_CHUNK_ROWS: usize = 256;
pub(crate) const STANDARD_BATCH_INFERENCE_CHUNK_ROWS: usize = 4096;
pub(crate) const OBLIVIOUS_SIMD_LANES: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum OptimizedRuntime {
    BinaryClassifier {
        nodes: Vec<OptimizedBinaryClassifierNode>,
        class_labels: Vec<f64>,
    },
    StandardClassifier {
        nodes: Vec<OptimizedClassifierNode>,
        root: usize,
        class_labels: Vec<f64>,
    },
    ObliviousClassifier {
        feature_indices: Vec<usize>,
        threshold_bins: Vec<u16>,
        leaf_values: Vec<Vec<f64>>,
        class_labels: Vec<f64>,
    },
    BinaryRegressor {
        nodes: Vec<OptimizedBinaryRegressorNode>,
    },
    ObliviousRegressor {
        feature_indices: Vec<usize>,
        threshold_bins: Vec<u16>,
        leaf_values: Vec<f64>,
    },
    ForestClassifier {
        trees: Vec<OptimizedRuntime>,
        class_labels: Vec<f64>,
    },
    ForestRegressor {
        trees: Vec<OptimizedRuntime>,
    },
    BoostedBinaryClassifier {
        trees: Vec<OptimizedRuntime>,
        tree_weights: Vec<f64>,
        base_score: f64,
        class_labels: Vec<f64>,
    },
    BoostedRegressor {
        trees: Vec<OptimizedRuntime>,
        tree_weights: Vec<f64>,
        base_score: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum OptimizedClassifierNode {
    Leaf(Vec<f64>),
    Binary {
        feature_index: usize,
        threshold_bin: u16,
        children: [usize; 2],
        missing_bin: Option<u16>,
        missing_child: Option<usize>,
        missing_probabilities: Option<Vec<f64>>,
    },
    Multiway {
        feature_index: usize,
        child_lookup: Vec<usize>,
        max_bin_index: usize,
        missing_bin: Option<u16>,
        missing_child: Option<usize>,
        fallback_probabilities: Vec<f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum OptimizedBinaryClassifierNode {
    Leaf(Vec<f64>),
    Branch {
        feature_index: usize,
        threshold_bin: u16,
        jump_index: usize,
        jump_if_greater: bool,
        missing_bin: Option<u16>,
        missing_jump_index: Option<usize>,
        missing_probabilities: Option<Vec<f64>>,
    },
    ObliqueBranch {
        feature_indices: [usize; 2],
        weights: [f64; 2],
        threshold: f64,
        jump_index: usize,
        jump_if_greater: bool,
        missing_probabilities: Vec<f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum OptimizedBinaryRegressorNode {
    Leaf(f64),
    Branch {
        feature_index: usize,
        threshold_bin: u16,
        jump_index: usize,
        jump_if_greater: bool,
        missing_bin: Option<u16>,
        missing_jump_index: Option<usize>,
        missing_value: Option<f64>,
    },
    ObliqueBranch {
        feature_indices: [usize; 2],
        weights: [f64; 2],
        threshold: f64,
        jump_index: usize,
        jump_if_greater: bool,
        missing_value: f64,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct InferenceExecutor {
    pub(crate) thread_count: usize,
    pub(crate) pool: Option<Arc<rayon::ThreadPool>>,
}

impl InferenceExecutor {
    pub(crate) fn new(thread_count: usize) -> Result<Self, OptimizeError> {
        let pool = if thread_count > 1 {
            Some(Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(thread_count)
                    .build()
                    .map_err(|err| OptimizeError::ThreadPoolBuildFailed(err.to_string()))?,
            ))
        } else {
            None
        };

        Ok(Self { thread_count, pool })
    }

    pub(crate) fn predict_rows<F>(&self, n_rows: usize, predict_row: F) -> Vec<f64>
    where
        F: Fn(usize) -> f64 + Sync + Send,
    {
        if self.thread_count == 1 || n_rows < PARALLEL_INFERENCE_ROW_THRESHOLD {
            return (0..n_rows).map(predict_row).collect();
        }

        self.pool
            .as_ref()
            .expect("thread pool exists when parallel inference is enabled")
            .install(|| (0..n_rows).into_par_iter().map(predict_row).collect())
    }

    pub(crate) fn fill_chunks<F>(&self, outputs: &mut [f64], chunk_rows: usize, fill_chunk: F)
    where
        F: Fn(usize, &mut [f64]) + Sync + Send,
    {
        if self.thread_count == 1 || outputs.len() < PARALLEL_INFERENCE_ROW_THRESHOLD {
            for (chunk_index, chunk) in outputs.chunks_mut(chunk_rows).enumerate() {
                fill_chunk(chunk_index * chunk_rows, chunk);
            }
            return;
        }

        self.pool
            .as_ref()
            .expect("thread pool exists when parallel inference is enabled")
            .install(|| {
                outputs
                    .par_chunks_mut(chunk_rows)
                    .enumerate()
                    .for_each(|(chunk_index, chunk)| fill_chunk(chunk_index * chunk_rows, chunk));
            });
    }
}

pub(crate) fn resolve_inference_thread_count(
    physical_cores: Option<usize>,
) -> Result<usize, OptimizeError> {
    let available = num_cpus::get_physical().max(1);
    let requested = physical_cores.unwrap_or(available);

    if requested == 0 {
        return Err(OptimizeError::InvalidPhysicalCoreCount {
            requested,
            available,
        });
    }

    Ok(requested.min(available))
}
