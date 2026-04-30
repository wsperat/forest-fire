use super::*;
use crate::tree::shared::{
    FeatureHistogram, HistogramBin, build_feature_histograms, subtract_feature_histograms,
};

/// Per-bin class accumulator.
///
/// Counts are stored as `f64` so that sample weights can be accumulated
/// directly. With unit weights the values are exact integers; the arithmetic
/// is otherwise identical.
#[derive(Debug, Clone)]
pub(super) struct ClassificationHistogramBin {
    pub(super) counts: Vec<f64>,
}

impl ClassificationHistogramBin {
    pub(super) fn new(num_classes: usize) -> Self {
        Self {
            counts: vec![0.0f64; num_classes],
        }
    }

    pub(super) fn size(&self) -> f64 {
        self.counts.iter().sum()
    }
}

impl HistogramBin for ClassificationHistogramBin {
    fn subtract(parent: &Self, child: &Self) -> Self {
        Self {
            counts: parent
                .counts
                .iter()
                .zip(child.counts.iter())
                .map(|(parent, child)| parent - child)
                .collect(),
        }
    }

    fn is_observed(&self) -> bool {
        self.counts.iter().any(|count| *count > 0.0)
    }
}

pub(super) type ClassificationFeatureHistogram = FeatureHistogram<ClassificationHistogramBin>;

pub(super) fn build_classification_node_histograms(
    table: &dyn TableAccess,
    class_indices: &[usize],
    rows: &[usize],
    num_classes: usize,
) -> Vec<ClassificationFeatureHistogram> {
    build_feature_histograms(
        table,
        rows,
        |_| ClassificationHistogramBin::new(num_classes),
        |_feature_index, payload, row_idx| {
            payload.counts[class_indices[row_idx]] += table.sample_weight(row_idx);
        },
    )
}

pub(super) fn subtract_classification_node_histograms(
    parent: &[ClassificationFeatureHistogram],
    child: &[ClassificationFeatureHistogram],
) -> Vec<ClassificationFeatureHistogram> {
    subtract_feature_histograms(parent, child)
}
