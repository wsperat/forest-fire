use super::*;
use crate::tree::shared::{
    FeatureHistogram, HistogramBin, build_feature_histograms, subtract_feature_histograms,
};

#[derive(Debug, Clone)]
pub(super) struct ClassificationHistogramBin {
    pub(super) counts: Vec<usize>,
}

impl ClassificationHistogramBin {
    pub(super) fn new(num_classes: usize) -> Self {
        Self {
            counts: vec![0usize; num_classes],
        }
    }

    pub(super) fn size(&self) -> usize {
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
        self.counts.iter().any(|count| *count > 0)
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
            payload.counts[class_indices[row_idx]] += 1;
        },
    )
}

pub(super) fn subtract_classification_node_histograms(
    parent: &[ClassificationFeatureHistogram],
    child: &[ClassificationFeatureHistogram],
) -> Vec<ClassificationFeatureHistogram> {
    subtract_feature_histograms(parent, child)
}
