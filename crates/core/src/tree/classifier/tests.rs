use super::*;
use crate::{CanaryFilter, FeaturePreprocessing, Model, NumericBinBoundary, SplitStrategy};
use forestfire_data::{DenseTable, NumericBins};

fn and_table() -> DenseTable {
    DenseTable::new(
        vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    .unwrap()
}

fn criterion_choice_table() -> DenseTable {
    DenseTable::with_options(
        vec![
            vec![0.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        ],
        vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        0,
        NumericBins::Auto,
    )
    .unwrap()
}

fn canary_target_table() -> DenseTable {
    let x: Vec<Vec<f64>> = (0..8).map(|value| vec![value as f64]).collect();
    let probe = DenseTable::with_options(x.clone(), vec![0.0; 8], 1, NumericBins::Auto).unwrap();
    let canary_index = probe.n_features();
    let mut observed_bins = (0..probe.n_rows())
        .map(|row_idx| probe.binned_value(canary_index, row_idx))
        .collect::<Vec<_>>();
    observed_bins.sort_unstable();
    observed_bins.dedup();
    let threshold = observed_bins[observed_bins.len() / 2];
    let y = (0..probe.n_rows())
        .map(|row_idx| {
            if probe.binned_value(canary_index, row_idx) >= threshold {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    DenseTable::with_options(x, y, 1, NumericBins::Auto).unwrap()
}

fn canary_target_table_with_noise_feature() -> DenseTable {
    let x: Vec<Vec<f64>> = (0..8)
        .map(|value| vec![value as f64, (value % 2) as f64])
        .collect();
    let probe = DenseTable::with_options(x.clone(), vec![0.0; 8], 1, NumericBins::Auto).unwrap();
    let canary_index = probe.n_features();
    let mut observed_bins = (0..probe.n_rows())
        .map(|row_idx| probe.binned_value(canary_index, row_idx))
        .collect::<Vec<_>>();
    observed_bins.sort_unstable();
    observed_bins.dedup();
    let threshold = observed_bins[observed_bins.len() / 2];
    let y = (0..probe.n_rows())
        .map(|row_idx| {
            if probe.binned_value(canary_index, row_idx) >= threshold {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    DenseTable::with_options(x, y, 1, NumericBins::Auto).unwrap()
}

fn oblique_canary_target_table() -> DenseTable {
    let x = vec![
        vec![0.0, 9.0],
        vec![1.0, 4.0],
        vec![2.0, 7.0],
        vec![3.0, 2.0],
        vec![4.0, 8.0],
        vec![5.0, 1.0],
        vec![6.0, 6.0],
        vec![7.0, 3.0],
        vec![8.0, 5.0],
        vec![9.0, 0.0],
    ];
    let probe =
        DenseTable::with_options(x.clone(), vec![0.0; x.len()], 1, NumericBins::Fixed(32)).unwrap();
    let left_canary = probe.n_features();
    let right_canary = left_canary + 1;
    let y = (0..probe.n_rows())
        .map(|row_idx| {
            if probe.binned_value(left_canary, row_idx) <= probe.binned_value(right_canary, row_idx)
            {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();

    DenseTable::with_options(x, y, 1, NumericBins::Fixed(32)).unwrap()
}

fn randomized_permutation_table() -> DenseTable {
    DenseTable::with_options(
        vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 2.0],
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 2.0],
            vec![1.0, 1.0, 2.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        0,
        NumericBins::Fixed(8),
    )
    .unwrap()
}

fn oblique_classification_table() -> DenseTable {
    DenseTable::with_options(
        vec![
            vec![-2.0, 1.0],
            vec![1.0, -2.0],
            vec![-1.0, 2.0],
            vec![2.0, -1.0],
            vec![-3.0, 1.0],
            vec![1.0, -3.0],
            vec![-1.0, 3.0],
            vec![3.0, -1.0],
        ],
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        0,
        NumericBins::Fixed(64),
    )
    .unwrap()
}

fn greedy_vs_optimal_depth_two_table() -> DenseTable {
    DenseTable::with_options(
        vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        0,
        NumericBins::Fixed(8),
    )
    .unwrap()
}

#[test]
fn id3_fits_basic_boolean_pattern() {
    let table = and_table();
    let model = train_id3(&table).unwrap();

    assert_eq!(model.algorithm(), DecisionTreeAlgorithm::Id3);
    assert_eq!(model.criterion(), Criterion::Entropy);
    assert_eq!(model.predict_table(&table), table_targets(&table));
}

#[test]
fn c45_fits_basic_boolean_pattern() {
    let table = and_table();
    let model = train_c45(&table).unwrap();

    assert_eq!(model.algorithm(), DecisionTreeAlgorithm::C45);
    assert_eq!(model.criterion(), Criterion::Entropy);
    assert_eq!(model.predict_table(&table), table_targets(&table));
}

#[test]
fn cart_fits_basic_boolean_pattern() {
    let table = and_table();
    let model = train_cart(&table).unwrap();

    assert_eq!(model.algorithm(), DecisionTreeAlgorithm::Cart);
    assert_eq!(model.criterion(), Criterion::Gini);
    assert_eq!(model.predict_table(&table), table_targets(&table));
}

#[test]
fn randomized_fits_basic_boolean_pattern() {
    let table = and_table();
    let model = train_randomized(&table).unwrap();

    assert_eq!(model.algorithm(), DecisionTreeAlgorithm::Randomized);
    assert_eq!(model.criterion(), Criterion::Gini);
    assert_eq!(model.predict_table(&table), table_targets(&table));
}

#[test]
fn randomized_classifier_is_repeatable_for_fixed_seed_and_varies_across_seeds() {
    let table = randomized_permutation_table();
    let make_options = |random_seed| DecisionTreeOptions {
        max_depth: 4,
        max_features: Some(2),
        random_seed,
        ..DecisionTreeOptions::default()
    };

    let base_model = train_randomized_with_criterion_parallelism_and_options(
        &table,
        Criterion::Gini,
        Parallelism::sequential(),
        make_options(77),
    )
    .unwrap();
    let repeated_model = train_randomized_with_criterion_parallelism_and_options(
        &table,
        Criterion::Gini,
        Parallelism::sequential(),
        make_options(77),
    )
    .unwrap();
    let unique_serializations = (0..32u64)
        .map(|seed| {
            Model::DecisionTreeClassifier(
                train_randomized_with_criterion_parallelism_and_options(
                    &table,
                    Criterion::Gini,
                    Parallelism::sequential(),
                    make_options(seed),
                )
                .unwrap(),
            )
            .serialize()
            .unwrap()
        })
        .collect::<std::collections::BTreeSet<_>>();

    assert_eq!(
        Model::DecisionTreeClassifier(base_model.clone())
            .serialize()
            .unwrap(),
        Model::DecisionTreeClassifier(repeated_model)
            .serialize()
            .unwrap()
    );
    assert!(unique_serializations.len() >= 4);
}

#[test]
fn cart_classifier_supports_oblique_split_strategy() {
    let table = oblique_classification_table();
    let model = train_cart_with_criterion_parallelism_and_options(
        &table,
        Criterion::Gini,
        Parallelism::sequential(),
        DecisionTreeOptions {
            max_depth: 1,
            max_features: Some(2),
            split_strategy: SplitStrategy::Oblique,
            ..DecisionTreeOptions::default()
        },
    )
    .unwrap();

    assert_eq!(model.predict_table(&table), table_targets(&table));
    match model.structure() {
        TreeStructure::Standard { nodes, root } => {
            assert!(matches!(nodes[*root], TreeNode::ObliqueSplit { .. }));
        }
        TreeStructure::Oblivious { .. } => panic!("expected standard tree"),
    }
}

#[test]
fn randomized_classifier_supports_oblique_split_strategy() {
    let table = oblique_classification_table();
    let model = train_randomized_with_criterion_parallelism_and_options(
        &table,
        Criterion::Gini,
        Parallelism::sequential(),
        DecisionTreeOptions {
            max_depth: 1,
            max_features: Some(2),
            random_seed: 11,
            split_strategy: SplitStrategy::Oblique,
            ..DecisionTreeOptions::default()
        },
    )
    .unwrap();

    assert_eq!(model.predict_table(&table), table_targets(&table));
    match model.structure() {
        TreeStructure::Standard { nodes, root } => {
            assert!(matches!(nodes[*root], TreeNode::ObliqueSplit { .. }));
        }
        TreeStructure::Oblivious { .. } => panic!("expected standard tree"),
    }
}

#[test]
fn oblivious_fits_basic_boolean_pattern() {
    let table = and_table();
    let model = train_oblivious(&table).unwrap();

    assert_eq!(model.algorithm(), DecisionTreeAlgorithm::Oblivious);
    assert_eq!(model.criterion(), Criterion::Gini);
    assert_eq!(model.predict_table(&table), table_targets(&table));
}

#[test]
fn cart_can_choose_between_gini_and_entropy() {
    let table = criterion_choice_table();
    let options = DecisionTreeOptions {
        max_depth: 1,
        ..DecisionTreeOptions::default()
    };
    let gini_model = train_classifier(
        &table,
        DecisionTreeAlgorithm::Cart,
        Criterion::Gini,
        Parallelism::sequential(),
        options.clone(),
    )
    .unwrap();
    let entropy_model = train_classifier(
        &table,
        DecisionTreeAlgorithm::Cart,
        Criterion::Entropy,
        Parallelism::sequential(),
        options.clone(),
    )
    .unwrap();

    let root_feature = |model: &DecisionTreeClassifier| match &model.structure {
        TreeStructure::Standard { nodes, root } => match &nodes[*root] {
            TreeNode::BinarySplit { feature_index, .. } => *feature_index,
            node => panic!("expected binary root split, found {node:?}"),
        },
        TreeStructure::Oblivious { .. } => panic!("expected standard tree"),
    };

    assert_eq!(gini_model.criterion(), Criterion::Gini);
    assert_eq!(entropy_model.criterion(), Criterion::Entropy);
    assert_eq!(root_feature(&gini_model), 0);
    assert_eq!(root_feature(&entropy_model), 1);
}

#[test]
fn optimal_cart_depth_two_can_differ_from_greedy() {
    let table = greedy_vs_optimal_depth_two_table();
    let greedy = train_cart_with_criterion_parallelism_and_options(
        &table,
        Criterion::Gini,
        Parallelism::sequential(),
        DecisionTreeOptions {
            max_depth: 2,
            builder: BuilderStrategy::Greedy,
            ..DecisionTreeOptions::default()
        },
    )
    .unwrap();
    let optimal = train_cart_with_criterion_parallelism_and_options(
        &table,
        Criterion::Gini,
        Parallelism::sequential(),
        DecisionTreeOptions {
            max_depth: 2,
            builder: BuilderStrategy::Optimal,
            ..DecisionTreeOptions::default()
        },
    )
    .unwrap();

    let targets = table_targets(&table);
    let greedy_errors = greedy
        .predict_table(&table)
        .into_iter()
        .zip(targets.iter())
        .filter(|(pred, target)| pred != *target)
        .count();
    let optimal_errors = optimal
        .predict_table(&table)
        .into_iter()
        .zip(targets.iter())
        .filter(|(pred, target)| pred != *target)
        .count();

    assert_eq!(greedy_errors, 2);
    assert_eq!(optimal_errors, 0);

    let greedy_root_feature = match greedy.structure() {
        TreeStructure::Standard { nodes, root } => match &nodes[*root] {
            TreeNode::BinarySplit { feature_index, .. } => *feature_index,
            node => panic!("expected greedy binary root split, found {node:?}"),
        },
        TreeStructure::Oblivious { .. } => panic!("expected standard tree"),
    };
    let (optimal_root_feature, left_child, right_child) = match optimal.structure() {
        TreeStructure::Standard { nodes, root } => match &nodes[*root] {
            TreeNode::BinarySplit {
                feature_index,
                left_child,
                right_child,
                ..
            } => (*feature_index, *left_child, *right_child),
            node => panic!("expected optimal binary root split, found {node:?}"),
        },
        TreeStructure::Oblivious { .. } => panic!("expected standard tree"),
    };

    assert_eq!(greedy_root_feature, 0);
    assert_eq!(optimal_root_feature, 2);

    match optimal.structure() {
        TreeStructure::Standard { nodes, .. } => {
            let left_feature = match &nodes[left_child] {
                TreeNode::BinarySplit { feature_index, .. } => *feature_index,
                node => panic!("expected left child split, found {node:?}"),
            };
            let right_feature = match &nodes[right_child] {
                TreeNode::BinarySplit { feature_index, .. } => *feature_index,
                node => panic!("expected right child split, found {node:?}"),
            };

            assert_eq!(left_feature, 0);
            assert_eq!(right_feature, 1);
        }
        TreeStructure::Oblivious { .. } => panic!("expected standard tree"),
    }
}

#[test]
fn lookahead_matches_optimal_when_limits_cover_the_full_depth_two_search() {
    let table = greedy_vs_optimal_depth_two_table();
    let lookahead = train_cart_with_criterion_parallelism_and_options(
        &table,
        Criterion::Gini,
        Parallelism::sequential(),
        DecisionTreeOptions {
            max_depth: 2,
            builder: BuilderStrategy::Lookahead,
            lookahead_depth: 2,
            lookahead_top_k: 3,
            lookahead_weight: 1.0,
            beam_width: 1,
            ..DecisionTreeOptions::default()
        },
    )
    .unwrap();
    let optimal = train_cart_with_criterion_parallelism_and_options(
        &table,
        Criterion::Gini,
        Parallelism::sequential(),
        DecisionTreeOptions {
            max_depth: 2,
            builder: BuilderStrategy::Optimal,
            ..DecisionTreeOptions::default()
        },
    )
    .unwrap();

    assert_eq!(
        lookahead.predict_table(&table),
        optimal.predict_table(&table)
    );

    let lookahead_root_feature = match lookahead.structure() {
        TreeStructure::Standard { nodes, root } => match &nodes[*root] {
            TreeNode::BinarySplit { feature_index, .. } => *feature_index,
            node => panic!("expected lookahead binary root split, found {node:?}"),
        },
        TreeStructure::Oblivious { .. } => panic!("expected standard tree"),
    };
    let optimal_root_feature = match optimal.structure() {
        TreeStructure::Standard { nodes, root } => match &nodes[*root] {
            TreeNode::BinarySplit { feature_index, .. } => *feature_index,
            node => panic!("expected optimal binary root split, found {node:?}"),
        },
        TreeStructure::Oblivious { .. } => panic!("expected standard tree"),
    };

    assert_eq!(lookahead_root_feature, optimal_root_feature);
    assert_eq!(lookahead_root_feature, 2);
}

#[test]
fn rejects_non_finite_class_labels() {
    let table = DenseTable::new(vec![vec![0.0], vec![1.0]], vec![0.0, f64::NAN]).unwrap();

    let err = train_id3(&table).unwrap_err();
    assert!(matches!(
        err,
        DecisionTreeError::InvalidTargetValue { row: 1, value } if value.is_nan()
    ));
}

#[test]
fn stops_standard_tree_growth_when_a_canary_wins() {
    let table = canary_target_table();
    for trainer in [train_id3, train_c45, train_cart] {
        let model = trainer(&table).unwrap();
        let preds = model.predict_table(&table);

        assert!(preds.iter().all(|pred| *pred == preds[0]));
        assert_ne!(preds, table_targets(&table));
    }
}

#[test]
fn stops_oblivious_tree_growth_when_a_canary_wins() {
    let table = canary_target_table();
    let model = train_oblivious(&table).unwrap();
    let preds = model.predict_table(&table);

    assert!(preds.iter().all(|pred| *pred == preds[0]));
    assert_ne!(preds, table_targets(&table));
}

#[test]
fn top_n_canary_filter_can_choose_real_classifier_split() {
    let table = canary_target_table();
    let model = train_cart_with_criterion_parallelism_and_options(
        &table,
        Criterion::Gini,
        Parallelism::sequential(),
        DecisionTreeOptions {
            canary_filter: CanaryFilter::TopN(2),
            ..DecisionTreeOptions::default()
        },
    )
    .unwrap();
    let preds = model.predict_table(&table);

    assert!(preds.iter().any(|pred| *pred != preds[0]));
}

#[test]
fn top_fraction_canary_filter_can_choose_real_classifier_split() {
    let table = canary_target_table_with_noise_feature();
    let model = train_cart_with_criterion_parallelism_and_options(
        &table,
        Criterion::Gini,
        Parallelism::sequential(),
        DecisionTreeOptions {
            canary_filter: CanaryFilter::TopFraction(0.5),
            ..DecisionTreeOptions::default()
        },
    )
    .unwrap();
    let preds = model.predict_table(&table);

    assert!(preds.iter().any(|pred| *pred != preds[0]));
}

#[test]
fn oblique_canary_filter_blocks_oblique_growth_when_canary_pair_wins() {
    let table = oblique_canary_target_table();
    let model = train_cart_with_criterion_parallelism_and_options(
        &table,
        Criterion::Gini,
        Parallelism::sequential(),
        DecisionTreeOptions {
            canary_filter: CanaryFilter::TopN(1),
            split_strategy: SplitStrategy::Oblique,
            max_depth: 1,
            ..DecisionTreeOptions::default()
        },
    )
    .unwrap();
    let preds = model.predict_table(&table);

    assert!(preds.iter().all(|pred| *pred == preds[0]));
    assert_ne!(preds, table_targets(&table));
}

#[test]
fn oblique_classifier_routes_missing_features_independently() {
    let table = DenseTable::with_options(
        vec![
            vec![f64::NAN, 1.0],
            vec![1.0, f64::NAN],
            vec![f64::NAN, f64::NAN],
            vec![2.0, 2.0],
        ],
        vec![0.0; 4],
        0,
        NumericBins::Fixed(16),
    )
    .unwrap();
    let model = DecisionTreeClassifier {
        algorithm: DecisionTreeAlgorithm::Cart,
        criterion: Criterion::Gini,
        structure: TreeStructure::Standard {
            nodes: vec![
                TreeNode::Leaf {
                    class_index: 0,
                    sample_count: 2,
                    class_counts: vec![2.0, 0.0],
                },
                TreeNode::Leaf {
                    class_index: 1,
                    sample_count: 2,
                    class_counts: vec![0.0, 2.0],
                },
                TreeNode::ObliqueSplit {
                    feature_indices: vec![0, 1],
                    weights: vec![1.0, 0.5],
                    missing_directions: vec![
                        crate::tree::shared::MissingBranchDirection::Left,
                        crate::tree::shared::MissingBranchDirection::Right,
                    ],
                    threshold: 1.5,
                    left_child: 0,
                    right_child: 1,
                    sample_count: 4,
                    impurity: 0.0,
                    gain: 1.0,
                    class_counts: vec![2.0, 2.0],
                },
            ],
            root: 2,
        },
        options: DecisionTreeOptions::default(),
        class_labels: vec![0.0, 1.0],
        num_features: 2,
        feature_preprocessing: vec![
            FeaturePreprocessing::Numeric {
                bin_boundaries: vec![],
                missing_bin: 16,
            },
            FeaturePreprocessing::Numeric {
                bin_boundaries: vec![],
                missing_bin: 16,
            },
        ],
        training_canaries: 0,
    };

    assert_eq!(model.predict_table(&table), vec![0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn manually_built_classifier_models_serialize_for_each_tree_type() {
    let preprocessing = vec![
        FeaturePreprocessing::Binary,
        FeaturePreprocessing::Numeric {
            bin_boundaries: vec![
                NumericBinBoundary {
                    bin: 0,
                    upper_bound: 1.0,
                },
                NumericBinBoundary {
                    bin: 127,
                    upper_bound: 10.0,
                },
            ],
            missing_bin: 128,
        },
    ];
    let options = DecisionTreeOptions::default();
    let class_labels = vec![10.0, 20.0];

    let id3 = Model::DecisionTreeClassifier(DecisionTreeClassifier {
        algorithm: DecisionTreeAlgorithm::Id3,
        criterion: Criterion::Entropy,
        class_labels: class_labels.clone(),
        structure: TreeStructure::Standard {
            nodes: vec![
                TreeNode::Leaf {
                    class_index: 0,
                    sample_count: 3,
                    class_counts: vec![3.0, 0.0],
                },
                TreeNode::Leaf {
                    class_index: 1,
                    sample_count: 2,
                    class_counts: vec![0.0, 2.0],
                },
                TreeNode::MultiwaySplit {
                    feature_index: 1,
                    fallback_class_index: 0,
                    branches: vec![(0, 0), (127, 1)],
                    missing_child: None,
                    sample_count: 5,
                    impurity: 0.48,
                    gain: 0.24,
                    class_counts: vec![3.0, 2.0],
                },
            ],
            root: 2,
        },
        options: options.clone(),
        num_features: 2,
        feature_preprocessing: preprocessing.clone(),
        training_canaries: 0,
    });
    let c45 = Model::DecisionTreeClassifier(DecisionTreeClassifier {
        algorithm: DecisionTreeAlgorithm::C45,
        criterion: Criterion::Entropy,
        class_labels: class_labels.clone(),
        structure: TreeStructure::Standard {
            nodes: vec![
                TreeNode::Leaf {
                    class_index: 0,
                    sample_count: 3,
                    class_counts: vec![3.0, 0.0],
                },
                TreeNode::Leaf {
                    class_index: 1,
                    sample_count: 2,
                    class_counts: vec![0.0, 2.0],
                },
                TreeNode::MultiwaySplit {
                    feature_index: 1,
                    fallback_class_index: 0,
                    branches: vec![(0, 0), (127, 1)],
                    missing_child: None,
                    sample_count: 5,
                    impurity: 0.48,
                    gain: 0.24,
                    class_counts: vec![3.0, 2.0],
                },
            ],
            root: 2,
        },
        options: options.clone(),
        num_features: 2,
        feature_preprocessing: preprocessing.clone(),
        training_canaries: 0,
    });
    let cart = Model::DecisionTreeClassifier(DecisionTreeClassifier {
        algorithm: DecisionTreeAlgorithm::Cart,
        criterion: Criterion::Gini,
        class_labels: class_labels.clone(),
        structure: TreeStructure::Standard {
            nodes: vec![
                TreeNode::Leaf {
                    class_index: 0,
                    sample_count: 3,
                    class_counts: vec![3.0, 0.0],
                },
                TreeNode::Leaf {
                    class_index: 1,
                    sample_count: 2,
                    class_counts: vec![0.0, 2.0],
                },
                TreeNode::BinarySplit {
                    feature_index: 0,
                    threshold_bin: 0,
                    missing_direction: crate::tree::shared::MissingBranchDirection::Node,
                    left_child: 0,
                    right_child: 1,
                    sample_count: 5,
                    impurity: 0.48,
                    gain: 0.24,
                    class_counts: vec![3.0, 2.0],
                },
            ],
            root: 2,
        },
        options: options.clone(),
        num_features: 2,
        feature_preprocessing: preprocessing.clone(),
        training_canaries: 0,
    });
    let randomized = Model::DecisionTreeClassifier(DecisionTreeClassifier {
        algorithm: DecisionTreeAlgorithm::Randomized,
        criterion: Criterion::Entropy,
        class_labels: class_labels.clone(),
        structure: TreeStructure::Standard {
            nodes: vec![
                TreeNode::Leaf {
                    class_index: 0,
                    sample_count: 3,
                    class_counts: vec![3.0, 0.0],
                },
                TreeNode::Leaf {
                    class_index: 1,
                    sample_count: 2,
                    class_counts: vec![0.0, 2.0],
                },
                TreeNode::BinarySplit {
                    feature_index: 0,
                    threshold_bin: 0,
                    missing_direction: crate::tree::shared::MissingBranchDirection::Node,
                    left_child: 0,
                    right_child: 1,
                    sample_count: 5,
                    impurity: 0.48,
                    gain: 0.2,
                    class_counts: vec![3.0, 2.0],
                },
            ],
            root: 2,
        },
        options: options.clone(),
        num_features: 2,
        feature_preprocessing: preprocessing.clone(),
        training_canaries: 0,
    });
    let oblivious = Model::DecisionTreeClassifier(DecisionTreeClassifier {
        algorithm: DecisionTreeAlgorithm::Oblivious,
        criterion: Criterion::Gini,
        class_labels,
        structure: TreeStructure::Oblivious {
            splits: vec![ObliviousSplit {
                feature_index: 0,
                threshold_bin: 0,
                missing_directions: Vec::new(),
                sample_count: 4,
                impurity: 0.5,
                gain: 0.25,
            }],
            leaf_class_indices: vec![0, 1],
            leaf_sample_counts: vec![2, 2],
            leaf_class_counts: vec![vec![2.0, 0.0], vec![0.0, 2.0]],
        },
        options,
        num_features: 2,
        feature_preprocessing: preprocessing,
        training_canaries: 0,
    });

    for (tree_type, model) in [
        ("id3", id3),
        ("c45", c45),
        ("cart", cart),
        ("randomized", randomized),
        ("oblivious", oblivious),
    ] {
        let json = model.serialize().unwrap();
        assert!(json.contains(&format!("\"tree_type\":\"{tree_type}\"")));
        assert!(json.contains("\"task\":\"classification\""));
    }
}

#[test]
fn cart_classifier_assigns_training_missing_values_to_best_child() {
    let table = DenseTable::with_canaries(
        vec![
            vec![0.0],
            vec![0.0],
            vec![1.0],
            vec![1.0],
            vec![f64::NAN],
            vec![f64::NAN],
        ],
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        0,
    )
    .unwrap();

    let model = train_cart(&table).unwrap();

    assert_eq!(
        model.predict_table(&table),
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    );
    let wrapped = Model::DecisionTreeClassifier(model.clone());
    assert_eq!(
        wrapped.predict_rows(vec![vec![f64::NAN]]).unwrap(),
        vec![0.0]
    );
}

#[test]
fn cart_classifier_defaults_unseen_missing_to_node_majority() {
    let table = DenseTable::with_canaries(
        vec![vec![0.0], vec![0.0], vec![1.0]],
        vec![0.0, 0.0, 1.0],
        0,
    )
    .unwrap();

    let model = train_cart(&table).unwrap();

    let wrapped = Model::DecisionTreeClassifier(model.clone());
    assert_eq!(
        wrapped.predict_rows(vec![vec![f64::NAN]]).unwrap(),
        vec![0.0]
    );
}

fn table_targets(table: &dyn TableAccess) -> Vec<f64> {
    (0..table.n_rows())
        .map(|row_idx| table.target_value(row_idx))
        .collect()
}
