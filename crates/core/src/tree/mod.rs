//! Tree learners used across the project.
//!
//! `classifier` and `regressor` are the first-order learners used directly by
//! decision trees and random forests. `second_order` is the gradient/hessian-
//! driven learner used by gradient boosting.

pub mod classifier;
pub mod regressor;
pub mod second_order;
pub(crate) mod shared;
