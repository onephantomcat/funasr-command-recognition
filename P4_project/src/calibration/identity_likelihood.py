"""Class-Balanced Identity Likelihood Model (P4-15).

Trains a low-capacity logistic regression with class-balanced weights
to map identity evidence z_I to calibrated p_I^bal.

Key constraints:
  - Sum of positive weights == sum of negative weights
  - Scaler/imputer fit only on training fold
  - Missing values filled from training fold stats + missing indicator
  - No ASR features, labels, paths, or scene info in z_I
  - Output interpreted as class-balanced evidence (reference prior = 1/2)
  - Extreme probabilities clipped before logit transform
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Clipping constants
EPS_PROB = 1e-8
CLIP_PROB_MIN = 1e-6
CLIP_PROB_MAX = 1 - 1e-6


class IdentityCalibrator:
    """Class-balanced logistic regression for identity likelihood.

    Trained with equal total weight for positive and negative samples.
    Supports missing value imputation and optional Platt/beta calibration.
    """

    def __init__(self, l2_penalty: float = 1.0, fit_intercept: bool = True):
        self.l2_penalty = l2_penalty
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.scaler_mean_: Optional[np.ndarray] = None
        self.scaler_std_: Optional[np.ndarray] = None
        self.impute_values_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.fitted_ = False

    def _compute_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute class-balanced weights: Σw_pos = Σw_neg."""
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        if n_pos == 0 or n_neg == 0:
            return np.ones_like(y, dtype=np.float64)

        w_pos = 1.0 / (2.0 * n_pos) if n_pos > 0 else 1.0
        w_neg = 1.0 / (2.0 * n_neg) if n_neg > 0 else 1.0

        weights = np.where(y == 1, w_pos, w_neg)
        return weights

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        missing_mask: Optional[np.ndarray] = None,
    ) -> "IdentityCalibrator":
        """Fit the class-balanced logistic regression.

        Args:
            X: [N, D] feature matrix (may contain NaN for missing values).
            y: [N] binary labels (0 or 1). Target present = 1.
            missing_mask: [N, D] bool mask indicating missing values.

        Returns:
            self
        """
        N, D = X.shape
        self.n_features_in_ = D

        # Handle missing values
        if missing_mask is None:
            missing_mask = np.isnan(X) | np.isinf(X)

        # Compute imputation values from training data only
        self.impute_values_ = np.zeros(D)
        X_imputed = X.copy()
        for j in range(D):
            col_mask = missing_mask[:, j]
            if np.all(col_mask):
                self.impute_values_[j] = 0.0
            elif np.any(col_mask):
                self.impute_values_[j] = np.median(X[~col_mask, j])
            X_imputed[col_mask, j] = self.impute_values_[j]

        # Standardize (fit on training data only)
        self.scaler_mean_ = np.mean(X_imputed, axis=0)
        self.scaler_std_ = np.std(X_imputed, axis=0)
        self.scaler_std_[self.scaler_std_ < 1e-8] = 1.0
        X_scaled = (X_imputed - self.scaler_mean_) / self.scaler_std_

        # Add missing indicators
        X_aug = np.hstack([X_scaled, missing_mask.astype(np.float64)])

        # Compute class-balanced weights
        weights = self._compute_weights(y)

        # Logistic regression with L2 regularization via IRLS or SGD
        self._fit_logistic(X_aug, y, weights)
        self.fitted_ = True

        return self

    def _fit_logistic(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray):
        """Fit logistic regression with class-balanced weights using IRLS.

        Minimizes: -Σ w_i [y_i log(p_i) + (1-y_i) log(1-p_i)] + λ||β||^2
        """
        N, D = X.shape
        if self.fit_intercept:
            X_aug = np.hstack([np.ones((N, 1)), X])
        else:
            X_aug = X

        beta = np.zeros(X_aug.shape[1])
        lambda_reg = self.l2_penalty

        for iteration in range(100):
            eta = X_aug @ beta
            eta = np.clip(eta, -50, 50)
            p = 1.0 / (1.0 + np.exp(-eta))
            p = np.clip(p, CLIP_PROB_MIN, CLIP_PROB_MAX)

            # Gradient
            W_diag = weights * p * (1 - p)
            grad = X_aug.T @ (weights * (p - y)) + lambda_reg * beta
            grad[0] -= lambda_reg * beta[0]  # don't regularize intercept

            # Hessian approximation
            H = X_aug.T @ (X_aug * W_diag[:, np.newaxis]) + lambda_reg * np.eye(X_aug.shape[1])
            H[0, 0] -= lambda_reg  # don't regularize intercept

            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H, grad, rcond=None)[0]

            beta = beta - delta

            if np.max(np.abs(delta)) < 1e-6:
                break

        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = beta.copy()

    def predict_proba(self, X: np.ndarray, missing_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict class-balanced probability p_I^bal.

        Returns array of shape [N] in [0, 1].
        """
        if not self.fitted_:
            raise RuntimeError("Model not fitted")

        # Impute
        if missing_mask is None:
            missing_mask = np.isnan(X) | np.isinf(X)

        X_imputed = X.copy()
        for j in range(X.shape[1]):
            col_mask = missing_mask[:, j]
            if np.any(col_mask) and self.impute_values_ is not None:
                X_imputed[col_mask, j] = self.impute_values_[j]

        # Scale
        X_scaled = (X_imputed - self.scaler_mean_) / self.scaler_std_

        # Augment
        X_aug = np.hstack([X_scaled, missing_mask.astype(np.float64)])

        if self.fit_intercept:
            X_aug = np.hstack([np.ones((X_aug.shape[0], 1)), X_aug])
            beta = np.concatenate([[self.intercept_], self.coef_])
        else:
            beta = self.coef_

        eta = X_aug @ beta
        eta = np.clip(eta, -50, 50)
        prob = 1.0 / (1.0 + np.exp(-eta))
        return np.clip(prob, CLIP_PROB_MIN, CLIP_PROB_MAX)

    def predict_log_lr(self, X: np.ndarray, missing_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict log likelihood ratio log Λ_I = log(p / (1-p))."""
        prob = self.predict_proba(X, missing_mask)
        return np.log(np.clip(prob, EPS_PROB, 1 - EPS_PROB)
                      / np.clip(1 - prob, EPS_PROB, 1 - EPS_PROB))

    def check_balance(self, y: np.ndarray) -> dict:
        """Verify class-balanced weights sum equally."""
        weights = self._compute_weights(y)
        pos_w = np.sum(weights[y == 1])
        neg_w = np.sum(weights[y == 0])
        return {
            "total_weight": float(pos_w + neg_w),
            "pos_weight": float(pos_w),
            "neg_weight": float(neg_w),
            "balanced": abs(pos_w - neg_w) < 1e-10,
            "ratio": float(pos_w / neg_w) if neg_w > 0 else None,
        }
