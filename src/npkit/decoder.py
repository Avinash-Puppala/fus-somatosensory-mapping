"""
npkit.decoder — PCA + shrinkage-LDA decoder (NumPy only).

Replaces sklearn's PCA + LinearDiscriminantAnalysis from the original
src/decoder.py. Two reasons this is worth doing by hand:

  * removes the scikit-learn dependency (uninstallable in the sandbox), and
  * the LDA is written out explicitly with shrinkage, which is both more
    transparent to read and more robust than vanilla LDA when the number of
    features approaches the number of trials (the within-class covariance
    becomes singular and plain LDA blows up). Shrinkage pulls the covariance
    toward a scaled identity:  Sigma_reg = (1-lambda)*Sigma + lambda*mean_var*I.

The decoder fits PCA on training data only, then LDA in PCA space, exactly
mirroring the cross-validated structure of the original.
"""

from __future__ import annotations
import numpy as np


class PCA:
    """Minimal PCA via SVD. Centres on the training mean only."""

    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> "PCA":
        self.mean_ = X.mean(axis=0, keepdims=True)
        Xc = X - self.mean_
        # economy SVD; rows of Vt are principal axes
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:k]            # (k, n_features)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class ShrinkageLDA:
    """
    Multiclass Linear Discriminant Analysis with a shared, shrinkage-regularized
    within-class covariance. Equivalent to sklearn's LDA(solver='lsqr',
    shrinkage=lambda) but written out for clarity.

    Prediction uses the linear discriminant for each class k:
        g_k(x) = x^T Sigma^-1 mu_k - 0.5 mu_k^T Sigma^-1 mu_k + log(prior_k)
    and assigns argmax_k g_k(x).
    """

    def __init__(self, shrinkage: float = 0.1):
        self.shrinkage = shrinkage

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ShrinkageLDA":
        self.classes_ = np.unique(y)
        n, d = X.shape
        means, priors, Sw = [], [], np.zeros((d, d))
        for c in self.classes_:
            Xc = X[y == c]
            mu = Xc.mean(axis=0)
            means.append(mu)
            priors.append(len(Xc) / n)
            dev = Xc - mu
            Sw += dev.T @ dev
        Sw /= (n - len(self.classes_))                      # pooled within-class cov

        lam = self.shrinkage
        Sw_reg = (1 - lam) * Sw + lam * np.trace(Sw) / d * np.eye(d)

        self.means_ = np.array(means)                       # (C, d)
        self.priors_ = np.array(priors)                     # (C,)
        self.Sw_inv_ = np.linalg.pinv(Sw_reg)               # (d, d)
        # Precompute the linear coefficients and intercepts.
        self.coef_ = self.means_ @ self.Sw_inv_             # (C, d)
        self.intercept_ = (
            -0.5 * np.einsum("cd,cd->c", self.coef_, self.means_)
            + np.log(self.priors_)
        )
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_.T + self.intercept_           # (n, C)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


class LinearDecoder:
    """
    PCA -> shrinkage-LDA pipeline matching the Norman et al. method.

    fit(X2d, y) where X2d is (n_trials, n_voxels) already collapsed over time.
    voxel_importance() projects the LDA directions back to voxel space to give
    the weight map used for the somatotopic visualization.
    """

    def __init__(self, n_components: int = 20, shrinkage: float = 0.1):
        self.pca = PCA(n_components)
        self.lda = ShrinkageLDA(shrinkage)

    def fit(self, X2d: np.ndarray, y: np.ndarray) -> "LinearDecoder":
        Z = self.pca.fit_transform(X2d)
        self.lda.fit(Z, y)
        return self

    def predict(self, X2d: np.ndarray) -> np.ndarray:
        return self.lda.predict(self.pca.transform(X2d))

    def predict_with_margin(self, X2d: np.ndarray):
        """
        Return (predictions, margin). Margin = top1 - top2 of the LDA decision
        scores: a per-trial confidence the device can threshold on to decide
        whether to act (stimulate) or abstain on an uncertain/non-responsive trial.
        """
        scores = self.lda.decision_function(self.pca.transform(X2d))   # (n, C)
        order = np.sort(scores, axis=1)
        margin = order[:, -1] - order[:, -2]
        preds = self.lda.classes_[np.argmax(scores, axis=1)]
        return preds, margin

    def voxel_importance(self) -> np.ndarray:
        """L2 norm across classes of (LDA coef in PCA space) projected to voxels."""
        in_voxel = self.lda.coef_ @ self.pca.components_     # (C, n_voxels)
        return np.linalg.norm(in_voxel, axis=0)              # (n_voxels,)
