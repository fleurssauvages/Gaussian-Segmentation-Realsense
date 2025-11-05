"""
gmm_splatting_params.py

Dependencies:
    pip install numpy scipy scikit-learn

Produces: saved npz file with arrays:
    mus: (M,3)
    covs: (M,3,3)
    weights: (M,)
    radii: (M,)   # Mahalanobis radii to use for each Gaussian
    optionally: colors, alphas
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2
from scipy.linalg import inv

# ---------- Utilities ----------
def regularize_cov(S, eps=1e-6):
    """Make covariance numerically stable."""
    S = np.array(S, dtype=float)
    S = (S + S.T) / 2.0
    diag_eps = eps * np.eye(S.shape[0])
    return S + diag_eps

def mahalanobis_sq(x, mu, invS):
    d = x - mu
    return np.einsum('ij,ij->i', d @ invS, d)

# ---------- Fit GMM ----------
def fit_gmm(X, K=32, covariance_type='full', random_state=0, reg_covar=1e-6, max_iter=200):
    """
    Fit a GMM and regularize covariances.
    Returns a fitted sklearn GaussianMixture object with regularized covariances.
    """
    gm = GaussianMixture(n_components=K,
                         covariance_type=covariance_type,
                         reg_covar=reg_covar,
                         max_iter=max_iter,
                         random_state=random_state,
                         init_params='kmeans')
    gm.fit(X)
    # regularize covariances manually for safety
    if covariance_type == 'full':
        gm.covariances_ = np.array([regularize_cov(S, eps=reg_covar) for S in gm.covariances_])
    elif covariance_type == 'diag':
        gm.covariances_ = np.array([max(v, reg_covar) for v in gm.covariances_])
    return gm

# ---------- Prune small components ----------
def prune_components(mus, covs, weights, min_weight=1e-3, min_points=10, X=None, assign=None):
    """
    Remove components with weights below min_weight or with fewer than min_points assigned.
    If X & assign (hard assignments) provided, we can check assigned points.
    """
    keep = []
    for k, w in enumerate(weights):
        if w < min_weight:
            continue
        if X is not None and assign is not None:
            nk = int((assign == k).sum())
            if nk < min_points:
                continue
        keep.append(k)
    mus = mus[keep]
    covs = covs[keep]
    weights = weights[keep]
    weights = weights / weights.sum()
    return mus, covs, weights

# ---------- Merge components ----------
def bhattacharyya_distance(mu1, S1, mu2, S2):
    """
    Symmetric-ish distance between two Gaussians (Bhattacharyya distance).
    Lower => more similar/overlapping.
    """
    S = 0.5 * (S1 + S2)
    invS = np.linalg.inv(regularize_cov(S))
    diff = mu1 - mu2
    term1 = 0.125 * diff.T @ invS @ diff
    term2 = 0.5 * 0.5 * np.log(np.linalg.det(regularize_cov(S)) / np.sqrt(np.linalg.det(regularize_cov(S1)) * np.linalg.det(regularize_cov(S2)) + 1e-20) + 1e-20)
    return float(term1 + term2)

def merge_pair(mu1, S1, w1, mu2, S2, w2):
    """Merge two Gaussians by moment matching (weighted mean and covariance)."""
    w = w1 + w2
    mu = (w1 * mu1 + w2 * mu2) / w
    # combined covariance = weighted covariances + spread of means
    S = (w1 * (S1 + np.outer(mu1 - mu, mu1 - mu)) + w2 * (S2 + np.outer(mu2 - mu, mu2 - mu))) / w
    return mu, S, w

def merge_components(mus, covs, weights, thresh=0.1):
    """
    Greedy merging: find pairs with Bhattacharyya distance < thresh and merge them.
    thresh controls how aggressive merging is (smaller -> stricter).
    """
    mus = list(mus)
    covs = list(covs)
    weights = list(weights)
    merged = True
    while merged:
        merged = False
        n = len(mus)
        if n < 2:
            break
        best = (None, None, 1e9)  # (i,j,dist)
        for i in range(n):
            for j in range(i+1, n):
                d = bhattacharyya_distance(mus[i], covs[i], mus[j], covs[j])
                if d < best[2]:
                    best = (i, j, d)
        if best[2] < thresh:
            i, j, d = best[0], best[1], best[2]
            mu_new, S_new, w_new = merge_pair(mus[i], covs[i], weights[i], mus[j], covs[j], weights[j])
            # replace i with merged, remove j
            mus[i] = mu_new
            covs[i] = regularize_cov(S_new)
            weights[i] = w_new
            # pop j
            mus.pop(j); covs.pop(j); weights.pop(j)
            # renormalize weights
            weights = list(np.array(weights) / np.sum(weights))
            merged = True
    return np.array(mus), np.array(covs), np.array(weights)

# ---------- Compute radii for splatting ----------
def compute_radii(X, mus, covs, responsibilities=None, method='probability', p=0.99, percentile=99.0):
    """
    method:
        - 'probability': use chi2 ppf (global p) -> r global or per component same value
        - 'percentile': per-component empirical percentile of Mahalanobis distances
        - 'max': per-component max distance (guaranteed)
    Returns radii array (len = n_components)
    """
    M = len(mus)
    radii = np.zeros(M, dtype=float)
    df = X.shape[1]
    if method == 'probability':
        r_global = np.sqrt(chi2.ppf(p, df))
        radii[:] = r_global
        return radii

    # need hard assignment or responsibilities
    if responsibilities is None:
        raise ValueError("responsibilities required for 'percentile' or 'max' methods")

    assign = responsibilities.argmax(axis=1)
    for k in range(M):
        idx = np.where(assign == k)[0]
        if idx.size == 0:
            # fallback to probability-based
            radii[k] = np.sqrt(chi2.ppf(p, df))
            continue
        Xk = X[idx]
        invS = np.linalg.inv(regularize_cov(covs[k]))
        d2 = mahalanobis_sq(Xk, mus[k], invS)
        if method == 'percentile':
            radii[k] = np.sqrt(np.percentile(d2, percentile))
        elif method == 'max':
            radii[k] = np.sqrt(d2.max())
    return radii

# ---------- Save / Load ----------
def save_gaussians(path, mus, covs, weights, radii, colors=None, alphas=None):
    np.savez_compressed(path, mus=mus, covs=covs, weights=weights, radii=radii,
                        colors=colors, alphas=alphas)

def load_gaussians(path):
    d = np.load(path)
    return d['mus'], d['covs'], d['weights'], d['radii'], d.get('colors'), d.get('alphas')

# ---------- Full pipeline function ----------
def build_gaussians_from_points(X, K=64, prune_weight=1e-3, prune_points=10,
                                merge_thresh=0.08, radius_method='percentile',
                                radius_p=0.99, radius_percentile=99.0, verbose=True,
                                outlier_removal=True, outlier_thresh=3.5,
                                clamp_scale_min=0.5, clamp_scale_max=100.0,
                                default_scale_when_empty=0.0):
    """
    Robust pipeline to build Gaussians from points.

    New/important params:
      - outlier_removal (bool): remove extreme global outliers before GMM fit
      - outlier_thresh (float): threshold for MAD-based outlier removal (higher -> fewer removed)
      - radius_method: 'probability' | 'percentile' | 'max' (keeps old options)
      - radius_percentile: used when radius_method=='percentile' (e.g. 95 or 99)
      - clamp_scale_min / clamp_scale_max: hard clamps on the covariance scaling factor `s`.
      - default_scale_when_empty: used when a component has no assigned points.

    Returns:
      mus, covs_scaled, weights, radii
    """

    # --- Optional global outlier removal (simple robust filter) ---
    if outlier_removal:
        # Use median absolute deviation per-dimension to detect extreme points
        X = np.asarray(X, dtype=float)
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0) + 1e-12
        # compute a robust "z" per point (L1 scaled)
        robust_z = np.max(np.abs(X - med) / mad, axis=1)
        keep_mask = robust_z < outlier_thresh
        if verbose:
            removed = (~keep_mask).sum()
            print(f"Outlier removal enabled: removed {removed} / {len(X)} points (threshold={outlier_thresh})")
        X_fit = X[keep_mask]
    else:
        X_fit = np.asarray(X, dtype=float)

    if X_fit.shape[0] < 3:
        raise ValueError("Not enough points to fit GMM after outlier removal.")

    # 1) Fit GMM on cleaned points
    gm = fit_gmm(X_fit, K=K, covariance_type='full', reg_covar=1e-6)
    mus = gm.means_
    covs = gm.covariances_
    weights = gm.weights_
    if verbose:
        print(f"Fitted GMM with {len(mus)} components on {X_fit.shape[0]} points.")

    # responsibilities & hard assignment (for the cleaned points)
    resp = gm.predict_proba(X_fit)
    assign = resp.argmax(axis=1)

    # 2) prune
    mus, covs, weights = prune_components(mus, covs, weights, min_weight=prune_weight,
                                         min_points=prune_points, X=X_fit, assign=assign)
    if verbose:
        print(f"After pruning: {len(mus)} components.")

    # recompute responsibilities/assign for pruned set (optional safety)
    # Recompute resp/assign for the pruned components to use for percentile calculations:
    if len(mus) != gm.means_.shape[0]:
        # build a small GMM object to recompute responsibilities (cheap)
        # We'll compute responsibilities manually using Gaussian pdf formulas:
        from scipy.stats import multivariate_normal
        M = len(mus)
        # ensure covs are regularized
        covs = np.array([regularize_cov(S) for S in covs])
        resp = np.zeros((X_fit.shape[0], M), dtype=float)
        for k in range(M):
            try:
                mv = multivariate_normal(mean=mus[k], cov=covs[k], allow_singular=True)
                resp[:, k] = weights[k] * mv.pdf(X_fit)
            except Exception:
                # fallback to small constant if something fails
                resp[:, k] = weights[k] * 1e-12
        # normalize responsibilities
        denom = resp.sum(axis=1, keepdims=True) + 1e-12
        resp = resp / denom
        assign = resp.argmax(axis=1)

    # 3) merge (optional)
    if merge_thresh is not None and merge_thresh > 0:
        mus, covs, weights = merge_components(mus, covs, weights, thresh=merge_thresh)
        if verbose:
            print(f"After merging: {len(mus)} components.")

    # recompute responsibilities/assign after merge for final scaling
    # we approximate responsibilities again (same method)
    from scipy.stats import multivariate_normal
    M = len(mus)
    covs = np.array([regularize_cov(S) for S in covs])
    resp = np.zeros((X_fit.shape[0], M), dtype=float)
    for k in range(M):
        try:
            mv = multivariate_normal(mean=mus[k], cov=covs[k], allow_singular=True)
            resp[:, k] = weights[k] * mv.pdf(X_fit)
        except Exception:
            resp[:, k] = weights[k] * 1e-12
    resp = resp / (resp.sum(axis=1, keepdims=True) + 1e-12)
    assign = resp.argmax(axis=1)

    # 4) compute radii (unchanged API)
    radii = compute_radii(X_fit, mus, covs, responsibilities=resp, method=radius_method,
                          p=radius_p, percentile=radius_percentile)
    if verbose:
        print("Computed radii (first 10):", radii[:10])

    # 5) Rescale covariances to enclose data using percentile and clamp scales
    M = len(mus)
    covs_scaled = np.empty_like(covs)
    scales = np.ones(M, dtype=float)

    for k in range(M):
        mu = mus[k]
        S = regularize_cov(covs[k])
        idx = np.where(assign == k)[0]
        if idx.size == 0:
            # no assigned points -> keep original covariance
            covs_scaled[k] = S * default_scale_when_empty
            scales[k] = default_scale_when_empty
            continue

        Xk = X_fit[idx]
        # compute Mahalanobis squared distances using inv(S)
        try:
            invS = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            invS = inv(S + 1e-8 * np.eye(S.shape[0]))

        dif = Xk - mu  # (n_k,3)
        d2 = np.einsum('ij,ij->i', dif @ invS, dif)  # squared Mahalanobis

        if radius_method == 'max':
            s = float(d2.max())
        elif radius_method == 'probability':
            # keep existing chi2-based scaling (global)
            s = float(chi2.ppf(radius_p, X.shape[1]))
        else:  # percentile (recommended)
            s = float(np.percentile(d2, radius_percentile))

        # ensure no tiny or huge scaling
        s_clamped = float(np.clip(s, clamp_scale_min, clamp_scale_max))
        scales[k] = s_clamped
        covs_scaled[k] = S * s_clamped

    if verbose:
        print("Scale stats — min:", scales.min(), "median:", np.median(scales), "max:", scales.max())

    return mus, covs_scaled, weights, radii
