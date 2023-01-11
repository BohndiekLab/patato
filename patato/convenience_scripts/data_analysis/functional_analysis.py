#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import h5py
import numpy as np
import skfda
from ...core.image_structures import image_structure_types
from scipy.linalg import solve_triangular
from skfda.misc.regularization import compute_penalty_matrix
from skfda.representation.basis import FDataBasis
from skfda.representation.grid import FDataGrid
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


def functional_k_means(xs, K=3, seed=None, maxit=1000):
    rng = np.random.default_rng(seed=seed)
    N = xs[0].shape[0]  # number of samples
    grpN = int(np.ceil(N / K))

    r = rng.random(N)
    p = r.argsort() // grpN
    p_prev = p
    assert np.max(p) < K
    residuals = np.zeros((K, N))
    for t in range(maxit):
        residuals[:] = 0
        for i in range(K):
            for x in xs:
                xp = x[p == i]
                centroid = xp.mean()
                res = x - centroid
                residuals[i] += np.sum((res.coefficients @ x.basis.gram_matrix()) * res.coefficients, axis=1)
        p = np.argmin(residuals, axis=0)
        if np.all(p == p_prev) or np.all(p == p[0]):
            # terminate
            break
        p_prev = p.copy()
    return p


def load_data(oe_file, dce_file, recon_num="0", dceunmixed_num="0ICG",
              bg_name="background_", umask_num = None):
    if umask_num is None:
        umask_num = recon_num
    recon_name = 'Backprojection Preclinical'
    spine_name = 'reference_'
    spine_num = '0'
    bg_num = '0'
    tumour_names = ['tumour_right', 'tumour_left', 'tumour_']
    tumour_num = '0'
    oeh5file = h5py.File(oe_file, "r")
    dceh5file = h5py.File(dce_file, "r")
    ICG = dceh5file["unmixed"][recon_name][dceunmixed_num][:, 2]
    so2 = oeh5file["so2"][recon_name][recon_num][:]
    dso2 = oeh5file["dso2"][recon_name][recon_num][:]
    dicg = dceh5file["dicg"][recon_name][dceunmixed_num][:]
    baseline_icg = dceh5file["baseline_icg"][recon_name][dceunmixed_num][:]
    sigma_icg = dceh5file["baseline_icg_sigma"][recon_name][dceunmixed_num][:]
    baseline_so2 = oeh5file["baseline_so2"][recon_name][recon_num][:]
    sigma_so2 = oeh5file["baseline_so2_sigma"][recon_name][recon_num][:]
    bg_mask = oeh5file["unmixed_masks"][recon_name][umask_num][bg_name][bg_num][:]  # bodge - might need to fix this
    spine_mask = oeh5file["unmixed_masks"][recon_name][umask_num][spine_name][spine_num][:]
    tumour_masks = []
    for tumour_name in tumour_names:
        if tumour_name in oeh5file["unmixed_masks"][recon_name][umask_num]:
            tumour_masks.append(oeh5file["unmixed_masks"][recon_name][umask_num][tumour_name][tumour_num][:])
    if len(tumour_masks) > 1:
        tumour_mask = np.logical_or(*tumour_masks)
    else:
        tumour_mask = tumour_masks[0]
    thb = np.sum(oeh5file["unmixed"][recon_name][recon_num][:], axis=1)
    steps_so2 = oeh5file["dso2"].attrs["steps"]
    steps_icg = dceh5file["dicg"].attrs["steps"]
    oeh5file.close()
    dceh5file.close()
    return so2, dso2, baseline_so2, sigma_so2, ICG, dicg, baseline_icg, \
           sigma_icg, thb, bg_mask, spine_mask, tumour_mask, steps_so2, steps_icg


def generate_map(mask, values, transpose=True):
    output = np.zeros(mask.shape) * np.nan
    if transpose:
        image_structure_types.T[image_structure_types.T] = values
    else:
        output[mask] = values
    return output


class MultiFPCA(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_components=3,
                 centering=True,
                 regularization=None,
                 weights=None,
                 components_basis=None
                 ):
        self.n_components = n_components
        self.centering = centering
        self.regularization = regularization
        self.weights = weights
        self.components_basis = None

    def _center_if_necessary(self, Xs, *, learn_mean=True):
        if learn_mean:
            self.mean_ = [X.mean() for X in Xs]
        if self.centering:
            output = []
            for X, m in zip(Xs, self.mean_):
                output.append(X - m)
            return output
        else:
            return Xs

    def _fit_basis(self, Xs, y=None):
        """Computes the first n_components principal components and saves them.
        The eigenvalues associated with these principal components are also
        saved. For more details about how it is implemented please view the
        referenced book.

        Args:
            X (FDataBasis):
                the functional data object to be analysed in basis
                representation
            y (None, not used):
                only present for convention of a fit function

        Returns:
            self (object)

        References:
            .. [RS05-8-4-2] Ramsay, J., Silverman, B. W. (2005). Basis function
                expansion of the functions. In *Functional Data Analysis*
                (pp. 161-164). Springer.

        """
        final_matrices = []
        l_matrices = []
        self._j_matrix = []
        self._X_basis = []
        Xs = self._center_if_necessary(Xs)
        for X in Xs:
            # the maximum number of components is established by the target basis
            # if the target basis is available.
            n_basis = (self.components_basis.n_basis if self.components_basis
                       else X.basis.n_basis)
            n_samples = X.n_samples

            # check that the number of components is smaller than the sample size
            if self.n_components > X.n_samples:
                raise AttributeError("The sample size must be bigger than the "
                                     "number of components")

            # check that we do not exceed limits for n_components as it should
            # be smaller than the number of attributes of the basis
            if self.n_components > n_basis:
                raise AttributeError("The number of components should be "
                                     "smaller than the number of attributes of "
                                     "target principal components' basis.")

            # if centering is True then subtract the mean function to each function
            # in FDataBasis

            # setup principal component basis if not given
            components_basis = self.components_basis
            if components_basis is not None:
                # First fix domain range if not already done
                components_basis = components_basis.copy(
                    domain_range=X.basis.domain_range)
                g_matrix = components_basis.gram_matrix()
                # the matrix that are in charge of changing the computed principal
                # components to target matrix is essentially the inner product
                # of both basis.
                j_matrix = X.basis.inner_product_matrix(components_basis)
            else:
                # if no other basis is specified we use the same basis as the passed
                # FDataBasis Object
                components_basis = X.basis.copy()
                g_matrix = components_basis.gram_matrix()
                j_matrix = g_matrix

            self._X_basis.append(X.basis)
            self._j_matrix.append(j_matrix)

            # Apply regularization / penalty if applicable
            regularization_matrix = compute_penalty_matrix(
                basis_iterable=(components_basis,),
                regularization_parameter=1,
                regularization=self.regularization)

            # apply regularization
            g_matrix = (g_matrix + regularization_matrix)

            # obtain triangulation using cholesky
            l_matrix = np.linalg.cholesky(g_matrix)
            l_matrices.append(l_matrix)
            # we need L^{-1} for a multiplication, there are two possible ways:
            # using solve to get the multiplication result directly or just invert
            # the matrix. We choose solve because it is faster and more stable.
            # The following matrix is needed: L^{-1}*J^T
            l_inv_j_t = solve_triangular(l_matrix, np.transpose(j_matrix),
                                         lower=True)

            # the final matrix, C(L-1Jt)t for svd or (L-1Jt)-1CtC(L-1Jt)t for PCA
            final_matrix_a = (X.coefficients @ np.transpose(l_inv_j_t) /
                              np.sqrt(n_samples))
            final_matrices.append(final_matrix_a)

        final_matrix = np.concatenate(final_matrices, axis=1)
        # initialize the pca module provided by scikit-learn
        pca = PCA(n_components=self.n_components)
        pca.fit(final_matrix)

        # we choose solve to obtain the component coefficients for the
        # same reason: it is faster and more efficient
        i = 0
        self.components_ = []
        for l_matrix, X in zip(l_matrices, Xs):
            component_coefficients = solve_triangular(np.transpose(l_matrix),
                                                      np.transpose(pca.components_[:, i:i + X.n_basis]),
                                                      lower=False)
            component_coefficients = np.transpose(component_coefficients)
            self.components_.append(X.copy(basis=X.basis.copy(),
                                           coefficients=component_coefficients,
                                           sample_names=(None,) * self.n_components))
            i += X.n_basis

        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        self.explained_variance_ = pca.explained_variance_
        return self

    def _transform_basis(self, Xs, y=None):
        """Computes the n_components first principal components score and
        returns them.

        Args:
            X (FDataBasis):
                the functional data object to be analysed
            y (None, not used):
                only present because of fit function convention

        Returns:
            (array_like): the scores of the data with reference to the
            principal components
        """
        output = []
        for X, jmat, comp, xbas in zip(Xs, self._j_matrix, self.components_, self._X_basis):
            if X.basis != xbas:
                raise ValueError("The basis used in fit is different from "
                                 "the basis used in transform.")
            output.append((X.coefficients @ jmat
                           @ image_structure_types.T))
        # in this case it is the inner product of our data with the components
        return sum(output)

    def fit(self, X, y=None):
        """Computes the n_components first principal components and saves them
        inside the FPCA object, both FDataGrid and FDataBasis are accepted

        Args:
            X (FDataGrid or FDataBasis):
                the functional data object to be analysed
            y (None, not used):
                only present for convention of a fit function

        Returns:
            self (object)
        """
        if isinstance(X[0], FDataBasis):
            return self._fit_basis(X, y)
        else:
            raise AttributeError("X must be either FDataGrid or FDataBasis")

    def transform(self, X, y=None):
        """Computes the n_components first principal components score and
        returns them.

        Args:
            X (FDataGrid or FDataBasis):
                the functional data object to be analysed
            y (None, not used):
                only present because of fit function convention

        Returns:
            (array_like): the scores of the data with reference to the
            principal components
        """

        X = self._center_if_necessary(X, learn_mean=False)

        if isinstance(X[0], FDataBasis):
            return self._transform_basis(X, y)
        else:
            raise AttributeError("X must be either FDataBasis")

    def fit_transform(self, X, y=None, **fit_params):
        """Computes the n_components first principal components and their scores
        and returns them.
        Args:
            X (FDataGrid or FDataBasis):
                the functional data object to be analysed
            y (None, not used):
                only present for convention of a fit function

        Returns:
            (array_like): the scores of the data with reference to the
            principal components
        """
        self.fit(X, y)
        return self.transform(X, y)


class MultiRegressor:
    def __init__(self, regressor, ipmat, **kwargs):
        self.reg_args = kwargs
        self.regressor = regressor
        self.regressors = []
        self.ipmat = ipmat
        self.coef_ = None

    def fit(self, model, Y):
        model = model @ self.ipmat
        self.coef_ = np.zeros((model.shape[1], Y.shape[1]))
        if len(self.regressors) == 0:
            self.regressors = [self.regressor(**self.reg_args) for _ in range(Y.shape[1])]
        for step, reg in enumerate(self.regressors):
            Y_train = Y[:, step:step + 1]
            reg.fit(model, Y_train.flatten())
            self.coef_[:, step] = reg.coef_
        return self

    def predict(self, model):
        model = model @ self.ipmat
        t = self.regressors[0].predict(model)
        output = np.zeros(t.shape + (len(self.regressors),))
        for step, reg in enumerate(self.regressors):
            output[:, step] = reg.predict(model)
        return output


def smooth(data, steps, knot_d=10):
    fd = skfda.FDataGrid(
        data_matrix=image_structure_types.T,
        grid_points=np.arange(data.shape[0]),
    )
    knots = []
    for i in range(len(steps) - 1):
        knots += list(np.arange(steps[i + 1], steps[i] + knot_d // 2, -knot_d)[::-1])
    if knots[0] != 0:
        knots = [0] + knots
    if knots[-1] != data.shape[0] - 1:
        knots += data.shape[0] - 1
    basis = skfda.representation.basis.BSpline(domain_range=(0, data.shape[0] - 1),
                                               knots=np.array(knots), order=3)
    fd_basis = fd.to_basis(basis)
    result = np.squeeze(fd_basis.evaluate(np.arange(data.shape[0]))).T
    return result.reshape(data.shape), fd_basis, basis


def frecl(x, y, K, regressor, maxit=300, seed=None):
    rng = np.random.default_rng(seed=seed)
    N = y.shape[0]  # number of samples = y.shape[0]
    grpN = int(np.ceil(N / K))

    r = rng.random(N)
    p = r.argsort() // grpN
    p_prev = p
    assert np.max(p) < K
    residuals = np.empty((K, y.shape[0]))
    converged = False
    for t in range(maxit):
        for i in range(K):
            x_p = x[p == i]
            y_p = y[p == i]
            model = regressor.fit(x_p, y_p)
            y_predict = model.predict(x)
            res = (y_predict - y) ** 2
            if len(res.shape) == 2:
                res = np.sum(res, axis=1)
            residuals[i] = res
        p = np.argmin(residuals, axis=0)
        if np.all(p == p_prev) or np.all(p == p[0]):
            # terminate
            converged = True
            break
        p_prev = p.copy()
    # Apply models
    y_predicted = np.zeros_like(y)
    models = []
    for i in range(K):
        x_p = x[p == i]
        y_p = y[p == i]
        model = regressor.fit(x_p, y_p)
        y_predicted[p == i] = model.predict(x_p)
        models.append(model.coef_.copy())
    return p, y_predicted, models, converged

def concensus_frecl(x, y, K, regressor, runs, seed=None):
    mat = np.zeros((x.shape[0], x.shape[0]))
    for i in range(runs):
        p, y_p, models, converged = frecl(x, y, K, regressor, seed=seed)
        if not converged:
            continue
        Al = p[:, None] == p[None, :]
        mat += Al*1.
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=K).fit(mat).labels_