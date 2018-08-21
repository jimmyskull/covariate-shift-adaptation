"""
Importance Estimation methods.
"""
import logging
import collections
import sys
from copy import deepcopy

import numpy as np
import torch
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


from .torch_utils import pairwise_distances_squared, gaussian_kernel


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PCIF(BaseEstimator, ClassifierMixin):
    r"""
    Probabilistic Classifier Importance Fitting.

    Trains a probabilistic classifier to distinguish between samples
    from training and test distributions. Then given a feature vector
    :math:`x`, we can use the trained classifier along with Bayes' rule
    to estimate the probability density ratio :math:`w(x)` as follows:

    .. math::

        w(x) = \frac{n_{train} \cdot p(test|x)}
                    {n_{test} \cdot p(train|x)},

    where :math:`n_{train}` and :math:`n_{test}` are the number of
    training and test samples used to fit the model respectively, and
    :math:`p(train|x)` and :math:`p(test|x)` are the probabilities that
    :math:`x` was sampled from the training and test distributions
    respectively, as predicted by the trained classifier.

    Attributes
    ----------
    n_train_ : int
        Number of samples from training distribution used to fit the
        model.

    n_test_ : int
        Number of samples from test distribution used to fit the model.

    estimator_ : estimator object
        Fitted probabilistic classifier.

    cv_results_ :

    best_score_ :

    best_params_ :
    """

    def fit(self, train_data, test_data, estimator=None):
        """
        Fit a probabilistic classifier to the input data.

        Fits a probabilistic classifier to the input training and test
        data to predict :math:`p(test|x)`.

        - If an estimator with the scikit-learn interface is provided,
        this estimator is fit to the data.

        - If an object that inherits
          :class:`sklearn.model_selection.BaseSearchCV` is
          provided, model selection is run and the best estimator is
          subsequently fit to all the data.

        Parameters
        ----------
        train_data : array-like, shape = [n_samples, n_features]
            Input data from training distribution, where each row is a
            feature vector.

        test_data : array-like, shape = [n_samples, n_features]
            Input data from test distribution, where each row is a
            feature vector.

        estimator : estimator object
            If estimator, assumed to implement the scikit-learn
            estimator interface.
        """
        # Construct the target (1 if test, 0 if train).
        self.n_train_ = train_data.shape[0]
        self.n_test_ = test_data.shape[0]
        n = self.n_train_ + self.n_test_
        y = np.concatenate((np.zeros(self.n_train_), np.ones(self.n_test_)))

        # Stack and shuffle features and target.
        X = np.vstack((train_data, test_data))
        i_shuffle = np.random.choice(n, n, replace=False)
        X = X[i_shuffle]
        y = y[i_shuffle]

        # Fit estimator.
        if isinstance(estimator, BaseSearchCV):
            logging.info('Running model selection...')
            if estimator.refit == False:
                estimator.refit = True
            estimator.fit(X, y)
            logging.info('Best score = {}'.format(estimator.best_score_))
            self.cv_results_ = estimator.cv_results_
            self.estimator_ = estimator.best_estimator_
            self.best_score_ = estimator.best_score_
            self.best_params_ = estimator.best_params_
            logging.info('Done!')
        else:
            logging.info('Fitting estimator...')
            self.estimator_ = estimator.fit(X, y)
            logging.info("Done!")


    def predict(self, X):
        r"""
        Estimate importance weights for input data.

        For each feature vector :math:`x`, the trained probabilistic
        classifier is used to estimate the probability density ratio

        .. math::

            w(x) = \frac{n_{train} \cdot p(test|x)}
                        {n_{test} \cdot p(train|x)},

        where :math:`n_{train}` and :math:`n_{test}` are the number of
        training and test samples used to train the model respectively,
        and :math:`p(train|x)` and :math:`p(test|x)` are the
        probabilities that x was sampled from the training and test
        distributions respectively, as predicted by the trained
        classifier.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input data, where each row is a feature vector.

        Returns
        -------
        w : ndarray, shape (len(X),)
            Estimated importance weight for each input.
            w[i] corresponds to importance weight of X[i]
        """
        check_is_fitted(self, 'estimator_')
        p = self.estimator_.predict_proba(X)
        w = importance_weights(p, self.n_train_, self.n_test_)
        return w

    def fit_predict(self, train_data, test_data, X, estimator=None):
        self.fit(estimator, train_data, test_data)
        w = self.predict(X)
        return w

    def predict_oos(self, train_data, test_data, estimator=None, n_splits=5):
        if estimator is None:
            check_is_fitted(
                self,
                'estimator_',
                msg='Either provide an estimator or run fit method first!')
            estimator = deepcopy(self.estimator_)

        # stack features and construct the target (1 if test, 0 if train)
        X = np.vstack((train_data, test_data))
        y = np.concatenate((np.zeros(train_data.shape[0]),
                            np.ones(test_data.shape[0])))

        # split the data into n_splits folds, and for 1 to n_splits,
        # train on (n_splits - 1)-folds and use the fitted estimator to
        # predict weights for the other fold
        w = np.zeros_like(y)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        for idtrain_data, idtest_data in skf.split(X, y):

            # fit on (n_splits - 1)-folds
            estimator.fit(X[idtrain_data], y[idtrain_data])

            # predict probabilities for other fold
            p = estimator.predict_proba(X[idtest_data])

            # predict weights for other fold
            n_tr = (y[idtrain_data] == 0).sum()
            n_te = (y[idtrain_data] == 1).sum()
            w[idtest_data] = importance_weights(p, n_tr, n_te)

        # split into training and test weights
        w_tr = w[y == 0]
        w_te = w[y == 1]

        return w_tr, w_te


def importance_weights(p, n_train, n_test, logits=False):
    if len(p.shape) > 1:
        p = p[:, 1]
    if logits:
        w = (n_train / n_test) * np.exp(p)
    else:
        w = (n_train / n_test) * (p / (1 - p))
    return w


class uLSIF(object):
    r"""
    Unconstrained Least Squares Importance Fitting (uLSIF).

    Implementation of uLSIF algorithm as described in
    Machine Learning in Non-Stationary Environments - Introduction to
    Covariate Shift Adaption,
    M. Sugiyama and M. Kawanabe, 2012.

    Gaussian kernel basis functions are fit to samples from training
    and test distributions to approximate the probability density ratio

    .. math::

        w(x) = \frac{p_{test}(x)}{p_{train}(x)},

    where :math:`p_{train}(x)` and :math:`p_{test}(x)` are the
    probabilities that the feature vector :math:`x` comes from the
    training and test distributions respectively. The fitting is done
    through minimisation of the squared-loss between the model and
    the true probability density ratio function.

    Once fitted the model can be used to estimate the probability
    density ratio, or importance, of any :math:`x`.

    Parameters
    ----------
    n_kernels : int, default: 100
        Number of Guassian kernels to use in the model.

    Attributes
    ----------
    C_ : torch tensor
        Kernel centres, where each row is a randomly chosen sample from
        the test distribution.

    alpha_ : torch tensor
        Coefficients of fitted model.

    sigma_ : scalar
        Kernel width of fitted model.
    """

    def __init__(self, n_kernels=100):
        # parameters
        self.n_kernels = n_kernels
        # attributes
        self.C_ = None
        self.alpha_ = None
        self.sigma_ = None

    def fit(self, train_data, test_data, sigma, lam, random_seed=42):
        """
        Fit the model to the input training and test data.

        Gaussian kernel basis functions are fit to the data by
        minimising the squared-loss between the model and the true
        probability density ratio function.

        - If scalars provided for both kernel width (``sigma``) and
          regularisation strength (``lam``), the model with these
          hyperparameters is fit to the data.

        - If more than one value provided for either of the
          hyperparameters, a hyperparameter search is performed via
          leave-on-out cross-validation and the best parameters are
          used to fit the model.

        Parameters
        ----------
        train_data : array-like, shape = [n_samples, n_features]
            Input data from training distribution, where each row is a
            feature vector.

        test_data: array-like, shape = [n_samples, n_features]
            Input data from test distribution, where each row is a
            feature vector.

        sigma: scalar or iterable
            Gaussian kernel width. If iterable, hyperparameter search
            will be run.

        lam: scalar or iterable
            Regularisation strength. If iterable, hyperparameter search
            will be run.

        random_seed: int, RandomState instance or None, default: None
            The seed of the pseudo random number generator to use when
            sampling the data. TODO: is this correct?
        """
        with torch.no_grad():
            np.random.seed(random_seed)

            # Convert training and test data to torch tensors
            train_data = torch.from_numpy(train_data).float().to(DEVICE)
            test_data = torch.from_numpy(test_data).float().to(DEVICE)

            # Randomly choose kernel centres from test_data without
            # replacement.
            n_te = test_data.size(0)
            t = min(self.n_kernels, test_data.size(0))
            self.C_ = test_data[np.random.choice(n_te, t, replace=False)]
            # shape (t, d)

            # Compute the squared l2-norm of the difference between
            # every point in train_data and every point in C,
            # element (l, i) should contain the squared l2-norm
            # between C[l] and train_data[i].
            logging.info('Computing distance matrix for train_dataain...')
            D_tr = pairwise_distances_squared(self.C_, train_data) # shape (t, n_tr)

            # do the same for test_data
            logging.info('Computing distance matrix for test_datast...')
            D_te = pairwise_distances_squared(self.C_, test_data) # shape (t, n_te)

            # check if we need to run a hyperparameter search
            search_sigma = isinstance(sigma, (collections.Sequence, np.ndarray)) and \
                            (len(sigma) > 1)
            search_lam = isinstance(lam, (collections.Sequence, np.ndarray)) and \
                            (len(lam) > 1)
            if search_sigma | search_lam:
                logging.info('Running hyperparameter search...')
                sigma, lam = self.loocv(train_data, D_tr, test_data, D_te, sigma, lam)
            else:
                if isinstance(sigma, (collections.Sequence, np.ndarray)):
                    sigma = sigma[0]
                if isinstance(lam, (collections.Sequence, np.ndarray)):
                    lam = lam[0]

            logging.info('Computing optimal solution...')
            train_data = gaussian_kernel(D_tr, sigma)  # shape (t, n_tr)
            test_data = gaussian_kernel(D_te, sigma) # shape (t, n_te)
            H, h = self.kernel_arrays(train_data, test_data) # shapes (t, t) and (t, 1)
            alpha = (H + (lam * torch.eye(t)).to(DEVICE)).inverse().mm(h) # shape (t, 1)
            self.alpha_ = torch.max(torch.zeros(1).to(DEVICE), alpha) # shape (t, 1)
            self.sigma_ = sigma
            logging.info('Done!')


    def predict(self, X):
        r"""
        Estimate importance weights for input data.

        For each feature vector :math:`x`, uses the fitted model to
        estimate the probability density ratio

        .. math::

            w(x) = \frac{p_{test}(x)}{p_{train}(x)},

        where :math:`p_{train}` is the probability density of the
        training distribution and :math:`p_{test}` is the probability
        density of the test distribution.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input data from training distribution, where each row is a
            feature vector.

        Returns
        -------
        w : vector of shape (len(X),)
            Estimated importance weight for each input.
            w[i] corresponds to importance weight of X[i]
        """
        with torch.no_grad():
            assert self.alpha_ is not None, "Need to run fit method before calling predict!"

            # convert data to torch tensors
            X = torch.from_numpy(X).float().to(DEVICE)

            # compute the squared l2-norm of the difference between
            # every point in X and every point in C,
            # element (l, i) should contain the squared l2-norm
            # between C[l] and X[i]
            D = pairwise_distances_squared(self.C_, X) # shape (t, n)

            # compute gaussian kernel
            X = gaussian_kernel(D, self.sigma_)  # shape (t, n_tr)

            # compute importance weights
            w = self.alpha_.t().mm(X).squeeze().cpu().numpy() # shape (n_tr,)

        return w


    def loocv(self, train_data, D_tr, test_data, D_te, sigma_range, lam_range):
        """
        Hyperprameter search via leave-one-out cross-validation (LOOCV).

        Computes LOOCV squared-loss for every combination of the
        Guassian kernel width and regularisation strength and returns
        the parameters which correspond to the smallest loss.

        Parameters
        ----------
        train_data : torch tensor
            Input data from training distribution, where each row is a
            feature vector.

        D_tr : torch tensor
            Squared l2-norm of the difference between every kernel
            centre and every row in train_data.
            Element ``(l, i)`` should contain the squared l2-norm
            between the *l*-th kernel centre and ``train_data[i]``

        test_data : torch tensor
            Input data from test distribution, where each row is a
            feature vector.

        D_te : torch tensor
            Squared l2-norm of the difference between every kernel
            centre and every point in test_data.
            Element ``(l, i)`` should contain the squared l2-norm
            between the *l*-th kernel centre and ``test_data[i]``.

        sigma_range : scalar or iterable
            Guassian kernel width. If scalar will be converted to list.

        lam_range : scalar or iterable
            Regularisation strength. If scalar will be converted to
            list.

        Returns
        -------
        sigma_hat : scalar
            Guassian kernel width corresponding to lowest LOOCV loss.

        lam_hat : scalar
            Regularisation strength corresponding to lowest LOOCV loss.
        """
        with torch.no_grad():

            # make sure hyperparameter ranges are iterables
            if not isinstance(sigma_range, (collections.Sequence, np.ndarray)):
                sigma_range = [sigma_range]
            if not isinstance(lam_range, (collections.Sequence, np.ndarray)):
                lam_range = [lam_range]

            # define some useful variables
            n_tr, d = train_data.size()
            n_te = test_data.size(0)
            n = min(n_tr, n_te)
            t = min(self.n_kernels, n_te)
            ones_t = torch.ones((t, 1), device=DEVICE)
            ones_n = torch.ones((n, 1), device=DEVICE)
            diag_n_idx = torch.cat((torch.range(0, n-1).view(1, -1).long(),
                                    torch.range(0, n-1).view(1, -1).long()))
            losses = np.zeros((len(sigma_range), len(lam_range)))

            # For each candidate of Gaussian kernel width...
            for sigma_idx, sigma in enumerate(sigma_range):

                # Apply the Gaussian kernel function to the elements of
                # D_tr and D_te.
                # Reuse variables train_data and test_data as we won't
                # need the originals again
                train_data = gaussian_kernel(D_tr, sigma)  # shape (t, n_tr)
                test_data = gaussian_kernel(D_te, sigma)   # shape (t, n_te)

                # Compute kernel arrays.
                # Shapes (t, t) and (t, 1)
                H, h = self.kernel_arrays(train_data, test_data)

                # For what follows train_data and test_data must have
                # the same shape, so choose n points randomly from each.
                train_data = train_data[:, np.random.choice(n_tr, n, replace=False)] # shape (t, n)
                test_data = test_data[:, np.random.choice(n_te, n, replace=False)] # shape (t, n)

                # for each candidate of regularisation parameter...
                for lam_idx, lam in enumerate(lam_range):

                    # compute the t x t matrix B
                    B = H + torch.eye(t, device=DEVICE) * (lam * (n_tr - 1)) / n_tr # shape (t, t)

                    # compute the t x n matrix B_0
                    B_inv = B.inverse() # shape (t, t)
                    B_inv_train_data = B_inv.mm(train_data) # shape (t, n)
                    diag_num = h.t().mm(B_inv_train_data).squeeze() # shape (n,)
                    diag_denom = (
                        n_tr * ones_n.t()
                        - ones_t.t().mm(train_data * B_inv_train_data)
                    ).squeeze() # shape (n,)
                    diag_sparse = torch.sparse.FloatTensor(
                        diag_n_idx,
                        (diag_num / diag_denom).cpu(),
                        torch.Size([n, n])).to(DEVICE) # sparse (n, n)
                    B_0 = (
                        B_inv.mm(h).mm(ones_n.t())
                        + (diag_sparse.t().mm(B_inv_train_data.t())).t()) # shape (t, n)

                    # compute B_1
                    diag_num = ones_t.t().mm(test_data * B_inv_train_data).squeeze() # shape (n,)
                    diag_sparse = torch.sparse.FloatTensor(
                        diag_n_idx,
                        (diag_num / diag_denom).cpu(),
                        torch.Size([n, n])).to(DEVICE) # sparse (n, n)
                    B_1 = (
                        B_inv.mm(test_data)
                        + (diag_sparse.t().mm(B_inv_train_data.t())).t()) # shape (t, n)

                    # compute B_2
                    B_2 = ((n_tr - 1) / (n_tr * (n_te - 1))) * (n_te * B_0 - B_1) # shape (t, n)
                    B_2 = torch.max(torch.zeros(1).to(DEVICE), B_2) # shape (t, n)

                    # compute leave-one-out CV loss
                    loss_1 = ((train_data * B_2).t().mm(ones_t).pow(2).sum() / (2 * n)).item()
                    loss_2 = (ones_t.t().mm(test_data * B_2).mm(ones_n) / n).item()
                    losses[sigma_idx, lam_idx] = loss_1 - loss_2
                    logging.info(
                        f'sigma = {sigma:0.5f}, '
                        f'lambda = {lam:0.5f}, '
                        f'loss = {losses[sigma_idx, lam_idx]:0.5f}')

            # get best hyperparameters
            sigma_idx, lam_idx = np.unravel_index(np.argmin(losses), losses.shape)
            sigma_hat, lam_hat = sigma_range[sigma_idx], lam_range[lam_idx]
            logging.info(
                f'\nbest loss = {losses[sigma_idx, lam_idx]:0.5f} for '
                f'sigma = {sigma_hat:0.5f} and '
                f'lambda = {lam_hat:0.5f}')

        return sigma_hat, lam_hat


    def kernel_arrays(self, train_data, test_data):
        r"""
        Computes kernel matrix H and vector h from algorithm.

        :math:`H[l, l']` is equal to the sum over i=1:n_tr of

        .. math::

            \sum_{1}^{n_{train}}
            \mathrm{exp}
            \left\{
                \frac{-(||x_i - c_l||^2 + -||x_i - c_l'||^2)}
                     {2 \sigma^2}
            \right\}

        where :math:`n_tr` is the number of samples from the training
        distribution, :math:`x_i` is the *i*-th sample from the training
        distribution and :math:`c_l` is the l-th kernel centre.

        :math:`h[l]` is equal to the sum over i=1:n_te of

        .. math::

            \mathrm{exp}
            \left\{
                \frac{-||x_i - c_l||^2}
                     {2 \sigma^2}
            \right\}

        where :math:`n_te` is the number of samples from the test
        distribution, :math:`x_i` is the *i*-th sample from the test
        distribution and :math:`c_l` is the :math:`l-th` kernel centre.

        Parameters
        ----------
        train_data : torch tensor
            ``train_data[l, i]`` is equal to the Gaussian kernel of the
            squared l2-norm of the difference between the *l*-th kernel
            centre and the *i*-th sample from the training distribution

            .. math::

                \mathrm{exp}
                \left\{
                    \frac{-||x_i - c_l||^2}
                         {2 \sigma^2}
                \right\}.

        test_data: torch tensor
            ``test_data[l, i]`` is equal to the Gaussian kernel of the
            squared l2-norm of the difference between the *l*-th kernel
            centre and the *i*-th sample from the test distribution

            .. math::

                \mathrm{exp}
                \left\{
                    \frac{-||x_i - c_l||^2}
                         {2 \sigma^2}
                \right\}.

        Returns
        -------
        H : torch tensor
            ``H[l, l']`` is equal to the sum over
            ``i=1:train_data.size(1)``
            of ``train_data[l, i] * train_data[l', i] / train_data.size(1)``

        h : torch tensor
            ``h[l]`` is equal to the sum over ``i=1:test_data.size(1)``
            of ``test_data[l, i] / test_data.size(1)``
        """
        # compute H
        n_tr = train_data.size(1)
        H = train_data.mm(train_data.t()) / n_tr # shape (t, t)

        # compute h
        n_te = test_data.size(1)
        h = test_data.sum(dim=1, keepdim=True) / n_te # shape (t, 1)

        return H, h
