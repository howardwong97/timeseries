from __future__ import annotations

import datetime as dt
import itertools
from functools import cached_property
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from timeseries.distribution import (
    Distribution,
    MultivariateDistribution,
)
from timeseries.linalg import outer_product2d
from timeseries.multivariate.base import (
    ConditionalCorrelationForecast,
    ConditionalCorrelationModel,
    LagLike,
)


class DCC(ConditionalCorrelationModel):
    """
    Dynamic Conditional Correlation (DCC) multivariate volatility model using univariate
    GARCH(p, o, q) conditional variance models
    """

    def __init__(
        self,
        y: Union[np.ndarray, pd.DataFrame],
        m: int = 1,
        l: int = 0,
        n: int = 1,
        distribution: Optional[MultivariateDistribution] = None,
        p: LagLike = 1,
        o: LagLike = 0,
        q: LagLike = 1,
        univariate_dists: Optional[Distribution, List[Distribution]] = None,
        constant: bool = False,
        lags: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        y : ndarray, DataFrame
            Dependent variables. Must have 2 or more columns of data
        m : int
            Order of symmetric innovations
        l : int
            Order of asymmetric innovations
        n : int
            Order of lagged correlation
        distribution : MultivariateDistribution, None
            The multivariate error distribution. Default is MultivariateNormal
        p : int, list (int)
            Integer(s) representing the order of symmetric innovations in the univariate
            volatility models. If a list of integers, it must have length equal to the
            number of columns of y
        o : int, list (int)
            Integer(s) representing the order of asymmetric innovations in the univariate
            volatility models. If a list of integers, it must have length equal to the
            number of columns of y
        q : int, list (int)
            Integer(s) representing the order of lagged conditional variance in the
            univariate volatility models. If a list of integers, it must have length equal
            to the number of columns of y
        univariate_dists : Distribution, list (Distribution), None
            Error distributions for the univariate GARCH models
        constant : bool
            Flag indicating whether to include a constant in the mean model specification
        lags : int, None
            Number of lags if estimating a vector auto-regression. Default is 0
        """
        super().__init__(y, distribution, p, o, q, univariate_dists, constant, lags)
        self.m: int = int(m)
        self.l: int = int(l)
        self.n: int = int(n)
        if m < 0 or l < 0 or n < 0:
            raise ValueError("Lag orders must be non-negative")
        if m == 0 and l == 0:
            raise ValueError("One of m or l must be strictly positive")

        self._name = "DCC" if self.l == 0 else "ADCC"

    def __str__(self) -> str:
        desc = self.name
        lags = [
            f"{k}: {v}" for k, v in (("m", self.m), ("l", self.l), ("n", self.n)) if v > 0
        ]
        desc += "(" + ", ".join(lags)
        desc += ", distribution: " + self.distribution.name + ")"

        return desc

    def __repr__(self) -> str:
        return self.__str__() + ", id: " + hex(id(self))

    @cached_property
    def num_intercept_params(self) -> int:
        return 0

    @cached_property
    def num_params(self) -> int:
        return self.m + self.l + self.n

    def parameter_names(self) -> List[str]:
        names = [f"alpha[{i + 1}]" for i in range(self.m)]
        names.extend([f"gamma[{i + 1}]" for i in range(self.l)])
        names.extend([f"beta[{i + 1}]" for i in range(self.n)])

        return names

    def bounds(self) -> List[Tuple[float, float]]:
        return [(0.0, 1.0)] * self.num_params

    def constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        a = np.zeros((self.num_params + 1, self.num_params))
        for i in range(self.num_params):
            a[i, i] = 1.0

        a[self.num_params, :] = -1.0
        a[self.num_params, self.m : self.m + self.l] = -self.scale
        b = np.zeros(self.num_params + 1)
        b[self.num_params] = -1.0

        return a, b

    def starting_values(self) -> np.ndarray:
        m, l, n = self.m, self.l, self.n
        alphas = [0.01, 0.03, 0.05, 0.1] if m > 0 else [0]
        gammas = [0.01, 0.03, 0.05] if l > 0 else [0]
        agbs = [0.99, 0.97, 0.95]
        params = list(itertools.product(*(alphas, gammas, agbs)))

        svs = []
        lls = np.zeros(len(params))
        for i, values in enumerate(params):
            alpha, gamma, agb = values
            beta = agb - alpha - self.scale * gamma
            sv = np.hstack(
                [alpha * np.ones(m) / m, gamma * np.ones(l) / l, beta * np.ones(n) / n]
            )
            svs.append(sv)
            lls[i] = self._gaussian_loglikelihood(sv)

        return svs[int(np.argmax(lls))]

    def compute_correlation(self, parameters: np.ndarray, qt: np.ndarray) -> np.ndarray:
        alpha, gamma, beta = self._parse_parameters(parameters)
        intercept = self.Qbar * (1 - np.sum(alpha) - np.sum(beta)) - self.Nbar * np.sum(
            gamma
        )
        self._dcc_recursion(alpha, gamma, beta, intercept, qt)

        qm12 = np.sqrt(np.diagonal(qt, axis1=1, axis2=2))
        corr = qt / outer_product2d(qm12)

        return corr

    def _one_step_forecast(self, parameters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = self.resids.shape[0]
        qt = np.zeros((t + 1, self.num_features, self.num_features))
        corr = self.compute_correlation(parameters, qt)

        return qt[-1, :, :], corr[-1, :, :]

    def forecast(
        self, parameters: np.ndarray, horizon: int
    ) -> ConditionalCorrelationForecast:
        q1, r1 = self._one_step_forecast(parameters)

        # Unconditional calculations
        qm12 = np.sqrt(np.diag(q1))
        er_1 = q1 / np.outer(qm12, qm12)
        r_bar = self.Rbar
        alpha, _, beta = self._parse_parameters(parameters)
        dcc_sum = np.sum(alpha) + np.sum(beta)

        # Initialize forecast matrices
        num_feat = self.num_features
        q_fore = np.zeros((horizon, num_feat, num_feat))
        corr_fore = np.zeros((horizon, num_feat, num_feat))
        q_fore[0, :, :] = q1
        corr_fore[0, :, :] = r1

        for i in range(1, horizon):
            corr_fore[i, :, :] = r_bar * (1 - dcc_sum**i) + er_1 * dcc_sum**i
            q_fore[i, :, :] = q_fore[0, :, :]

        # Forecast the variances using the GARCH models
        h_fore = np.zeros((horizon, num_feat))
        for i in range(self.num_features):
            model = self.garch_results[self._y_df.columns[i]]
            fcast = model.forecast(horizon=horizon)
            h_fore[:, i] = np.asarray(fcast)

        hh_fore = outer_product2d(np.sqrt(h_fore))
        cov_fore = corr_fore * hh_fore

        model_name = self.name
        if self._is_pandas and isinstance(
            self._y_df.index[-1], (dt.date, dt.datetime, pd.Timestamp)
        ):
            start = self._y_df.index[-1]
        else:
            start = None

        return ConditionalCorrelationForecast(
            h_fore, corr_fore, cov_fore, model_name, self._y_df.columns, start
        )

    def _dcc_recursion(
        self,
        alpha: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        intercept: np.ndarray,
        qt: np.ndarray,
    ) -> None:
        """
        Perform a recursion to compute the dynamics of the model

        Parameters
        ----------
        alpha : ndarray
            Symmetric innovation parameters
        gamma : ndarray
            Asymmetric innovation parameters
        beta : ndarray
            Lagged correlation parameters
        intercept : ndarray
            Model intercept
        qt : ndarray
            Array to store the dynamics (dimensions of the form T x K x K)

        Returns
        -------
        None
            qt is updated inplace.
        """
        for t in range(qt.shape[0]):
            qt[t, :, :] = intercept
            for i in range(len(alpha)):
                if t - i - 1 >= 0:
                    qt[t, :, :] += alpha[i] * self.std_data[t - i - 1, :, :]
                else:
                    qt[t, :, :] += alpha[i] * self.backcast
            for i in range(len(gamma)):
                if t - i - 1 >= 0:
                    qt[t, :, :] += gamma[i] * self.std_data_asym[t - i - 1, :, :]
                else:
                    qt[t, :, :] += gamma[i] * self.backcast_asym
            for i in range(len(beta)):
                if t - i - 1 >= 0:
                    qt[t, :, :] += beta[i] * qt[t - i - 1, :, :]
                else:
                    qt[t, :, :] += beta[i] * self.backcast

    def _parse_parameters(
        self, parameters: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the parameters of each component of the model as a tuple

        Parameters
        ----------
        parameters : ndarray
            Model parameters

        Returns
        -------
        alpha : ndarray
            Symmetric innovation parameters
        gamma : ndarray
            Asymmetric innovation parameters
        beta : ndarray
            Lagged correlation parameters
        """
        alpha = parameters[: self.m]
        gamma = parameters[self.m : self.m + self.l]
        beta = parameters[self.m + self.l : self.m + self.l + self.n]

        return alpha, gamma, beta

    def _gaussian_loglikelihood(self, parameters: np.ndarray) -> float:
        """
        Private implementation of a normal log likelihood used to estimate quantities
        that do not depend on the model distribution, such as starting values

        Parameters
        ----------
        parameters : ndarray
            Model parameters

        Returns
        -------
        llf : float
            Normal log likelihood
        """
        corr = self.compute_correlation(parameters, np.zeros_like(self.std_data))
        cov = corr * self._hh
        llf = self.distribution.loglikelihood(np.empty(0), self.resids, cov)

        return float(llf)

    def _transform_params_from_opt_to_standard(
        self, trans_parameters: np.ndarray
    ) -> np.ndarray:
        return trans_parameters
