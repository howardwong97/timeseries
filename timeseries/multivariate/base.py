from __future__ import annotations

import copy
import datetime as dt
import warnings
from abc import ABCMeta, abstractmethod
from functools import cached_property
from typing import Any, cast, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import sqrtm
from scipy.optimize import minimize, OptimizeResult
from statsmodels.base.model import ValueWarning
from statsmodels.iolib.summary import fmt_2cols, fmt_params, Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.tsa.vector_ar.var_model import VAR, VARResults

from timeseries.distribution import (
    Distribution,
    MultivariateDistribution,
    MultivariateNormal,
    Normal,
)
from timeseries.linalg import outer_product2d
from timeseries.univariate.garch import GARCH, GARCHResult
from timeseries.utils.array import ensure2d
from timeseries.utils.exceptions import (
    convergence_warning,
    CONVERGENCE_WARNING,
    ConvergenceWarning,
)
from timeseries.utils.formatter import format_float_fixed
from timeseries.utils.opt import combine_and_format_constraints

__all__ = [
    "ConditionalCorrelationModel",
    "ConditionalCorrelationForecast",
    "ConditionalCorrelationModelResult",
    "LagLike",
]

LagLike = Union[int, np.integer, List[Union[int, np.integer]], np.ndarray]

_callback_info = {"iter": 0, "llf": 0.0, "count": 0, "display": 1}


def _callback(*_: Any) -> None:
    """
    Callback for use in optimization

    Notes
    -----
    Uses global values to track iteration, iteration display frequency,
    log likelihood and function count
    """

    _callback_info["iter"] += 1
    disp = "Iteration: {0:>6},   Func. Count: {1:>6.3g},   Neg. LLF: {2}"
    if _callback_info["iter"] % _callback_info["display"] == 0:
        print(
            disp.format(
                _callback_info["iter"],
                _callback_info["count"],
                _callback_info["llf"],
            )
        )


def _check_garch_lag(lags: LagLike, size: int, name: str = "lag") -> List[int]:
    if isinstance(lags, (int, np.integer)):
        if lags < 0:
            raise ValueError(f"{name} must be non-negative")
        return [int(lags)] * size
    elif isinstance(lags, (list, np.ndarray)):
        if len(lags) != size or not all(int(i) == i and i >= 0 for i in lags):
            raise ValueError(f"{name} must contain {size} non-negative integers")
        return [int(i) for i in lags]
    else:
        raise TypeError(f"{name} must be an integer or a list/array of integers")


class ConditionalCorrelationModel(metaclass=ABCMeta):
    """Template for subclassing only"""

    def __init__(
        self,
        y: Union[np.ndarray, pd.DataFrame],
        distribution: Optional[MultivariateDistribution] = None,
        p: LagLike = 1,
        o: LagLike = 0,
        q: LagLike = 1,
        univariate_dists: Optional[Distribution, List[Distribution]] = None,
        constant: bool = False,
        lags: Optional[int] = None,
    ) -> None:
        self._name = ""
        self._y_original = y
        self._is_pandas = isinstance(y, (pd.Series, pd.DataFrame))
        self._y_df = pd.DataFrame(ensure2d(y))
        self._y = np.asarray(self._y_df, dtype=np.float64)
        self._num_features = self._y.shape[1]
        if not np.all(np.isfinite(self._y)):
            raise ValueError("NaN or inf values are present in y")
        if self._num_features < 2:
            raise ValueError("y must have 2 or more columns of data")

        self._fit_indices: List[int] = [0, int(self._y.shape[0])]

        self._normal = MultivariateNormal()
        if isinstance(distribution, MultivariateDistribution):
            self._distribution = distribution
        elif distribution is None:
            self._distribution = self._normal
        else:
            raise TypeError("distribution must inherit from MultivariateDistribution")

        self.p = _check_garch_lag(p, self._num_features, "p")
        self.o = _check_garch_lag(o, self._num_features, "o")
        self.q = _check_garch_lag(q, self._num_features, "q")

        if isinstance(univariate_dists, Distribution):
            self._univariate_dists = [univariate_dists] * self._num_features
        elif isinstance(univariate_dists, list):
            if len(univariate_dists) != self._num_features or not all(
                isinstance(d, Distribution) for d in univariate_dists
            ):
                raise ValueError(
                    f"univariate_dists must contain {self._num_features} Distribution "
                    f"instances"
                )
            self._univariate_dists: List[Distribution] = univariate_dists
        elif univariate_dists is None:
            self._univariate_dists = [Normal()] * self._num_features
        else:
            raise TypeError("univariate_dists must inherit from Distribution")

        self.constant: bool = constant
        self.lags: int = 0 if lags is None else int(lags)
        if self.lags < 0:
            raise ValueError("lags must be a non-negative integer, if not None")

        self._init_model()

    @property
    def name(self) -> str:
        """The name of the model"""
        return self._name

    @property
    def y(self) -> Union[np.ndarray, pd.DataFrame]:
        """The dependent variables"""
        return self._y_original

    @property
    def num_features(self) -> int:
        """Number of columns of data"""
        return self._num_features

    @property
    def distribution(self) -> MultivariateDistribution:
        """Set or get the multivariate error distribution"""
        return self._distribution

    @distribution.setter
    def distribution(self, value: MultivariateDistribution) -> None:
        if not isinstance(value, MultivariateDistribution):
            raise ValueError("Must subclass MultivariateDistribution")
        self._distribution = value

    @property
    def var_results(self) -> VARResults:
        """The results object of the vector auto-regression"""
        return self._var_results

    @property
    def garch_results(self) -> Dict[Any, GARCHResult]:
        """The result objects of the univariate GARCH models"""
        return self._garch_results

    @cached_property
    @abstractmethod
    def num_intercept_params(self) -> int:
        """
        Number of elements in the intercept to be estimated. Returns 0 if the intercept
        is not jointly estimated with the model dynamics parameters
        """

    @cached_property
    @abstractmethod
    def num_params(self) -> int:
        """
        Number of parameters in the model
        """

    @abstractmethod
    def parameter_names(self) -> List[str]:
        """
        Names of the model parameters

        Returns
        -------
        names : list (str)
            Parameter names
        """

    def _all_parameter_names(self) -> List[str]:
        """
        Names of model parameters and the distribution shape parameters

        Returns
        -------
        names : list (str)
            Parameter names
        """
        names = self.parameter_names()
        names.extend(self.distribution.parameter_names())

        return names

    @abstractmethod
    def bounds(self) -> List[Tuple[float, float]]:
        """
        Parameter bounds for use in optimization

        Returns
        -------
        bounds : list (tuple)
            List of parameter bounds where each element has the form (lower, upper)
        """

    @abstractmethod
    def constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct linear parameter constraints arrays

        Returns
        -------
        a : ndarray
            Constraint loadings
        b : ndarray
            Constraint values

        Notes
        -----
        Parameters satisfy the constraints a.dot(parameters) - b >= 0
        """

    @abstractmethod
    def starting_values(self) -> np.ndarray:
        """
        Compute starting values for the model parameters

        Returns
        -------
        sv : ndarray
            Starting values
        """

    @abstractmethod
    def compute_correlation(self, parameters: np.ndarray, qt: np.ndarray) -> np.ndarray:
        """
        Compute the conditional correlations using the model

        Parameters
        ----------
        parameters : ndarray
            Model parameters
        qt : ndarray
            Array to store the model dynamics

        Returns
        -------
        corr : ndarray
            Conditional correlations
        """

    def _compute_correlation_for_opt(
        self, parameters: np.ndarray, qt: np.ndarray
    ) -> np.ndarray:
        """Wrapper for compute_correlation. Optional to override"""
        return self.compute_correlation(parameters, qt)

    def _loglikelihood(
        self, parameters: np.ndarray, qt: np.ndarray, individual: bool = False
    ) -> Union[float, np.ndarray]:
        """
        Compute the (negative of the) log likelihood

        Parameters
        ----------
        parameters : ndarray
            Model parameters and distribution shape parameters
        qt : ndarray
            Array to store the model dynamics
        individual : bool
            Flag indicating whether to return the vector of individual log likelihoods
            (True) or the sum (False)

        Returns
        -------
        neg_llf : float, ndarray
            The log likelihood(s) times -1.0
        """

        cp, dp = parameters[: self.num_params], parameters[self.num_params :]
        corr = self.compute_correlation(cp, qt)
        cov = corr * self._hh
        llf = self.distribution.loglikelihood(dp, self.resids, cov, individual)

        return -1.0 * llf

    def _loglikelihood_for_opt(
        self, parameters: np.ndarray, qt: np.ndarray, individual: bool = False
    ) -> Union[float, np.ndarray]:
        """
        Compute the (negative of the) log likelihood. As opposed to `_loglikelihood`, this
        updates the callback info and uses `_compute_correlation_for_opt` instead of
        `compute_correlation`

        Parameters
        ----------
        parameters : ndarray
            Model parameters and distribution shape parameters
        qt : ndarray
            Array to store the model dynamics
        individual : bool
            Flag indicating whether to return the vector of individual log likelihoods
            (True) or the sum (False)

        Returns
        -------
        neg_llf : float, ndarray
            The log likelihood(s) times -1.0
        """
        _callback_info["count"] += 1

        cp, dp = parameters[: self.num_params], parameters[self.num_params :]
        corr = self._compute_correlation_for_opt(cp, qt)
        cov = corr * self._hh
        llf = self.distribution.loglikelihood(dp, self.resids, cov, individual)

        if not individual:
            _callback_info["llf"] = neg_llf = -float(llf)
            return neg_llf

        return cast(np.ndarray, -llf)

    @abstractmethod
    def _transform_params_from_opt_to_standard(
        self, trans_parameters: np.ndarray
    ) -> np.ndarray:
        """
        Transform the model parameters from the unrestricted optimization space to the
        model restricted parameter space. If not applicable, simply returns the
        parameters unchanged.

        Parameters
        ----------
        trans_parameters : ndarray
            Transformed model parameters

        Returns
        -------
        parameters : ndarray
            Model parameters
        """

    def fit(
        self,
        update_freq: int = 1,
        disp: bool = True,
        cov_type: Literal["robust", "classic"] = "robust",
        show_warning: bool = True,
        tol: Optional[float] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ConditionalCorrelationModelResult:
        starting_values = self.starting_values()
        starting_values = np.hstack(
            [starting_values, self.distribution.starting_values(self.std_resids)]
        )

        bounds = list(self.bounds())
        bounds.extend(self.distribution.bounds())

        constraints = (self.constraints(), self.distribution.constraints())
        offsets = np.array((self.num_params, self.distribution.num_params), dtype=int)
        ineq_constraints = combine_and_format_constraints(constraints, offsets)

        if update_freq <= 0 or not disp:
            _callback_info["display"] = 2**31
        else:
            _callback_info["display"] = update_freq

        options = {} if options is None else options
        options.setdefault("disp", disp)

        func = self._loglikelihood_for_opt
        qt = np.zeros_like(self.std_data)
        args = (qt, False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # noinspection PyTypeChecker
            opt = minimize(
                func,
                x0=starting_values,
                args=args,
                method="SLSQP",
                bounds=bounds,
                constraints=ineq_constraints,
                tol=tol,
                callback=_callback,
                options=options,
            )

        if show_warning:
            warnings.filterwarnings("always", "", ConvergenceWarning)
        else:
            warnings.filterwarnings("ignore", "", ConvergenceWarning)

        if opt.status != 0 and show_warning:
            warnings.warn(
                convergence_warning.format(code=opt.status, string_message=opt.message),
                ConvergenceWarning,
            )

        params = opt.x
        loglikelihood = -1.0 * opt.fun

        cp, dp = params[: self.num_params], params[self.num_params :]
        cp = self._transform_params_from_opt_to_standard(cp)
        params = np.hstack([cp, dp])

        corr = self.compute_correlation(cp, qt)
        cov = corr * self._hh

        nobs_orig = len(self._y)
        first_obs, last_obs = self._fit_indices
        resids_final = np.full_like(self._y, np.nan)
        resids_final[first_obs:last_obs, :] = self.resids
        var_final = np.full_like(self._y, np.nan)
        var_final[first_obs:last_obs, :] = self.H
        corr_final = np.full((nobs_orig, self.num_features, self.num_features), np.nan)
        corr_final[first_obs:last_obs, :, :] = corr
        cov_final = np.full((nobs_orig, self.num_features, self.num_features), np.nan)
        cov_final[first_obs:last_obs, :, :] = cov

        names = self._all_parameter_names()
        model_copy = copy.deepcopy(self)

        return ConditionalCorrelationModelResult(
            params,
            None,
            cov_type,
            resids_final,
            var_final,
            corr_final,
            cov_final,
            self._y_df,
            names,
            loglikelihood,
            self._is_pandas,
            first_obs,
            last_obs,
            opt,
            model_copy,
        )

    def compute_param_cov(self, params: np.ndarray, robust: bool = True) -> np.ndarray:
        """
        Computes parameter covariances using numerical derivatives.

        Parameters
        ----------
        params : ndarray
            Model parameters
        robust : bool, optional
            Flag indicating whether to use robust standard errors (True) or
            classic MLE (False)

        Returns
        -------
        vcv : ndarray
            Parameter variance-covariance matrix
        """
        nobs = self.resids.shape[0]
        kwargs = {"qt": np.zeros_like(self.std_data), "individual": False}
        hess = approx_hess(params, self._loglikelihood, kwargs=kwargs)
        hess /= nobs
        inv_hess = np.linalg.inv(hess)

        if robust:
            kwargs["individual"] = True
            scores = approx_fprime(params, self._loglikelihood, kwargs=kwargs)
            score_cov = np.cov(scores, rowvar=False)
            return inv_hess @ score_cov @ inv_hess / nobs

        return inv_hess / nobs

    @abstractmethod
    def forecast(
        self, parameters: np.ndarray, horizon: int
    ) -> ConditionalCorrelationForecast:
        """
        Compute multistep analytic forecasts of the variance, correlation and covariance

        Parameters
        ----------
        parameters : ndarray
            Model parameters
        horizon : int
            Number of time steps to forecast

        Returns
        -------
        forecast : ConditionalCorrelationForecast
            Object containing the forecasts
        """

    def _init_model(self) -> None:
        warnings.simplefilter("ignore", ValueWarning)
        var_model = VAR(self._y_original)
        trend = "c" if self.constant else "n"
        self._var_results: VARResults = var_model.fit(self.lags, trend=trend)

        self.resids = np.asarray(self._var_results.resid)
        self._fit_indices[0] = len(self._y_original) - len(self.resids)

        self._garch_results: Dict[Any, GARCHResult] = {}
        self.H = np.zeros_like(self.resids)
        names = self._y_df.columns
        for i in range(self.num_features):
            model = GARCH(
                self.resids[:, i],
                self.p[i],
                self.o[i],
                self.q[i],
                2.0,
                self._univariate_dists[i],
                True,
            )
            garch_result = model.fit(disp=False)
            self._garch_results[names[i]] = garch_result
            self.H[:, i] = np.asarray(garch_result.conditional_variance)

        self.std_resids = self.resids / np.sqrt(self.H)
        self.resids_asym = self.resids * (self.resids < 0)
        self.std_resids_asym = self.resids_asym / np.sqrt(self.H)

        self._hh = outer_product2d(np.sqrt(self.H))
        self.std_data = outer_product2d(self.std_resids)
        self.std_data_asym = outer_product2d(self.std_resids_asym)

        tau = min(75, self.resids.shape[0])
        w = 0.94 ** np.arange(tau)
        w /= w.sum()
        w = w[:, None, None]
        self.backcast = np.sum(w * self.std_data[:tau, :, :], axis=0)
        self.backcast_asym = np.sum(w * self.std_data_asym[:tau, :, :], axis=0)

        self.Qbar = self.std_data.mean(axis=0)
        self.Nbar = self.std_data_asym.mean(axis=0)
        q = np.sqrt(np.diag(self.Qbar))
        self.Rbar = self.Qbar / np.outer(q, q)
        self.Rbar = (self.Rbar + self.Rbar.T) / 2
        np.fill_diagonal(self.Rbar, 1)

        qm12 = np.asarray(sqrtm(np.linalg.inv(self.Qbar)), dtype=np.float64)
        self.scale = np.max(np.linalg.eigvals(qm12 @ self.Nbar @ qm12))
        self.g_scale = np.diag(self.Nbar)


class ConditionalCorrelationModelResult:
    """Results from the estimation of a ConditionalCorrelationModel"""

    def __init__(
        self,
        params: np.ndarray,
        param_cov: Optional[np.ndarray],
        cov_type: str,
        resids: np.ndarray,
        variance: np.ndarray,
        correlation: np.ndarray,
        covariance: np.ndarray,
        dep_var: pd.DataFrame,
        names: List[str],
        loglikelihood: float,
        is_pandas: bool,
        fit_start: int,
        fit_stop: int,
        optim_output: OptimizeResult,
        model: ConditionalCorrelationModel,
    ) -> None:
        """
        Parameters
        ----------
        params : ndarray
            Estimated parameters
        param_cov : ndarray, None
            Estimated variance-covariance matrix of params. If `None`, calls method to
            compute variance from the model when param_cov is first accessed
        cov_type : str
            The name of the covariance estimator used
        resids : ndarray
            Residuals from the model
        variance : ndarray
            Conditional variance from the model
        correlation : ndarray
            Conditional correlation from the model
        covariance : ndarray
            Conditional covariance from the model
        dep_var : pd.Series
            Dependent variable
        names : list (str)
            Model parameter names
        loglikelihood : float
            Model log likelihood
        is_pandas : bool
            Flag indicating whether the original input data is Series or DataFrame
        fit_start : int
            Index of the first observation of the sample used to estimate the model
        fit_stop : int
            Index (+1) of the last observation of the sample used to estimate the model
        optim_output : OptimizeResult
            Result of the log likelihood estimation
        model : ConditionalCorrelationModel
            THe model object used to estimate the parameters
        """
        self._params = params
        self._param_cov = param_cov
        self.cov_type = cov_type
        self._resids = resids
        self._variance = variance
        self._correlation = correlation
        self._covariance = covariance
        self._dep_var = dep_var
        self._index = dep_var.index
        self._cols = dep_var.columns
        self._multi_index = pd.MultiIndex.from_product((self._index, self._cols))
        self._names = list(names)
        self._loglikelihood = loglikelihood
        self._is_pandas = is_pandas
        self._fit_indices = (fit_start, fit_stop)
        self._optim_output = optim_output
        self._model = model
        self._datetime = dt.datetime.now()

    def __repr__(self) -> str:
        out = self.__str__() + "\n"
        out += self.__class__.__name__
        out += f", id: {hex(id(self))}"
        return out

    def __str__(self) -> str:
        return self.summary().as_text()

    def summary(self) -> Summary:
        """
        Constructs a summary of the results from a fit model

        Returns
        -------
        summary : Summary
            Object that contains tables and facilitates export to text, html or LaTeX
        """
        model = self.model
        model_name = model.name

        if model.lags > 0:
            mean_model_name = "VAR"
        elif model.constant:
            mean_model_name = "Constant"
        else:
            mean_model_name = "Zero"

        # Summary header
        top_left = [
            ("Mean Model:", mean_model_name),
            ("Volatility Models:", "GARCH"),
            ("Distribution:", model.distribution.name),
            ("Method:", "Maximum Likelihood"),
            ("Date:", self._datetime.strftime("%a, %b %d %Y")),
            ("Time:", self._datetime.strftime("%H:%M:%S")),
        ]

        top_right = [
            ("Log-Likelihood:", f"{self.loglikelihood:#10.6g}"),
            ("AIC:", f"{self.aic:#10.6g}"),
            ("BIC:", f"{self.bic:#10.6g}"),
            ("No. Observations:", f"{self.nobs}"),
            ("Df. Residuals:", f"{self.df_resids}"),
            ("Df. Model:", f"{self.df_model}"),
        ]

        title = model_name + " Model Results"
        stubs, vals = [], []
        for stub, val in top_left:
            stubs.append(stub)
            vals.append([val])

        table = SimpleTable(vals, txt_fmt=fmt_2cols, title=title, stubs=stubs)

        # Create Summary instance
        summary = Summary()
        fmt = fmt_2cols
        fmt["data_fmts"][1] = "%18s"

        top_right = [("%-21s" % ("  " + k), v) for k, v in top_right]
        stubs, vals = [], []
        for stub, val in top_right:
            stubs.append(stub)
            vals.append([val])

        table.extend_right(SimpleTable(vals, stubs=stubs))
        summary.tables.append(table)

        conf_int = np.asarray(self.conf_int())
        conf_int_str = []
        for c in conf_int:
            c0 = format_float_fixed(c[0], 7, 3)
            c1 = format_float_fixed(c[1], 7, 3)
            conf_int_str.append(f"[{c0},{c1}]")

        stubs = list(self._names)
        header = ["coef", "std err", "t", "P>|t|", "95.0% Conf. Int."]
        table_vals = (
            np.asarray(self.params),
            np.asarray(self.std_err),
            np.asarray(self.tvalues),
            np.asarray(self.pvalues),
            pd.Series(conf_int_str),
        )
        # (0,0) is a dummy format
        formats = [(10, 4), (9, 3), (9, 3), (9, 3), (0, 0)]
        param_table_data = []
        for pos in range(len(table_vals[0])):
            row = []
            for i, table_val in enumerate(table_vals):
                val = table_val[pos]
                if isinstance(val, (np.float64, float)):
                    converted = format_float_fixed(val, *formats[i])
                else:
                    converted = val
                row.append(converted)
            param_table_data.append(row)

        ic = model.num_intercept_params
        mc = model.num_params - ic
        dc = model.distribution.num_params
        counts = (ic, mc, dc)
        titles = ("Intercept", "Correlation Model", "Distribution")
        total = 0

        for title, count in zip(titles, counts):
            if count == 0:
                continue
            table_data = param_table_data[total : total + count]
            table_stubs = stubs[total : total + count]
            total += count
            table = SimpleTable(
                table_data,
                stubs=table_stubs,
                txt_fmt=fmt_params,
                headers=header,
                title=title,
            )
            summary.tables.append(table)

        extra_text = ["Covariance estimator: " + self.cov_type]

        if self.convergence_flag:
            string_message = self._optim_output.message
            extra_text.append(CONVERGENCE_WARNING.format(msg=string_message))

        summary.add_extra_txt(extra_text)

        return summary

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Parameter confidence intervals

        Parameters
        ----------
        alpha : float, optional
            Size (prob.) to use when constructing the confidence interval.

        Returns
        -------
        ci : DataFrame
            Array where the ith row contains the confidence interval  for the
            ith parameter
        """
        cv = stats.norm.ppf(1.0 - alpha / 2.0)
        se = self.std_err
        params = self.params

        return pd.DataFrame(
            np.vstack((params - cv * se, params + cv * se)).T,
            columns=["lower", "upper"],
            index=self._names,
        )

    @cached_property
    def model(self) -> ConditionalCorrelationModel:
        """Model instance used to produce the fit"""
        return self._model

    @property
    def var_results(self) -> VARResults:
        """Results object for the vector auto-regression"""
        return self.model.var_results

    @property
    def garch_results(self) -> Dict[Any, GARCHResult]:
        """The results of the univariate GARCH fits"""
        return self.model.garch_results

    @property
    def df_resids(self) -> int:
        """Degrees of freedom of the residuals"""
        return self.var_results.df_resid

    @property
    def df_model(self) -> int:
        """Degrees of freedom of the model"""
        return self.var_results.df_model

    @cached_property
    def fit_start(self) -> int:
        """Index of the first observation used to estimate the model"""
        return self._fit_indices[0]

    @cached_property
    def fit_stop(self) -> int:
        """The last observation used to estimate the model. This is the last index + 1"""
        return self._fit_indices[1]

    @cached_property
    def nobs(self) -> int:
        """Number of observations used to estimate the model"""
        return self.fit_stop - self.fit_start

    @cached_property
    def params(self) -> pd.Series:
        """Estimated model parameters"""
        return pd.Series(self._params, index=self._names, name="params")

    @cached_property
    def num_params(self) -> int:
        """Number of parameters in the model"""
        return len(self.params)

    @cached_property
    def resids(self) -> Union[np.ndarray, pd.DataFrame]:
        """Model residuals"""
        if self._is_pandas:
            return pd.DataFrame(self._resids, index=self._index, columns=self._cols)
        return self._resids

    @cached_property
    def conditional_variance(self) -> Union[np.ndarray, pd.DataFrame]:
        """Estimated conditional variances"""
        if self._is_pandas:
            return pd.DataFrame(self._variance, index=self._index, columns=self._cols)
        return self._variance

    @cached_property
    def conditional_correlation(self) -> Union[np.ndarray, pd.DataFrame]:
        """Estimated conditional correlations"""
        if self._is_pandas:
            corr = self._correlation.reshape((-1, len(self._cols)))
            return pd.DataFrame(corr, index=self._multi_index, columns=self._cols)
        return self._correlation

    @cached_property
    def conditional_covariance(self) -> Union[np.ndarray, pd.DataFrame]:
        """Estimated conditional covariances"""
        if self._is_pandas:
            cov = self._covariance.reshape((-1, len(self._cols)))
            return pd.DataFrame(cov, index=self._multi_index, columns=self._cols)
        return self._covariance

    @cached_property
    def loglikelihood(self) -> float:
        """Model log likelihood"""
        return self._loglikelihood

    @cached_property
    def aic(self) -> float:
        """Akaike Information Criteria"""
        return -2 * self.loglikelihood + 2 * self.num_params

    @cached_property
    def bic(self) -> float:
        """Schwarz/Bayesian Information Criteria"""
        return -2 * self.loglikelihood + np.log(self.nobs) * self.num_params

    @cached_property
    def param_cov(self) -> pd.DataFrame:
        """Parameter covariance"""
        if self._param_cov is not None:
            param_cov = self._param_cov
        else:
            params = np.asarray(self.params)
            if self.cov_type == "robust":
                param_cov = self.model.compute_param_cov(params)
            else:
                param_cov = self.model.compute_param_cov(params, robust=False)

        return pd.DataFrame(param_cov, columns=self._names, index=self._names)

    @cached_property
    def pvalues(self) -> pd.Series:
        """
        Array of p-values for the t-statistics
        """
        pvals = np.asarray(stats.norm.sf(np.abs(self.tvalues)) * 2, dtype=float)
        return pd.Series(pvals, index=self._names, name="pvalues")

    @cached_property
    def std_err(self) -> pd.Series:
        """
        Array of parameter standard errors
        """
        se = np.asarray(np.sqrt(np.diag(self.param_cov)), dtype=float)
        return pd.Series(se, index=self._names, name="std_err")

    @cached_property
    def tvalues(self) -> pd.Series:
        """
        Array of t-statistics testing the null that the coefficient are 0
        """
        tvalues = self.params / self.std_err
        tvalues.name = "tvalues"
        return tvalues

    @cached_property
    def convergence_flag(self) -> int:
        """
        scipy.optimize.minimize result flag
        """
        return self._optim_output.status

    @property
    def optimization_result(self) -> OptimizeResult:
        """
        Information about the convergence of the loglikelihood optimization

        Returns
        -------
        optim_result : OptimizeResult
            Result from numerical optimization of the log-likelihood.
        """
        return self._optim_output

    def forecast(
        self, parameters: Optional[np.ndarray, pd.Series] = None, horizon: int = 1
    ) -> ConditionalCorrelationForecast:
        """
        Compute forecasts of the variance, correlation and covariance using the model

        Parameters
        ----------
        parameters : ndarray, pd.Series, None
            Alternative model parameters to use. If not provided, the parameters estimated
            when fitting the model are used
        horizon : int
            Number of steps to forecast

        Returns
        -------
        forecasts : ConditionalCorrelationForecast
            Object containing the forecasts
        """
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be an integer >= 1")
        if parameters is None:
            parameters = self._params
        else:
            if (
                parameters.size != np.array(self._params).size
                or parameters.ndim != self._params.ndim
            ):
                raise ValueError("parameters have incorrect dimensions")

        parameters = np.asarray(parameters)

        return self.model.forecast(parameters, horizon)


class ConditionalCorrelationForecast:
    """
    Container for forecasts of the variance, correlation and covariance from an estimated
    ConditionalCorrelationModel
    """

    def __init__(
        self,
        variance: np.ndarray,
        correlation: np.ndarray,
        covariance: np.ndarray,
        model_name: str,
        column_names: Union[pd.Index, List[str]],
        start: Optional[Union[dt.date, dt.datetime, pd.Timestamp]] = None,
    ) -> None:
        """
        Parameters
        ----------
        variance : ndarray
            Forecasted variances
        correlation : ndarray
            Forecasted correlations
        covariance : ndarray
            Forecasted covariances
        model_name : str
            Name of the model
        column_names : pd.Index, list (str)
            Names of the features
        start : date, datetime, Timestamp, None
            Start date from which the forecast is produced
        """
        self._variance = variance
        self._correlation = correlation
        self._covariance = covariance
        self._model_name = model_name
        self._column_names = pd.Index(column_names, name=None)
        self._horizon = self._variance.shape[0]
        self._index = pd.Index(
            [f"h[{i + 1}]" for i in range(self._horizon)], name="horizon"
        )
        self._start = start
        self._multi_index = pd.MultiIndex.from_product((self._index, self._column_names))

    def __str__(self) -> str:
        desc = "ConditionalCorrelationForecast("
        desc += f"model: {self._model_name}, horizon: {self._horizon}"
        if self._start is not None:
            desc += f", {self._start.strftime('%Y-%m-%d')}"

        return desc + ")"

    def __repr__(self) -> str:
        return self.__str__() + ", id: " + hex(id(self))

    @cached_property
    def variance(self) -> pd.DataFrame:
        """Variance forecasts"""
        return pd.DataFrame(self._variance, index=self._index, columns=self._column_names)

    @cached_property
    def correlation(self) -> pd.DataFrame:
        """Correlation forecasts"""
        corr = self._correlation.reshape((-1, len(self._column_names)))
        return pd.DataFrame(corr, index=self._multi_index, columns=self._column_names)

    @cached_property
    def covariance(self) -> pd.DataFrame:
        """Covariance forecasts"""
        cov = self._covariance.reshape((-1, len(self._column_names)))
        return pd.DataFrame(cov, index=self._multi_index, columns=self._column_names)
