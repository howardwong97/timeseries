from __future__ import annotations

import copy
import datetime as dt
import itertools
import warnings
from functools import cached_property
from typing import Any, cast, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, OptimizeResult
from statsmodels.iolib.summary import fmt_2cols, fmt_params, Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.numdiff import approx_fprime, approx_hess

from timeseries.distribution import (
    Distribution,
    GeneralizedError,
    Normal,
    SkewStudent,
    StudentsT,
    SUPPORTED_UNIVARIATE_DISTRIBUTIONS,
)
from timeseries.utils.array import ensure1d
from timeseries.utils.exceptions import (
    convergence_warning,
    CONVERGENCE_WARNING,
    ConvergenceWarning,
    data_scale_warning,
    DataScaleWarning,
)
from timeseries.utils.formatter import format_float_fixed
from timeseries.utils.opt import combine_and_format_constraints

__all__ = ["GARCH", "GARCHResult", "garch_model"]

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


class GARCH:
    """
    GARCH and related model estimation

    The following models can be specified using GARCH:
        * ARCH(p)
        * GARCH(p,q)
        * GJR-GARCH(p,o,q)
        * AVARCH(p)
        * AVGARCH(p,q)
        * TARCH(p,o,q)
        * Models with arbitrary, pre-specified powers
    """

    def __init__(
        self,
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        p: int = 1,
        o: int = 0,
        q: int = 1,
        power: float = 2.0,
        distribution: Optional[Distribution] = None,
        rescale: Optional[bool] = None,
    ) -> None:
        """
        Parameters
        ----------
        y : ndarray, pd.Series, pd.DataFrame
            Dependent variable
        p : int
            Order of symmetric innovations
        o : int
            Order of asymmetric innovations
        q : int
            Order of lagged conditional variance
        power : float
            Power to use with the innovations, abs(resids) ** power
        distribution : Distribution, None
            The error distribution. Default is Normal
        rescale : bool, None
            Flag indicating whether to automatically rescale data if the scale of the
            data is likely to produce convergence issues when estimating model parameters.
            If False, the model is estimated on the data without transformation.  If True,
            then y is rescaled and the new scale is reported in the estimation results.
        """
        self._y_original = y
        self._is_pandas = isinstance(y, (pd.Series, pd.DataFrame))
        self._y_series = cast(pd.Series, ensure1d(y, "y", True))
        self._y = np.asarray(self._y_series, dtype=np.float64)
        if not np.all(np.isfinite(self._y)):
            raise ValueError("NaN or inf values are present in y")

        self.p: int = int(p)
        self.o: int = int(o)
        self.q: int = int(q)
        self.power: float = power
        self._num_params = 1 + self.p + self.o + self.q
        if p < 0 or o < 0 or q < 0:
            raise ValueError("Lag lengths must be non-negative")
        if p == 0 and o == 0:
            raise ValueError("One of p or o must be strictly positive")
        if power <= 0.0:
            raise ValueError("power must be strictly positive")

        self._normal = Normal()
        if isinstance(distribution, Distribution):
            self._distribution = distribution
        elif distribution is None:
            self._distribution = self._normal
        else:
            raise TypeError("distribution must inherit from Distribution")

        self.rescale: Union[bool, None] = rescale
        self.scale: float = 1.0
        self._scale_changed: bool = False

    def __str__(self) -> str:
        descr = self.name

        if self.power != 1.0 and self.power != 2.0:
            descr = descr[:-1] + ", "
        else:
            descr += "("

        for k, v in (("p", self.p), ("o", self.o), ("q", self.q)):
            if v > 0:
                descr += k + ": " + str(v) + ", "

        descr = descr[:-2] + ")"

        return descr

    def __repr__(self) -> str:
        return self.__str__() + ", id: " + hex(id(self))

    @property
    def name(self) -> str:
        """The name of the model"""
        p, o, q, power = self.p, self.o, self.q, self.power
        if power == 2.0:
            if o == 0 and q == 0:
                return "ARCH"
            elif o == 0:
                return "GARCH"
            else:
                return "GJR-GARCH"
        elif power == 1.0:
            if o == 0 and q == 0:
                return "AVARCH"
            elif o == 0:
                return "AVGARCH"
            else:
                return "TARCH/ZARCH"
        else:
            if o == 0 and q == 0:
                return f"Power ARCH (power: {self.power:0.1f})"
            elif o == 0:
                return f"Power GARCH (power: {self.power:0.1f})"
            else:
                return f"Asym. Power GARCH (power: {self.power:0.1f})"

    @property
    def y(self) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """The dependent variable"""
        return self._y_original

    @property
    def num_params(self) -> int:
        """The number of parameters in the model"""
        return self._num_params

    @property
    def distribution(self) -> Distribution:
        """Set or get the error distribution"""
        return self._distribution

    @distribution.setter
    def distribution(self, value: Distribution) -> None:
        if not isinstance(value, Distribution):
            raise ValueError("Must subclass Distribution")
        self._distribution = value

    @property
    def scale_changed(self) -> bool:
        """Flag indicating whether to data has been adjusted/rescaled"""
        return self._scale_changed

    @property
    def resids(self) -> np.ndarray:
        """The residuals used to estimate the model. This is just the input
        data as the mean is assumed to be zero"""
        return self._y

    def parameter_names(self) -> List[str]:
        """
        Names of the model parameters

        Returns
        -------
        names : list (str)
            Parameter names
        """
        names = ["omega"]
        names.extend([f"alpha[{i + 1}]" for i in range(self.p)])
        names.extend([f"gamma[{i + 1}]" for i in range(self.o)])
        names.extend([f"beta[{i + 1}]" for i in range(self.q)])

        return names

    def _all_parameter_names(self) -> List[str]:
        """
        Names of the model parameters and the distribution shape parameters

        Returns
        -------
        names : list (str)
            Model and distribution parameters names
        """
        names = self.parameter_names()
        names.extend(self.distribution.parameter_names())

        return names

    def _check_scale(self) -> None:
        check = self.rescale in (None, True)
        if not check:
            return

        orig_scale = scale = self.resids.var()
        rescale = 1.0
        while not 0.1 <= scale < 10000.0 and scale > 0:
            if scale < 1.0:
                rescale *= 10
            else:
                rescale /= 10
            scale = orig_scale * rescale**2

        if rescale == 1.0:
            return
        if self.rescale is None:
            warnings.warn(
                data_scale_warning.format(orig_scale, rescale), DataScaleWarning
            )
            return

        self.scale = rescale

    def starting_values(self) -> np.ndarray:
        """
        Compute starting values for the model parameters. Performs a grid search of
        possible starting values and selects the set of parameters that produces the
        highest Gaussian log likelihood

        Returns
        -------
        sv : ndarray
            Starting values
        """
        p, o, q, power = self.p, self.o, self.q, self.power
        alphas = [0.01, 0.05, 0.1, 0.2]
        gammas = alphas
        abg = [0.5, 0.7, 0.9, 0.98]
        abgs = list(itertools.product(*[alphas, gammas, abg]))

        target = np.mean(abs(self.resids) ** power)
        scale = np.mean(self.resids**2) / (target ** (2.0 / power))
        target *= scale ** (power / 2)

        svs = []
        lls = np.zeros(len(abgs))
        for i, values in enumerate(abgs):
            alpha, gamma, agb = values
            sv = (1.0 - agb) * target * np.ones(p + o + q + 1)
            if p > 0:
                sv[1 : 1 + p] = alpha / p
                agb -= alpha
            if o > 0:
                sv[1 + p : 1 + p + o] = gamma / o
                agb -= gamma / 2.0
            if q > 0:
                sv[1 + p + o : 1 + p + o + q] = agb / q
            svs.append(sv)
            lls[i] = self._gaussian_loglikelihood(sv)

        return svs[int(np.argmax(lls))]

    def bounds(self) -> List[Tuple[float, float]]:
        """
        Parameter bounds for use in constrained optimization

        Returns
        -------
        bounds : list (tuple)
            List of parameter bounds where each element has form (lower, upper)
        """
        v = float(np.mean(np.abs(self.resids) ** self.power))

        bounds = [(1e-8 * v, 10.0 * float(v))]
        bounds.extend([(0.0, 1.0)] * self.p)
        for i in range(self.o):
            if i < self.p:
                bounds.append((-1.0, 2.0))
            else:
                bounds.append((0.0, 2.0))

        bounds.extend([(0.0, 1.0)] * self.q)

        return bounds

    def constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct constraints arrays for use in non-linear optimization

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
        p, o, q = self.p, self.o, self.q
        k_arch = p + o + q

        a = np.zeros((k_arch + 2, k_arch + 1))
        for i in range(k_arch + 1):
            a[i, i] = 1.0

        for i in range(o):
            if i < p:
                a[i + p + 1, i + 1] = 1.0

        a[k_arch + 1, 1:] = -1.0
        a[k_arch + 1, p + 1 : p + o + 1] = -0.5
        b = np.zeros(k_arch + 2)
        b[k_arch + 1] = -1.0

        return a, b

    def backcast(self) -> float:
        """
        Returns the value to use when initializing the recursion

        Returns
        -------
        backcast : float
        """
        tau = min(75, self.resids.shape[0])
        w = 0.94 ** np.arange(tau)
        w /= w.sum()
        backcast = np.sum(w * np.abs(self.resids[:tau]) ** self.power)

        return float(backcast)

    def compute_variance(self, parameters: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
        """
        Computes the conditional variance using the model

        Parameters
        ----------
        parameters : ndarray
            Model parameters
        sigma2 : ndarray
            Array in which to store the conditional variances

        Returns
        -------
        sigma2 : ndarray

        Notes
        -----
        sigma2 is modified inplace.
        """
        f_resids = np.abs(self.resids) ** self.power
        s_resids = np.sign(self.resids)
        backcast = self.backcast()

        self._garch_recursion(parameters, f_resids, s_resids, backcast, sigma2)
        inv_power = 2.0 / self.power
        sigma2 **= inv_power

        return sigma2

    def fit(
        self,
        update_freq: int = 1,
        disp: bool = True,
        cov_type: Literal["robust", "classic"] = "robust",
        show_warning: bool = True,
        tol: Optional[float] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GARCHResult:
        r"""
        Estimate model parameters

        Parameters
        ----------
        update_freq : int
            Frequency of iteration updates.  Output is generated every
            `update_freq` iterations. Set to 0 to disable iterative output.
        disp : {bool, "off", "final"}
            Either 'final' to print optimization result or 'off' to display
            nothing. If using a boolean, False is "off" and True is "final"
        cov_type : str
            Estimation method of parameter covariance.  Supported options are
            'robust', which does not assume the Information Matrix Equality
            holds and 'classic' which does.  In the ARCH literature, 'robust'
            corresponds to Bollerslev-Wooldridge covariance estimator.
        show_warning : bool
            Flag indicating whether convergence warnings should be shown.
        tol : float, optional
            Tolerance for termination.
        options : dict, optional
            Options to pass to `scipy.optimize.minimize`.  Valid entries
            include 'ftol', 'eps', 'disp', and 'maxiter'.

        Returns
        -------
        results : GARCHResult
            Object containing model results

        Notes
        -----
        A ConvergenceWarning is raised if the SciPy optimizer indicates
        difficulty finding the optimum.

        Parameters are optimized using SLSQP.
        """
        self._check_scale()
        if self.scale != 1.0 and not self.scale_changed:
            self._y = cast(np.ndarray, self.scale * np.asarray(self._y_original))
            self._scale_changed = True

        sv_volatility = self.starting_values()
        sigma2 = self.compute_variance(sv_volatility, np.zeros_like(self.resids))
        std_resids = self.resids / np.sqrt(sigma2)
        sv_distribution = self.distribution.starting_values(std_resids)
        starting_values = np.hstack([sv_volatility, sv_distribution])

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

        func = self._loglikelihood
        args = (sigma2, False)
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

        vp = params[: self.num_params]
        sigma2 = self.compute_variance(vp, sigma2)
        names = self._all_parameter_names()
        model_copy = copy.deepcopy(self)

        return GARCHResult(
            params,
            None,
            cov_type,
            self.resids,
            sigma2,
            self._y_series,
            names,
            loglikelihood,
            self._is_pandas,
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
        kwargs = {"sigma2": np.zeros_like(self.resids), "individual": False}
        hess = approx_hess(params, self._loglikelihood, kwargs=kwargs)
        hess /= nobs
        inv_hess = np.linalg.inv(hess)

        if robust:
            kwargs["individual"] = True
            scores = approx_fprime(params, self._loglikelihood, kwargs=kwargs)
            score_cov = np.cov(scores, rowvar=False)
            return inv_hess @ score_cov @ inv_hess / nobs

        return inv_hess / nobs

    def _garch_recursion(
        self,
        parameters: np.ndarray,
        f_resids: np.ndarray,
        s_resids: np.ndarray,
        backcast: float,
        sigma2: np.ndarray,
    ) -> None:
        """
        Performs the recursion of the model dynamics to compute the conditional variances

        Parameters
        ----------
        parameters : ndarray
            Model parameters
        f_resids : ndarray
            Absolute value of the residuals raised to the power in the model
        s_resids : ndarray
            Signs of the residuals (-1.0, 0.0, 1.0)
        backcast : float
            Value used to initialize the recursion
        sigma2 : ndarray
            Conditional variances with the same shape as f_resids and s_resids

        Notes
        -----
        sigma2 is modified inplace.
        """
        for t in range(sigma2.shape[0]):
            sigma2[t] = parameters[0]
            loc = 1
            for i in range(self.p):
                if t - i - 1 >= 0:
                    sigma2[t] += parameters[loc] * f_resids[t - i - 1]
                else:
                    sigma2[t] += parameters[loc] * backcast
                loc += 1
            for i in range(self.o):
                if t - i - 1 >= 0:
                    sigma2[t] += (
                        parameters[loc] * f_resids[t - i - 1] * (s_resids[t - i - 1] < 0)
                    )
                else:
                    sigma2[t] += parameters[loc] * 0.5 * backcast
                loc += 1
            for i in range(self.q):
                if t - i - 1 >= 0:
                    sigma2[t] += parameters[loc] * sigma2[t - i - 1]
                else:
                    sigma2[t] += parameters[loc] * backcast
                loc += 1

    def _loglikelihood(
        self,
        parameters: np.ndarray,
        sigma2: np.ndarray,
        individual: bool = False,
    ) -> Union[float, np.ndarray]:
        """
        Compute the (negative of the) log likelihood

        Parameters
        ----------
        parameters : ndarray
            Model and distribution parameters
        sigma2 : ndarray
            Array of conditional variances
        individual : bool
            Flag indicating whether to return the vector of individual log likelihoods
            (True) or the sum (False)

        Returns
        -------
        neg_llf : float, ndarray
            The log likelihood(s) times -1.0
        """
        _callback_info["count"] += 1

        vp, dp = parameters[: self.num_params], parameters[self.num_params :]
        sigma2 = self.compute_variance(vp, sigma2)
        llf = self.distribution.loglikelihood(dp, self.resids, sigma2, individual)

        if not individual:
            _callback_info["llf"] = neg_llf = -float(llf)
            return neg_llf

        return cast(np.ndarray, -llf)

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
        sigma2 = self.compute_variance(parameters, np.zeros_like(self.resids))
        llf = self._normal.loglikelihood(np.empty(0), self.resids, sigma2)

        return float(llf)

    def _one_step_forecast(
        self, parameters: np.ndarray, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        One-step ahead forecast of the conditional variance

        Parameters
        ----------
        parameters : ndarray
            Model parameters
        horizon : int
            Forecast horizon. Must be 1 or larger.

        Returns
        -------
        sigma2 : ndarray
            Time series of estimated conditional variances
        forecasts : ndarray
            horizon-length array containing the one-step ahead forecast of the conditional
            variance in the first location and zeros in the rest of the array
        """
        t = self.resids.shape[0]
        sigma2 = np.zeros(t + 1)
        self.compute_variance(parameters, sigma2)
        forecasts = np.zeros(horizon)
        forecasts[0] = sigma2[-1]
        sigma2 = sigma2[:-1]

        return sigma2, forecasts

    def forecast(self, parameters: np.ndarray, horizon: int = 1) -> pd.Series:
        """
        Analytic multistep variance forecasts from the model

        Parameters
        ----------
        parameters : ndarray
            Model parameters
        horizon : int
            Forecast horizon. Must be 1 or larger. Forecasts are produced for horizons
            in [1, horizon]

        Returns
        -------
        forecasts : pd.Series
            Variance forecasts
        """
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be an integer >= 1")

        sigma2, forecasts = self._one_step_forecast(parameters, horizon)
        p, o, q = self.p, self.o, self.q
        omega = parameters[0]
        alpha = parameters[1 : p + 1]
        gamma = parameters[p + 1 : p + o + 1]
        beta = parameters[p + o + 1 : p + o + q + 1]

        m = max(p, o, q)
        _resids = np.zeros(m + horizon)
        _asym_resids = np.zeros(m + horizon)
        _sigma2 = np.zeros(m + horizon)
        _resids[:m] = self.resids[-m:]
        _asym_resids[:m] = _resids[:m] * (_resids[:m] < 0)
        _sigma2[:m] = sigma2[-m:]

        for h in range(horizon):
            start_loc = h + m - 1
            forecasts[h] = omega
            for i in range(p):
                forecasts[h] += alpha[i] * _resids[start_loc - i] ** 2
            for i in range(o):
                forecasts[h] += gamma[i] * _asym_resids[start_loc - i] ** 2
            for i in range(q):
                forecasts[h] += beta[i] * _sigma2[start_loc - i]

            _resids[h + m] = np.sqrt(forecasts[h])
            _asym_resids[h + m] = np.sqrt(0.5 * forecasts[h])
            _sigma2[h + m] = forecasts[h]

        index = pd.Index([f"h[{i + 1}]" for i in range(horizon)], name="horizon")

        return pd.Series(forecasts / self.scale**2, index=index, name=self._y_series.name)


class GARCHResult:
    """Results from the estimation of a GARCH model"""

    def __init__(
        self,
        params: np.ndarray,
        param_cov: Optional[np.ndarray],
        cov_type: str,
        resids: np.ndarray,
        variance: np.ndarray,
        dep_var: pd.Series,
        names: List[str],
        loglikelihood: float,
        is_pandas: bool,
        optim_output: OptimizeResult,
        model: GARCH,
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
        dep_var : pd.Series
            Dependent variable
        names : list (str)
            Model parameter names
        loglikelihood : float
            Model log likelihood
        is_pandas : bool
            Flag indicating whether the original input data is Series or DataFrame
        optim_output : OptimizeResult
            Result of the log likelihood estimation
        model : GARCH
            THe model object used to estimate the parameters
        """
        self._params = params
        self._param_cov = param_cov
        self.cov_type = cov_type
        self._resids = resids
        self._variance = variance
        self._dep_var = dep_var
        self._index = dep_var.index
        self._dep_name = dep_var.name
        self._names = list(names)
        self._loglikelihood = loglikelihood
        self._is_pandas = is_pandas
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

        # Summary header
        top_left = [
            ("Dep. Variable:", self._dep_name),
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
            ("", ""),
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

        mc = model.num_params
        dc = model.distribution.num_params
        counts = (mc, dc)
        titles = ("Model", "Distribution")
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
    def model(self) -> GARCH:
        """GARCH model instance used to produce the fit"""
        return self._model

    @cached_property
    def scale(self) -> float:
        """Scale used to adjust the data"""
        return self.model.scale

    @cached_property
    def nobs(self) -> int:
        """Number of data points used to estimate the model"""
        return self.model.resids.shape[0]

    @cached_property
    def params(self) -> pd.Series:
        """Model parameters"""
        return pd.Series(self._params, index=self._names, name="params")

    @cached_property
    def num_params(self) -> int:
        """Number of parameters in the model"""
        return len(self.params)

    @cached_property
    def resids(self) -> Union[np.ndarray, pd.Series]:
        """Model residuals in the original scale"""
        resids = self._resids / self.scale
        if self._is_pandas:
            return pd.Series(resids, index=self._index, name="resids")

        return resids

    @cached_property
    def conditional_variance(self) -> Union[np.ndarray, pd.Series]:
        """Estimated conditional variance in the original scale"""
        sigma2 = self._variance / self.scale**2
        if self._is_pandas:
            return pd.Series(sigma2, index=self._index, name="cond_var")

        return sigma2

    @property
    def conditional_volatility(self) -> Union[np.ndarray, pd.Series]:
        """Square root of the conditional variance"""
        return np.sqrt(self.conditional_variance)

    @property
    def std_resids(self) -> Union[np.ndarray, pd.Series]:
        """Residuals divided by the conditional volatility"""
        return self.resids / self.conditional_volatility

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
        self, params: Optional[Union[np.ndarray, pd.Series]] = None, horizon: int = 1
    ) -> pd.Series:
        """
        Compute forecasts of the variance using the estimated model

        Parameters
        ----------
        params : ndarray, pd.Series, None
            Alternative model parameters to use. If not provided, the parameters estimated
            when fitting the model are used
        horizon : int
            Number of steps to forecast

        Returns
        -------
        forecasts : pd.Series
            Variance forecasts
        """
        if params is None:
            params = self._params
        else:
            if (
                params.size != np.array(self._params).size
                or params.ndim != self._params.ndim
            ):
                raise ValueError("params have incorrect dimensions")

        params = np.asarray(params)

        return self.model.forecast(params, horizon)


def garch_model(
    y: Union[np.ndarray, pd.Series, pd.DataFrame],
    p: int = 1,
    o: int = 0,
    q: int = 1,
    power: float = 2.0,
    dist: str = "normal",
    rescale: Optional[bool] = None,
) -> GARCH:
    """
    Initialize a GARCH instance with provided specifications

    Parameters
    ----------
    y : ndarray, pd.Series, pd.DataFrame
        Dependent variable
    p : int
        Order of symmetric innovations
    o : int
        Order of asymmetric innovations
    q : int
        Order of lagged conditional variance
    power : float
        Power to use with the innovations, abs(resids) ** power
    dist : str
        Name of the error distribution.  Currently supported options are:
            * Normal: 'norm', 'normal', 'gaussian' (default)
            * Student's t: 't', 'stdt', 'studentst'
            * Skewed Student's t: 'skewstudent', 'skewt'
            * Generalized Error Distribution: 'ged', 'generalized error'
    rescale : bool, None
        Flag indicating whether to automatically rescale data if the scale of the
        data is likely to produce convergence issues when estimating model parameters.
        If False, the model is estimated on the data without transformation.  If True,
        then y is rescaled and the new scale is reported in the estimation results.
    """
    dist_name = dist.lower()
    if dist_name not in SUPPORTED_UNIVARIATE_DISTRIBUTIONS:
        raise ValueError(f"{dist} is not a known univariate distribution")

    if dist_name in ("ged", "generalized error"):
        d = GeneralizedError()
    elif dist_name in ("skewt", "skewstudent"):
        d = SkewStudent()
    elif dist_name in ("t", "stdt", "studentst"):
        d = StudentsT()
    else:  # normal
        d = Normal()

    return GARCH(y, p, o, q, power, d, rescale)
