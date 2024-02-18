from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.special import gamma, gammaln, kv

from timeseries.linalg import matrix_power3d


__all__ = [
    "Distribution",
    "Normal",
    "StudentsT",
    "SkewStudent",
    "GeneralizedError",
    "MultivariateDistribution",
    "MultivariateNormal",
    "MultivariateStudentsT",
    "MultivariateLaplace",
    "MultivariateSkewStudent",
    "SUPPORTED_UNIVARIATE_DISTRIBUTIONS",
]

SUPPORTED_UNIVARIATE_DISTRIBUTIONS = [
    "norm",
    "normal",
    "gaussian",
    "t",
    "stdt",
    "studentst",
    "skewt",
    "skewstudent",
    "ged",
    "generalized error",
]


class Distribution(metaclass=ABCMeta):
    """Template for subclassing only"""

    def __init__(self) -> None:
        self._name = ""
        self.num_params: int = 0

    def __str__(self) -> str:
        return self.name + " distribution"

    def __repr__(self) -> str:
        return self.__str__() + ", id: " + hex(id(self))

    @property
    def name(self) -> str:
        """Name of the distribution"""
        return self._name

    @abstractmethod
    def parameter_names(self) -> List[str]:
        """
        Returns the names of the distribution shape parameters

        Returns
        -------
        names : list (str)
            Parameter names
        """

    @abstractmethod
    def bounds(self) -> List[Tuple[float, float]]:
        """
        Bounds for the distribution shape parameters for use in optimization

        Returns
        -------
        bounds : list (tuple)
            List of parameter bounds with elements of the form (lower, upper)
        """

    @abstractmethod
    def constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct linear parameter constraints arrays for use in optimization

        Returns
        -------
        a : ndarray
            Constraint loadings
        b : ndarray
            Constraint values

        Notes
        -----
        Parameters satisfy the constraints a.dot(parameters) - b
        """

    @abstractmethod
    def starting_values(self, std_resids: np.ndarray) -> np.ndarray:
        """
        Compute starting values for the distribution shape parameters

        Parameters
        ----------
        std_resids : ndarray
            Approximate standardized residuals from the model

        Returns
        -------
        sv : ndarray
            Starting values
        """

    @abstractmethod
    def loglikelihood(
        self,
        parameters: np.ndarray,
        resids: np.ndarray,
        sigma2: np.ndarray,
        individual: bool = False,
    ) -> Union[float, np.ndarray]:
        """
        Log likelihood evaluation

        Parameters
        ----------
        parameters : ndarray
            Distribution shape parameters
        resids : ndarray
            nobs array of model residuals
        sigma2 : ndarray
            nobs array of conditional variances
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        llf : float, ndarray
            The log likelihood(s)
        """


class Normal(Distribution):
    """Standard normal distribution for use in univariate ARCH models"""

    def __init__(self) -> None:
        super().__init__()
        self._name = "Normal"

    def parameter_names(self) -> List[str]:
        return []

    def bounds(self) -> List[Tuple[float, float]]:
        return []

    def constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.empty(0), np.empty(0)

    def starting_values(self, std_resids: np.ndarray) -> np.ndarray:
        return np.empty(0)

    def loglikelihood(
        self,
        parameters: np.ndarray,
        resids: np.ndarray,
        sigma2: np.ndarray,
        individual: bool = False,
    ) -> Union[float, np.ndarray]:
        lls = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + resids**2 / sigma2)

        return lls if individual else np.sum(lls)


class StudentsT(Distribution):
    """
    Standardized Student's t distribution for use in univariate ARCH models
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "Standardized Student's t"
        self.num_params: int = 1

    def parameter_names(self) -> List[str]:
        return ["nu"]

    def bounds(self) -> List[Tuple[float, float]]:
        return [(2.05, 500.0)]

    def constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([[1], [-1]]), np.array([2.05, -500.0])

    def starting_values(self, std_resids: np.ndarray) -> np.ndarray:
        k = stats.kurtosis(std_resids, fisher=False)
        sv = max((4.0 * k - 6.0) / (k - 3.0) if k > 3.75 else 12.0, 4.0)

        return np.array([sv])

    def loglikelihood(
        self,
        parameters: np.ndarray,
        resids: np.ndarray,
        sigma2: np.ndarray,
        individual: bool = False,
    ) -> Union[float, np.ndarray]:
        nu = parameters[0]
        lls = gammaln((nu + 1) / 2) - gammaln(nu / 2) - np.log(np.pi * (nu - 2)) / 2
        lls -= 0.5 * np.log(sigma2)
        lls -= (nu + 1) / 2 * np.log(1 + resids**2 / (sigma2 * (nu - 2)))

        return lls if individual else np.sum(lls)


class SkewStudent(Distribution):
    """
    Standardized Skew Student's t distribution for use in univariate ARCH models
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "Standardized Skew Student's t"
        self.num_params: int = 2

    def parameter_names(self) -> List[str]:
        return ["eta", "lambda"]

    def bounds(self) -> List[Tuple[float, float]]:
        return [(2.05, 300.0), (-1.0, 1.0)]

    def constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        a = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b = np.array([2.05, -300.0, -1, -1])

        return a, b

    def starting_values(self, std_resids: np.ndarray) -> np.ndarray:
        k = stats.kurtosis(std_resids, fisher=False)
        sv = max((4.0 * k - 6.0) / (k - 3.0) if k > 3.75 else 12.0, 4.0)

        return np.array([sv, 0.0])

    def loglikelihood(
        self,
        parameters: np.ndarray,
        resids: np.ndarray,
        sigma2: np.ndarray,
        individual: bool = False,
    ) -> Union[float, np.ndarray]:
        eta, lam = parameters
        const_c = (
            gammaln((eta + 1) / 2) - gammaln(eta / 2) - np.log(np.pi * (eta - 2)) / 2
        )
        const_a = 4 * lam * np.exp(const_c) * (eta - 2) / (eta - 1)
        const_b = np.sqrt(1 + 3 * lam**2 - const_a**2)

        std_resids = resids / np.sqrt(sigma2)
        lls = np.log(const_b) + const_c - np.log(sigma2) / 2
        if np.abs(lam) >= 1.0:
            lam = np.sign(lam) * (1.0 - 1e-6)

        llf_resid = np.square(
            (const_b * std_resids + const_a)
            / (1 + np.sign(std_resids + const_a / const_b) * lam)
        )
        lls -= (eta + 1) / 2 * np.log(1 + llf_resid / (eta - 2))

        return lls if individual else np.sum(lls)


class GeneralizedError(Distribution):
    """
    Generalized Error Distribution (GED) for use in univariate ARCH models
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "Generalized Error"
        self.num_params: int = 1

    def parameter_names(self) -> List[str]:
        return ["nu"]

    def bounds(self) -> List[Tuple[float, float]]:
        return [(1.01, 500.0)]

    def constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([[1], [-1]]), np.array([1.01, -500.0])

    def starting_values(self, std_resids: np.ndarray) -> np.ndarray:
        return np.array([1.5])

    def loglikelihood(
        self,
        parameters: np.ndarray,
        resids: np.ndarray,
        sigma2: np.ndarray,
        individual: bool = False,
    ) -> Union[float, np.ndarray]:
        nu = parameters[0]
        log_c = 0.5 * (-2 / nu * np.log(2) + gammaln(1 / nu) - gammaln(3 / nu))
        c = np.exp(log_c)

        lls = np.log(nu) - log_c - gammaln(1 / nu) - (1 + 1 / nu) * np.log(2)
        lls -= 0.5 * np.log(sigma2)
        lls -= 0.5 * np.abs(resids / (np.sqrt(sigma2) * c)) ** nu

        return lls if individual else np.sum(lls)


class MultivariateDistribution(metaclass=ABCMeta):
    """Template for subclassing only"""

    def __init__(self, ndim: Optional[int] = None) -> None:
        self._ndim = ndim
        self._name = ""
        self.num_params: int = 0

    def __str__(self) -> str:
        return self.name + " distribution"

    def __repr__(self) -> str:
        return self.__str__() + ", id: " + hex(id(self))

    @property
    def name(self) -> str:
        """Name of the distribution"""
        return self._name

    @property
    def ndim(self) -> Union[int, None]:
        """Set or get the number of dimensions of the distribution"""
        return self._ndim

    @ndim.setter
    def ndim(self, value: Union[int, np.integer]) -> None:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("Must be an integer >= 1")
        self._ndim = int(value)

    @abstractmethod
    def parameter_names(self) -> List[str]:
        """
        Returns the names of the distribution shape parameters

        Returns
        -------
        names : list (str)
            Parameter names
        """

    @abstractmethod
    def bounds(self) -> List[Tuple[float, float]]:
        """
        Bounds for the distribution shape parameters for use in optimization

        Returns
        -------
        bounds : list (tuple)
            List of parameter bounds with elements of the form (lower, upper)
        """

    @abstractmethod
    def constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct linear parameter constraints arrays for use in optimization

        Returns
        -------
        a : ndarray
            Constraint loadings
        b : ndarray
            Constraint values

        Notes
        -----
        Parameters satisfy the constraints a.dot(parameters) - b
        """

    @abstractmethod
    def starting_values(self, std_resids: np.ndarray) -> np.ndarray:
        """
        Compute starting values for the distribution shape parameters

        Parameters
        ----------
        std_resids : ndarray
            Approximate standardized residuals from the model

        Returns
        -------
        sv : ndarray
            Starting values
        """

    @abstractmethod
    def loglikelihood(
        self,
        parameters: np.ndarray,
        resids: np.ndarray,
        cov: np.ndarray,
        individual: bool = False,
    ) -> Union[float, np.ndarray]:
        """
        Log likelihood evaluation

        Parameters
        ----------
        parameters : ndarray
            Distribution shape parameters
        resids : ndarray
            Model residuals with dimensions T x K
        cov : ndarray
            Conditional covariances with dimensions of the form T x K x K
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        llf : float, ndarray
            The log likelihood(s)
        """


class MultivariateNormal(MultivariateDistribution):
    """Multivariate normal distribution for use in multivariate ARCH models"""

    def __init__(self, ndim: Optional[int] = None) -> None:
        """
        Parameters
        ----------
        ndim : int, None
            Number of dimensions
        """
        super().__init__(ndim)
        self._name = "Multivariate Normal"

    def parameter_names(self) -> List[str]:
        return []

    def bounds(self) -> List[Tuple[float, float]]:
        return []

    def constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.empty(0), np.empty(0)

    def starting_values(self, std_resids: np.ndarray) -> np.ndarray:
        return np.empty(0)

    def loglikelihood(
        self,
        parameters: np.ndarray,
        resids: np.ndarray,
        cov: np.ndarray,
        individual: bool = False,
    ) -> Union[float, np.ndarray]:
        lls = -0.5 * resids.shape[1] * np.log(2 * np.pi)
        lls -= 0.5 * np.log(np.linalg.det(cov))
        lls -= 0.5 * np.einsum("ij,ijk,ik->i", resids, np.linalg.inv(cov), resids)

        return lls if individual else np.sum(lls)


class MultivariateStudentsT(MultivariateDistribution):
    """
    Multivariate standardized Student's t distribution for use in multivariate ARCH models
    """

    def __init__(self, ndim: Optional[int] = None) -> None:
        """
        Parameters
        ----------
        ndim : int, None
            Number of dimensions
        """
        super().__init__(ndim)
        self._name = "Multivariate Student's t"
        self.num_params: int = 1

    def parameter_names(self) -> List[str]:
        return ["nu"]

    def bounds(self) -> List[Tuple[float, float]]:
        return [(2.05, 500.0)]

    def constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([[1], [-1]]), np.array([2.05, -500.0])

    def starting_values(self, std_resids: np.ndarray) -> np.ndarray:
        kurt = stats.kurtosis(std_resids)
        ndim = std_resids.shape[1]
        kappa = np.maximum(-2 / (ndim + 2) * 0.99, kurt / 3)
        avg_kappa = np.mean(kappa)

        return np.array([2 / avg_kappa + 4])

    def loglikelihood(
        self,
        parameters: np.ndarray,
        resids: np.ndarray,
        cov: np.ndarray,
        individual: bool = False,
    ) -> Union[float, np.ndarray]:
        nu = parameters[0]
        ndim = resids.shape[1]
        ll_term = np.einsum("ij,ijk,ik->i", resids, np.linalg.inv(cov), resids)

        lls = (
            gammaln((nu + ndim) / 2)
            - gammaln(nu / 2)
            - ndim / 2 * np.log(np.pi * (nu - 2))
        )
        lls -= 0.5 * np.log(np.linalg.det(cov))
        lls -= (nu + ndim) / 2 * np.log(1 + ll_term / (nu - 2))

        return lls if individual else np.sum(lls)


class MultivariateLaplace(MultivariateDistribution):
    """
    Multivariate Laplace distribution for use in multivariate ARCH models
    """

    def __init__(self, ndim: Optional[int] = None) -> None:
        """
        Parameters
        ----------
        ndim : int, None
            Number of dimensions
        """
        super().__init__(ndim)
        self._name = "Multivariate Laplace"

    def parameter_names(self) -> List[str]:
        return []

    def bounds(self) -> List[Tuple[float, float]]:
        return []

    def constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.empty(0), np.empty(0)

    def starting_values(self, std_resids: np.ndarray) -> np.ndarray:
        return np.empty(0)

    def loglikelihood(
        self,
        parameters: np.ndarray,
        resids: np.ndarray,
        cov: np.ndarray,
        individual: bool = False,
    ) -> Union[float, np.ndarray]:
        k = resids.shape[1]
        nu = (2 - k) / 2
        ll_term = np.einsum("ij,ijk,ik->i", resids, np.linalg.inv(cov), resids)

        lls = -np.log(0.5 * (2 * np.pi) ** (k / 2) * np.sqrt(np.linalg.det(cov)))
        lls += nu / 2 * np.log(0.5 * ll_term)
        lls += np.log(kv(nu, np.sqrt(2 * ll_term)))

        return lls if individual else np.sum(lls)


class MultivariateSkewStudent(MultivariateDistribution):
    def __init__(self, ndim: int) -> None:
        if not isinstance(ndim, (int, np.integer)) or ndim < 1:
            raise ValueError("ndim must be an integer >= 1")
        super().__init__(ndim)
        self._name = "Multivariate Skew Student's t"
        self.num_params: int = 1 + self.ndim

    def parameter_names(self) -> List[str]:
        return ["eta"] + [f"lambda[{i + 1}]" for i in range(self.ndim)]

    def bounds(self) -> List[Tuple[float, float]]:
        return [(2.05, 300.0)] + [(-1.0, 1.0)] * self.ndim

    def constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        a = np.zeros((self.num_params * 2, self.num_params))
        loc, value = 0, 1
        for i in range(self.num_params * 2):
            a[i, loc] = value
            if value == -1:
                loc += 1
            value *= -1

        b = np.array([2.05, -300.0] + [-1, -1] * self.ndim)

        return a, b

    def starting_values(self, std_resids: np.ndarray) -> np.ndarray:
        kurt = stats.kurtosis(std_resids)
        ndim = std_resids.shape[1]
        kappa = np.maximum(-2 / (ndim + 2) * 0.99, kurt / 3)
        avg_kappa = np.mean(kappa)

        eta = 2 / avg_kappa + 4
        skew = stats.skew(std_resids)
        lam = np.sqrt(eta / (eta - 2)) * skew

        return np.hstack([eta, lam])

    def loglikelihood(
        self,
        parameters: np.ndarray,
        resids: np.ndarray,
        cov: np.ndarray,
        individual: bool = False,
    ) -> Union[float, np.ndarray]:
        eta, lam = parameters[0], parameters[1:]
        hm12 = matrix_power3d(cov, -0.5)
        z = np.einsum("ijk,ik->ij", hm12, resids)
        omega = self._const_omega(parameters)
        xi = self._const_xi(parameters, omega)
        q = self._const_q(z, xi, omega)
        t_pdf_term = (hm12 - xi) @ lam * np.sqrt((eta + self.ndim) / (q + eta))

        lls = np.log(2) + np.log(self._t_d(parameters, omega, q))
        lls += np.log(stats.t.pdf(t_pdf_term, df=eta + self.ndim))
        lls -= 0.5 * np.log(np.linalg.det(cov))

        return lls if individual else np.sum(lls)

    def _t_d(
        self, parameters: np.ndarray, omega: np.ndarray, q: np.ndarray
    ) -> np.ndarray:
        eta = parameters[0]
        lhs_numerator = gamma((eta + self.ndim) / 2)
        lhs_denominator = (
            np.sqrt(np.linalg.det(omega))
            * (np.pi * eta) ** (self.ndim / 2)
            * gamma(eta / 2)
        )
        rhs = (1 + q / eta) ** (-(eta + self.ndim) / 2)

        return lhs_numerator / lhs_denominator * rhs

    def _const_omega(self, parameters: np.ndarray) -> np.ndarray:
        eta, lam = parameters[0], parameters[1:]
        if np.all(lam == 0):
            return (eta - 2) / eta * np.eye(self.ndim)

        numerator = np.pi * gamma(eta / 2) ** 2 * (eta - (eta - 2) * lam.T @ lam)
        denominator = (
            2
            * (lam.T @ lam)
            * (eta - 2)
            * (np.pi * gamma(eta / 2) ** 2 - (eta - 2) * gamma((eta - 1) / 2) ** 2)
        )
        inner_term = -1 + numerator / denominator * (-1 + self._const_k(parameters))
        return (
            (eta - 2)
            / eta
            * (np.eye(self.ndim) + 1 / (lam.T @ lam) * inner_term * np.outer(lam, lam))
        )

    @staticmethod
    def _const_xi(parameters: np.ndarray, omega: np.ndarray) -> np.ndarray:
        eta, lam = parameters[0], parameters[1:]
        rhs = omega @ lam / np.sqrt(1 + lam.T @ omega @ lam)
        lhs = -np.sqrt(eta / np.pi) * gamma((eta - 1) / 2) / gamma(eta / 2)

        return lhs * rhs

    @staticmethod
    def _const_q(z: np.ndarray, xi: np.ndarray, omega: np.ndarray) -> np.ndarray:
        return np.einsum("ij,jk,ik->i", z - xi, np.linalg.inv(omega), z - xi)

    @staticmethod
    def _const_k(parameters: np.ndarray) -> float:
        eta, lam = parameters[0], parameters[1:]
        numerator = (
            4
            * eta
            * (eta - 2)
            * (np.pi * gamma(eta / 2) ** 2 - (eta - 2) * gamma((eta - 1) / 2) ** 2)
            * lam.T
            @ lam
        )
        denominator = np.pi * gamma(eta / 2) ** 2 * (eta - (eta - 2) * lam.T @ lam) ** 2

        return float(np.sqrt(1 + numerator / denominator))
