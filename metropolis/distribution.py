from scipy.stats import bernoulli, norm
from typing import Dict
import numpy as np


class Distribution:
    def __init__(self):
        pass

    def get_likelihood(self, name: str, var_vals: Dict[str, float]) -> float:
        pass

    def in_support(self, sampled_val: float) -> bool:
        pass

    def get_params(self, var_vals: Dict[str, float]):
        pass

    def get_candidate_sample(self, prev_sample: float, var: float) -> float:
        candidate_sample = norm.rvs(loc=prev_sample, scale=var ** 0.5)

        return candidate_sample if self.in_support(candidate_sample) else prev_sample


class NormalDistribution(Distribution):
    def __init__(self, mean: object, var: object):
        Distribution.__init__(self)
        self.mean = mean
        self.var = var

    def get_likelihood(self, name: str, var_vals: Dict[str, float]) -> float:
        our_val = var_vals.get(name, None)

        mean, var = self.get_params(var_vals)

        if our_val is None:
            raise Exception(f'Could not find sampled value for {name}')

        if mean is None or var is None:
            raise Exception(f'Could not find values for mean and/or variance; mean = {mean}, var = {var}')

        likelihood = -np.log(var ** 0.5) - (0.5 * np.log(2 * np.pi)) - (((our_val - mean) ** 2) / (2 * var))

        return likelihood

    def get_params(self, var_vals: Dict[str, float]):
        if callable(self.mean):
            mean = self.mean(var_vals)

        else:
            mean = self.mean if not isinstance(self.mean, str) else var_vals.get(self.mean, None)

        var = self.var if not isinstance(self.var, str) else var_vals.get(self.var, None)

        return mean, var

    def in_support(self, sampled_val: float) -> bool:
        # return sampled_val > 0
        return True


class GammaDistribution(Distribution):
    def __init__(self, alpha: object, bta: object):
        Distribution.__init__(self)
        self.alpha = alpha
        self.beta = bta

    def get_likelihood(self, name: str, var_vals: Dict[str, float]) -> float:
        our_val = var_vals.get(name, None)
        alpha, bta = self.get_params(var_vals)

        if our_val is None:
            raise Exception(f'Could not find sampled value for {name}')

        if alpha is None or bta is None:
            raise Exception(f'Could not find values for alpha and/or beta; alpha = {alpha}, beta = {bta}')

        likelihood = (alpha - 1) * np.log(our_val) - (bta * our_val)

        return likelihood

    def get_params(self, var_vals: Dict[str, float]):
        if callable(self.alpha):
            alpha = self.alpha(var_vals)

        else:
            alpha = self.alpha if not isinstance(self.alpha, str) else var_vals.get(self.alpha, None)

        bta = self.beta if not isinstance(self.beta, str) else var_vals.get(self.beta, None)

        return alpha, bta

    def in_support(self, sampled_val: float) -> bool:
        return sampled_val > 0


class InverseGammaDistribution(Distribution):
    def __init__(self, alpha: object, bta: object):
        Distribution.__init__(self)
        self.alpha = alpha
        self.beta = bta

    def get_likelihood(self, name: str, var_vals: Dict[str, float]) -> float:
        our_val = var_vals.get(name, None)
        alpha, bta = self.get_params(var_vals)

        if our_val is None:
            raise Exception(f'Could not find sampled value for {name}')

        if alpha is None or bta is None:
            raise Exception(f'Could not find values for alpha and/or beta; alpha = {alpha}, beta = {bta}')

        likelihood = -(alpha + 1) * np.log(our_val) - (bta / our_val)

        return likelihood

    def get_params(self, var_vals: Dict[str, float]):
        alpha = self.alpha if not isinstance(self.alpha, str) else var_vals.get(self.alpha, None)
        bta = self.beta if not isinstance(self.beta, str) else var_vals.get(self.beta, None)

        return alpha, bta

    def in_support(self, sampled_val: float) -> bool:
        return sampled_val > 0


class PoissonDistribution(Distribution):
    def __init__(self, lmbda: object):
        Distribution.__init__(self)
        self.lmbda = lmbda

    def get_likelihood(self, name: str, var_vals: Dict[str, float]) -> float:
        our_val = var_vals.get(name, None)
        lmbda = self.get_params(var_vals)

        if our_val is None:
            raise Exception(f'Could not find sampled value for {name}')

        if lmbda is None:
            raise Exception(f'Could not find value for lambda; lambda = {lmbda}')

        likelihood = -lmbda + np.log(lmbda ** our_val) - np.log(np.math.factorial(our_val))
        # likelihood = poisson.pmf(our_val, mu=lmbda)

        return likelihood

    def get_params(self, var_vals: Dict[str, float]):
        lmbda = self.lmbda if not isinstance(self.lmbda, str) else var_vals.get(self.lmbda, None)

        return lmbda

    def get_candidate_sample(self, prev_sample: float, var: float) -> float:
        candidate_sample = float(round(norm.rvs(loc=prev_sample, scale=var)))

        return candidate_sample if self.in_support(candidate_sample) else prev_sample

    def in_support(self, sampled_val: float) -> bool:
        return sampled_val > 0 and sampled_val.is_integer()


class BetaDistribution(Distribution):
    def __init__(self, alpha: object, bta: object):
        Distribution.__init__(self)
        self.alpha = alpha
        self.beta = bta

    def get_likelihood(self, name: str, var_vals: Dict[str, float]) -> float:
        our_val = var_vals.get(name, None)
        alpha, bta = self.get_params(var_vals)

        if our_val is None:
            raise Exception(f'Could not find sampled value for {name}')

        if alpha is None or bta is None:
            raise Exception(f'Could not find values for alpha and/or beta; alpha = {alpha}, beta = {bta}')

        likelihood = (alpha - 1) * np.log(our_val) + (bta - 1) * np.log(1 - our_val)
        # likelihood = beta.pdf(our_val, a=alpha, b=bta)

        return likelihood

    def get_params(self, var_vals: Dict[str, float]):
        alpha = self.alpha if not isinstance(self.alpha, str) else var_vals.get(self.alpha, None)
        bta = self.beta if not isinstance(self.beta, str) else var_vals.get(self.beta, None)

        return alpha, bta

    def in_support(self, sampled_val: float) -> bool:
        return 0 < sampled_val <= 1


class BernoulliDistribution(Distribution):
    def __init__(self, p: object):
        Distribution.__init__(self)
        self.p = p

    def get_likelihood(self, name: str, var_vals: Dict[str, float]) -> float:
        our_val = var_vals.get(name, None)
        p = self.get_params(var_vals)

        if our_val is None:
            raise Exception(f'Could not find sampled value for {name}')

        if p is None:
            raise Exception(f'Could not find value for p; p = {p}')

        elif p == 0:
            p = 0.00001

        elif p == 1:
            p = 0.99999

        likelihood = np.log(p) if our_val == 1 else np.log(1 - p)

        return likelihood

    def get_params(self, var_vals: Dict[str, float]):
        p = self.p if not isinstance(self.p, str) else var_vals.get(self.p, None)

        return p

    def get_candidate_sample(self, prev_sample: float, var: float) -> float:
        candidate_sample = round(norm.rvs(loc=prev_sample, scale=var))

        return candidate_sample if self.in_support(candidate_sample) else prev_sample

    def in_support(self, sampled_val: float) -> bool:
        return sampled_val == 0 or sampled_val == 1


class BinomialDistribution(Distribution):
    def __init__(self, n: object, p: object):
        Distribution.__init__(self)
        self.n = n
        self.p = p

    def get_likelihood(self, name: str, var_vals: Dict[str, float]) -> float:
        our_val = var_vals.get(name, None)
        n, p = self.get_params(var_vals)

        if our_val is None:
            raise Exception(f'Could not find sampled value for {name}')

        if n is None or p is None:
            raise Exception(f'Could not find values for n and/or p; n = {n}, p = {p}')

        likelihood = bernoulli.pmf(our_val, n=n, p=p)

        return likelihood

    def get_params(self, var_vals: Dict[str, float]):
        n = self.n if not isinstance(self.n, str) else var_vals.get(self.n, None)
        p = self.p if not isinstance(self.p, str) else var_vals.get(self.p, None)

        return n, p

    def get_candidate_sample(self, prev_sample: float, var: float) -> float:
        candidate_sample = round(norm.rvs(loc=prev_sample, scale=var))

        return candidate_sample if self.in_support(candidate_sample) else prev_sample

    def in_support(self, sampled_val: float) -> bool:
        return sampled_val >= 0 and sampled_val.is_integer()
