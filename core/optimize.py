import numpy as np
from scipy.optimize import minimize


class WeightOptimize:

    """
    Class for calculating semi-optimized asset weights based on given historical returns and asset covariance matrix.
    Assumption: expected return = historical mean return -> semi-optimized but probably better than random or naive.
    """

    def __init__(
            self,
            asset_returns,
            covariance_matrix,
            nb_assets=10,
            risk_aversion=1
    ):

        self.asset_returns = asset_returns
        self.covariance_matrix = covariance_matrix
        self.nb_assets = nb_assets
        self.risk_aversion = risk_aversion
        self.expected_return = np.mean(self.asset_returns, axis=0)

    def objective_function(self, x):

        # utility = expected return - exposure to risk
        return self.risk_aversion / 2 * float((np.matmul(np.matmul(x.T, self.covariance_matrix), x))) - \
               float(np.matmul(x.T, self.expected_return))

    def optimize_weights(self, method='SLSQP'):

        """
        Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        method: optimization method (see link above)
        x0: initial guess

        returns: optimized weights for each asset
        """

        # bnds: lower and upper bound for the weights -> between 0 and 1
        bnds = tuple((0, 1) for _ in range(self.nb_assets))

        # using naive weights for initial guess
        x0 = np.array([1 / self.nb_assets for _ in range(self.nb_assets)])

        # the sum of all weights has to be equal one
        constrain = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # calculate 'optimal' portfolio weights
        solution = minimize(self.objective_function, x0, method=method, constraints=constrain, bounds=bnds)

        return solution.x


# TODO currently not working properly
class DynamicOptimize:

    """
    Optimizer class for dynamic optimizer agent. Doesn't require predictions on future returns.
    Objective function: E[R] - y 1/2 Var[R] - c -> variable x = new asset weights
    """

    def __init__(self, weights, asset_returns, covariance_matrix, nb_assets=10, risk_aversion=1, cost=0.05):
        self.weights = weights
        self.asset_returns = asset_returns
        self.covariance_matrix = covariance_matrix
        self.nb_assets = nb_assets
        self.risk_aversion = risk_aversion
        self.cost = cost
        self.expected_return = np.mean(self.asset_returns, axis=0)

    def objective_function(self, x):
        # utility = expected return - exposure to risk - cost for weight shifting
        return self.risk_aversion / 2 * float((np.matmul(np.matmul(x.T, self.covariance_matrix), x))) - \
               float(np.matmul(x.T, self.expected_return)) + \
               self.cost * np.sum(np.absolute(self.weights.T - x.T))

    def optimize_weights(self, method='SLSQP'):

        """
        Non linear programming using SciPy:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        method: optimization method (see link above)
        bnds: lower and upper bound for the weights -> between 0 and 1
        x0: initial guess

        Return: optimized weights each asset
        """

        bnds = tuple((0, 1) for _ in range(self.nb_assets))
        x0 = np.array([1 / self.nb_assets for _ in range(self.nb_assets)])  # using naive weights for initial guess
        constrain = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}   # the sum of all weights has to be equal one
        solution = minimize(self.objective_function, x0, method=method, constraints=constrain, bounds=bnds)
        return solution.x


def test_weight_optimizer():
    areturn = np.random.rand(50,10)
    covar = np.random.rand(10,10)
    optimized_weights = WeightOptimize(areturn, covar)
    return optimized_weights.optimize_weights()


def test_dynamic_optimizer():
    aretrun = np.random.rand(50,10)
    covar = np.random.rand(10,10)
    weights = np.array([1/10 for _ in range(10)])
    dynamic = DynamicOptimize(weights, aretrun, covar)
    return dynamic.optimize_weights()


if __name__ == '__main__':
    # just to test if the optimization is working as intended
    # w = test_weight_optimizer()
    w = test_dynamic_optimizer()
    print('New weights:\n', w)
