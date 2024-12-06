import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
import pymanopt

from pymanopt.autodiff import Function
from pymanopt.manifolds.manifold import Manifold
from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult

import time
from typing import Callable, Optional


class ManifoldDOE:
    def __init__(
        self,
        manifold: Manifold,
        cost: Function,
        constraints: list[Function],
        euclidean_gradient: Optional[Callable] = None,
        riemannean_gradient: Optional[Callable] = None,
        euclidean_hessian: Optional[Callable] = None,
        preconditioner: Optional[Callable] = None,
    ) -> None:
        self.manifold = manifold
        self.cost = cost
        self.constraints = constraints
        self.euclidean_gradient = euclidean_gradient
        self.riemannean_gradient = riemannean_gradient
        self.euclidean_hessian = euclidean_hessian
        self.preconditioner = preconditioner


class ManifoldSolver(Optimizer):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    def run(
        self,
        problem: ManifoldDOE,
        initial_point: Optional[npt.NDArray[np.float64]] = None,
        **kwargs
    ) -> OptimizerResult:
        self.dim = problem.manifold.random_point().size
        self.shape = problem.manifold.random_point().shape
        step_size = max(
            self._min_step_size, 0.1) if "step_size" not in kwargs else kwargs["step_size"]
        self.K_h = step_size * anp.eye(len(problem.constraints)) * 10
        self.K_theta = step_size * anp.eye(self.dim)
        theta = self.get_start(problem, initial_point)
        self.path = [theta]
        self.costs = [problem.cost(theta)]
        self.constraints = [np.abs(np.mean([c(theta) for c in problem.constraints]))]
        self.n_iters = 0
        time_start = time.process_time()
        stopping_criterion = "Converged"
        while True:
            if self.n_iters >= self._max_iterations:
                stopping_criterion = "Reached max iterations!"
                break
            if time.process_time() - time_start > self._max_time:
                stopping_criterion = "Reached max time!"
                break
            (f_theta, h_theta) = self.get_grads(problem, theta)
            h_vals = anp.array([constraint(theta)
                               for constraint in problem.constraints])
            pi_e = self.get_parameter_vector(f_theta, h_theta, h_vals)
            dtheta = -self.K_theta @ (f_theta + h_theta.T @ pi_e)
            if anp.linalg.norm(dtheta) < self._min_gradient_norm:
                stopping_criterion = "Gradient norm too small!"
                break
            theta = problem.manifold.retraction(
                theta, step_size * dtheta.reshape(self.shape)
            )
            self.path.append(theta)
            self.costs.append(problem.cost(theta))
            self.constraints.append(np.abs(np.mean([c(theta) for c in problem.constraints])))
            self.n_iters += 1
        self.path = np.asarray(self.path)
        return OptimizerResult(
            point=self.path[-1],
            cost=problem.cost(self.path[-1]),
            iterations=self.n_iters,
            stopping_criterion=stopping_criterion,
            time=time.process_time() - time_start,
            log={"path": self.path,
                 "costs": self.costs,
                 "constraint_violations": self.constraints}
        )

    def get_start(
        self, problem: ManifoldDOE, initial_point: Optional[npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        if initial_point is None:
            np.random.seed(599)
            return problem.manifold.random_point()
        else:
            return initial_point

    def get_grads(
        self, problem: ManifoldDOE, point: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        f_theta = problem.cost.get_gradient_operator()(point)
        f_theta = problem.manifold.projection(
            point, f_theta).reshape((self.dim))
        h_theta = np.array(
            [
                problem.manifold.projection(
                    point, constraint.get_gradient_operator()(point)
                ).reshape((self.dim))
                for constraint in problem.constraints
            ]
        )
        return (f_theta, h_theta)

    def get_parameter_vector(
        self,
        f_theta: npt.NDArray[np.float64],
        h_theta: npt.NDArray[np.float64],
        h_vals: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        M1 = -np.linalg.pinv(h_theta @ self.K_theta @ h_theta.T)
        M2 = h_theta @ self.K_theta @ f_theta - self.K_h @ h_vals
        return M1 @ M2


def run_experiment(manifold: Manifold,
                   cost: Function,
                   constraints: list[Callable],
                   **kwargs) -> OptimizerResult:
    cost = pymanopt.function.autograd(manifold)(cost)
    constraints = [pymanopt.function.autograd(
        manifold)(c) for c in constraints]
    problem = ManifoldDOE(manifold, cost, constraints)
    solver = ManifoldSolver(max_iterations=kwargs.get("max_iterations"))
    result = solver.run(problem, **kwargs)
    print(f'''Point found: {result.point}; iterations run: {
          result.iterations}; stopping reason: {result.stopping_criterion}''')
    return result
