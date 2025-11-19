"""
Representation of a shard
"""
import numpy as np
import random
from rt_search.analysis_stage.shards.hyperplanes import Hyperplane
from rt_search.analysis_stage.shards.searchable import *
import pulp


class Shard(Searchable):
    def __init__(self, A, b, symbols):
        self.A = A
        self.b = b
        self.symbols = symbols

    def in_space(self, point: Position) -> bool:
        point = np.array(point.sorted().values())
        return np.all(self.A @ point >= self.b)

    def calc_delta(self, start: Position, trajectory: Position) -> float:
        # TODO: Use code in notebook
        raise NotImplementedError()

    def get_interior_point(self, suspected_point: Optional[Position] = None) -> Position:
        """
        Find an interior point in the shard.
        :param suspected_point: a point that is within the shard bounds (might be generated in the extractor)
        :return: An interior point
        """
        def find_integer_point_milp(A, b, xmin, xmax) -> Optional[np.ndarray]:
            """
            Use PuLP MILP CBC solver to find feasible point
            :param A: Original hyperplane constraints (linear terms)
            :param b: Original hyperplane constraints (free terms)
            :param xmin: minimum bound on each variable
            :param xmax: maximum bound on each variable
            :return: Vector representing the feasible point
            """
            m, d = A.shape
            prob = pulp.LpProblem('find_int_point', pulp.LpStatusOptimal)
            vars = [
                pulp.LpVariable(f'x{i}', lowBound=int(xmin[i]), upBound=int(xmax[i]), cat='Integer')
                for i in range(d)
            ]

            # no objective, just feasibility: add 0 objective
            prob += 0
            for i in range(m):
                prob += pulp.lpSum(A[i, j] * vars[j] for j in range(d)) <= b[i]
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            if pulp.LpStatus[prob.status] != 'Optimal':
                return None
            return np.array([int(v.value()) for v in vars], dtype=int)

        if suspected_point is None:
            xmin = list(-5 * np.ones((len(self.symbols,))))
            xmax = list(5 * np.ones((len(self.symbols,))))
        else:
            xmin = [-3 + suspected_point[sym] for sym in self.symbols]
            xmax = [3 + suspected_point[sym] for sym in self.symbols]

        interior_pt = find_integer_point_milp(self.A, self.b, xmin, xmax)
        if interior_pt is None:
            raise Exception('No interior point')
        if not np.all(self.A @ interior_pt.T <= self.b):
            raise Exception('Invalid result!')
        return Position({sym: v for sym, v in zip(interior_pt, self.symbols)})

    # TODO: use MCMC in order to sample trajectories uniformly using R calculated the formula in the old extractor
    #   using matrix A and b=0 because calculating directions here.
    # TODO: later do the changes for shifts! - directions with no shiff but general shard with!
    def sample_trajectory(self, n_samples=Optional[int]) -> List[Position]:
        raise NotImplementedError()

    def __sample_trajectory(self):
        error = 1e-12

        def neighbors(x, A, b, R=None, neighbor_radius=1):
            # Return list of neighbor integer vectors of x given move set: coordinate +/-1
            # Optionally restrict to points within Euclidean distance R from origin (or some ref).
            d = len(x)
            neighs = []
            for j in range(d):
                for s in (-1, +1):
                    y = x.copy()
                    y[j] += s
                    if R is not None and np.linalg.norm(y) > R + error:
                        continue
                    if feasible_integer(y, A, b):
                        neighs.append(y)
            return neighs

        def integer_mh_sampler(A, b, x0, n_samples, burn=1000, thin=1, R=None):
            """
            Metropolis-Hastings sampler on integer lattice with target uniform over feasible integer points in intersection (bounded by R if given).
            - A,b define Ax <= b constraints
            - x0 initial feasible integer point (np.array int)
            """
            d = A.shape[1]
            x = x0.copy()
            samples = []
            total = burn + n_samples * thin
            for step in range(total):
                # propose neighbor by picking a coordinate and sign
                j = random.randrange(d)
                s = random.choice([-1, 1])
                y = x.copy();
                y[j] += s
                if R is not None and np.linalg.norm(y) > R + 1e-12:
                    accept = False
                elif not self.in_space(y, A, b):
                    accept = False
                else:
                    # compute degrees (number of feasible neighbors of each)
                    deg_x = len(neighbors(x, A, b, R=R))
                    deg_y = len(neighbors(y, A, b, R=R))
                    if deg_y == 0:
                        # shouldn't happen if y feasible, but guard
                        accept = False
                    else:
                        # Metropolis acceptance to target uniform: prob = min(1, deg_x / deg_y)
                        alpha = min(1.0, deg_x / deg_y) if deg_y > 0 else 0.0
                        accept = (random.random() < alpha)
                if accept:
                    x = y
                if step >= burn and ((step - burn) % thin == 0):
                    samples.append(x.copy())
            return np.array(samples)

        return None

    @staticmethod
    def generate_matrices(
            hyperplanes: List[sp.Expr], above_below_indicator: List[int]
    ) -> Tuple[np.ndarray, np.array, List[sp.Symbol]]:
        if (l_hps := len(hyperplanes)) != (l_ind := len(above_below_indicator)):
            raise ValueError(f"Number of hyperplanes does not match number of indicators {l_hps}!={l_ind}")
        if any(ind != 1 and ind != -1 for ind in above_below_indicator):
            raise ValueError(f"Indicators vector must be 1 (above) or -1 (below)")

        symbols = set()
        for hyperplane in hyperplanes:
            symbols.union(hyperplane.free_symbols)
        symbols = list(symbols)
        vectors = []
        free_terms = []

        for expr, ind in zip(hyperplanes, above_below_indicator):
            hp = Hyperplane(expr, symbols)
            if ind == 1:
                v, free = hp.as_above_vector
            else:
                v, free = hp.as_below_vector
            free_terms.append(free)
            vectors.append(v)
        return np.vstack(tuple(vectors)), np.array(free_terms), symbols


if __name__ == '__main__':
    a = np.array([[1, 2], [3, 4]])
    b = np.array([1,1])
    print(a @ b.T)