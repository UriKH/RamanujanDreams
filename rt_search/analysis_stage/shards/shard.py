"""
Representation of a shard
"""
import numpy as np
import math
from functools import reduce
from rt_search.analysis_stage.shards.hyperplanes import Hyperplane
from rt_search.analysis_stage.shards.searchable import *
from rt_search.utils.caching import *
import pulp
from typing import Union, Set, Iterator


class Shard(Searchable):
    def __init__(self,
                 A: np.ndarray,
                 b: np.array,
                 group: Tuple[sp.Symbol, ...],
                 shift: Position,
                 symbols: List[sp.Symbol]):
        """
        :param A: Matrix A defining the linear terms in the inequalities
        :param b: Vector b defining the free terms in the inequalities
        :param group: The shard group this shard is part of
        :param shift: The shift in start points required
        :param symbols: Symbols used by the CMF which this shard is part of
        """
        self.A = A
        self.b = b
        self.group = group
        self.symbols = symbols
        self.shift = np.array([shift[sym] for sym in self.symbols])

    def in_space(self, point: Position) -> bool:
        point = np.array(point.sorted().values())
        return np.all(self.A @ point >= self.b)

    def calc_delta(self, start: Position, trajectory: Position) -> float:
        # TODO: Use code in notebook
        raise NotImplementedError()

    def get_interior_point(self, suspected_point: Optional[Position] = None) -> Optional[Position]:
        """
        Find an interior point in the shard.
        :param suspected_point: a point that is within the shard bounds (might be generated in the extractor)
        :return: An interior point
        """
        if suspected_point is None:
            xmin = list(-5 * np.ones((len(self.symbols,))))
            xmax = list(5 * np.ones((len(self.symbols,))))
        else:
            xmin = [-3 + suspected_point[sym] for sym in self.symbols]
            xmax = [3 + suspected_point[sym] for sym in self.symbols]

        interior_pt = self.__find_integer_point_milp(self.A, self.b, xmin, xmax)
        if interior_pt is None:
            raise Exception('No interior point')
        if not np.all(self.A @ interior_pt.T <= self.b):
            raise Exception('Invalid result!')
        return Position({sym: v for sym, v in zip(interior_pt, self.symbols)})

    # TODO: use MCMC in order to sample trajectories uniformly using R calculated the formula in the old extractor
    #   using matrix A and b=0 because calculating directions here.
    # TODO: later do the changes for shifts! - directions with no shiff but general shard with!
    def sample_trajectory(self, n_samples=Optional[int]) -> List[Position]:
        """
        TODO: Implement this!
            Demands: gcd(coords) = 1, opt(within R radius), uniform sampling, dims 3-20
            Options: IHR (probably best). dim < 6 use Barvinok / LattE
        """

        def vec_gcd(u: np.ndarray) -> int:
            arr = np.abs(u.astype(int))
            nz = arr[arr != 0]
            if nz.size == 0:
                return 0
            return reduce(math.gcd, nz.tolist())

        def inside_cone(A: np.ndarray, u: np.ndarray) -> bool:
            # return True if A @ u <= 0 (elementwise)
            return np.all(A.dot(u) <= 0)

        def compute_t_interval(A: np.ndarray, u: np.ndarray, d: np.ndarray, R: int) -> Optional[Tuple[int, int]]:
            """
            For inequalities A (u + t d) <= 0 find integer interval [t_min, t_max]
            satisfying all row constraints and box ||u + t d||_inf <= R.
            Return None if empty.
            """
            # Start with wide interval
            tmin = -1e18
            tmax = 1e18

            # cone constraints A_i (u + t d) <= 0  => (A_i d) * t <= -A_i u
            Au = A.dot(u)
            Ad = A.dot(d)

            for i in range(A.shape[0]):
                ai_dot_d = Ad[i]
                rhs = -Au[i]  # ai^T u + t*(ai^T d) <= 0 -> t*(ai^T d) <= -ai^T u

                if ai_dot_d == 0:
                    # requires rhs >= 0 (i.e. ai^T u <= 0), otherwise infeasible for any t
                    if rhs < 0:
                        return None
                    else:
                        continue

                # For ai_dot_d > 0: t <= floor(rhs / ai_dot_d)
                if ai_dot_d > 0:
                    tt = math.floor(rhs / ai_dot_d)
                    if tt < tmax:
                        tmax = tt
                else:
                    # ai_dot_d < 0: t >= ceil(rhs / ai_dot_d)
                    tt = math.ceil(rhs / ai_dot_d)
                    if tt > tmin:
                        tmin = tt

                if tmin > tmax:
                    return None

            # box constraints: for each coordinate j: -R <= u_j + t d_j <= R
            for j in range(u.size):
                dj = int(d[j])
                uj = int(u[j])
                if dj == 0:
                    if abs(uj) > R:
                        return None
                    else:
                        continue
                # uj + t*dj <= R -> t <= floor((R - uj) / dj) if dj > 0, else t >= ceil((R - uj)/dj)
                # uj + t*dj >= -R -> t >= ceil((-R - uj) / dj) if dj > 0, else t <= floor((-R - uj)/dj)
                if dj > 0:
                    tmax = min(tmax, math.floor((R - uj) / dj))
                    tmin = max(tmin, math.ceil((-R - uj) / dj))
                else:
                    # dj < 0
                    tmax = min(tmax, math.floor((-R - uj) / dj))
                    tmin = max(tmin, math.ceil((R - uj) / dj))

                if tmin > tmax:
                    return None

            # convert to ints
            tmin = int(tmin)
            tmax = int(tmax)
            if tmin > tmax:
                return None
            return tmin, tmax

        def random_integer_direction(n: int, max_coord: int) -> np.ndarray:
            """
            Produce a random integer direction d with coordinates in [-max_coord, max_coord],
            and gcd(d) == 1 (so steps by t preserve lattice coverage).
            Avoid d = 0 vector.
            """
            while True:
                d = np.random.randint(-max_coord, max_coord + 1, size=n)
                if np.all(d == 0):
                    continue
                if vec_gcd(d) != 1:
                    # allow non-primitive directions too if you want, but primitive is safer for coverage
                    # skip to ensure minimal step gcd = 1
                    continue
                return d.astype(int)

        def integer_hit_and_run(
                A: np.ndarray,
                R: int,
                u0: np.ndarray,
                *,
                max_coord_direction: Optional[int] = None,
                tries_per_step: int = 20
        ) -> Iterator[np.ndarray]:
            """
            Generator yielding successive integer lattice points u satisfying A u <= 0, ||u||_inf <= R
            and gcd(u) == 1, using integer hit-and-run steps.
            - A: m x n numpy array
            - R: bounding box radius (infinity norm)
            - u0: initial feasible primitive vector (numpy int array)
            - max_coord_direction: max absolute coordinate for random direction sampling (default ~R or 2)
            - tries_per_step: how many directions to try before yielding the same point (avoid infinite loops)
            """
            n = A.shape[1]
            if max_coord_direction is None:
                max_coord_direction = max(1, min(2 * R, 5))

            u = u0.copy().astype(int)
            if not inside_cone(A, u) or vec_gcd(u) != 1 or np.max(np.abs(u)) > R:
                raise ValueError("u0 is not a feasible primitive point within box.")

            while True:
                moved = False
                for attempt in range(tries_per_step):
                    d = random_integer_direction(n, max_coord_direction)
                    interval = compute_t_interval(A, u, d, R)
                    if interval is None:
                        continue
                    tmin, tmax = interval
                    # note: could be large; sample uniform integer in [tmin, tmax]
                    if tmin > tmax:
                        continue
                    # if interval is tiny (single point t=0), maybe skip
                    # sample t uniformly:
                    t = np.random.randint(tmin, tmax + 1)
                    u_new = u + int(t) * d
                    # optional: enforce primitive
                    if vec_gcd(u_new) != 1:
                        # try to find any t in interval giving gcd=1 before giving up
                        # naive attempt: sample a few times, otherwise continue attempts
                        found = False
                        for _ in range(6):
                            t = np.random.randint(tmin, tmax + 1)
                            u_try = u + int(t) * d
                            if vec_gcd(u_try) == 1:
                                u_new = u_try
                                found = True
                                break
                        if not found:
                            continue
                    # accept (uniform in the allowed t-interval)
                    u = u_new
                    moved = True
                    break

                # yield current state (moved or not). Caller can decide whether to treat stationary steps as samples.
                yield u.copy()

                # if we failed to move after tries_per_step attempts, we still yield the current u (chain can stay put)

    # TODO: remove this if not used...
    # from scipy.optimize import linprog
    # @staticmethod
    # def __solve_linear_ineq(A: np.ndarray, b: np.array) -> Tuple[bool, List[int | float]]:
    #     """
    #     Checks if there exists a solution x for: Ax <= b
    #     :param A: linear part of the equations
    #     :param b: vector of free terms in the equations
    #     :return: if exists (True, solution) else (False, [])
    #     """
    #     _, d = A.shape
    #     bounds = [(None, None)] * d
    #     res = linprog(c=[0] * d, bounds=bounds, A_ub=A, b_ub=b, method="highs")
    #
    #     if res.success or res.status == 3:
    #         # status == 3 means "unbounded" in HiGHS, which is fine
    #         x = res.x.tolist() if res.x is not None else []
    #         return True, x
    #     return False, []

    @staticmethod
    @lru_cache
    def __find_integer_point_milp(
            A, b, xmin: Optional[List[int]] = None, xmax: Optional[List[int]] = None
    ) -> Optional[np.ndarray]:
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
            pulp.LpVariable(
                f'x{i}',
                lowBound=int(xmin[i]) if xmin is not None else None,
                upBound=int(xmax[i]) if xmax is not None else None,
                cat='Integer'
            )
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

    @staticmethod
    def generate_matrices(
            hyperplanes: Union[List[sp.Expr], List[Hyperplane]],
            above_below_indicator: Union[List[int], Tuple[int, ...]]
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
            if isinstance(expr, Hyperplane):
                hp = expr
            else:
                hp = Hyperplane(expr, symbols)
            if ind == 1:
                v, free = hp.as_above_vector
            else:
                v, free = hp.as_below_vector
            free_terms.append(free)
            vectors.append(v)
        return np.vstack(tuple(vectors)), np.array(free_terms), symbols

    @property
    def b_shifted(self):
        """
        Computes b with respect to shifted hyperplanes: Ax <= b' instead of Ax <= b
        :return: The shifted b vector
        """
        S = np.eye(self.shift.shape[0]) * self.shift
        return b + (self.A @ S).sum(axis=1)

    @cached_property
    def start_point(self):
        return self.__find_integer_point_milp(self.A, self.b_shifted)

    @cached_property
    def is_valid(self):
        if self.start_point is None:
            return False
        _, d = self.A.shape
        return self.__find_integer_point_milp(self.A, np.zeros(d)) is not None


if __name__ == '__main__':
    a = np.array([[1, 2], [3, 4]])
    b = np.array([1, 1])
    x, y = sp.symbols('x y')
    shard = Shard(a, b, Position({x: 0.5, y: 0.5}), [x, y])
    print(shard.b_shifted)
