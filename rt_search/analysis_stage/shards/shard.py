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

import numpy as np
import time
from numba import njit, float64, int64, boolean
from scipy.special import gamma, zeta


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

    def sample_trajectories(self, n_samples, *, strict: Optional[bool] = False) -> List[Position]:
        """
        Sample trajectories from the shard.
        :param n_samples: number of samples to generate
        :param strict: if compute as n_samples, else compute n_samples * fraction.
        (fraction of the cone is taking from the sphere)
        :return: a list of sampled trajectories
        """
        A, encoding_mask = self.mask_symbols(self.A, self.group, self.symbols)
        R, fraction = self.compute_required_radius(A, n_samples)
        if strict:
            # Compute the radius with some safety mesures
            n_samples = np.ceil(int(n_samples / fraction * 1.02))
            R = self.compute_ball_radius(len(self.symbols), n_samples)
        else:
            n_samples = int(np.ceil(n_samples * fraction))
        sampler = CHRRSampler(A, np.zeros_like(self.b), R=R, thinning=5)
        return [
            Position({sym: v for v, sym in zip(self.reconstruct(p, encoding_mask), self.symbols)})
            for p in sampler.sample(n_samples)
        ]

    @staticmethod
    def mask_symbols(A, group, symbols):
        encoding_mask = np.array([symbol in group for symbol in symbols])
        if np.all(encoding_mask):
            print('bad bad')
        return A[:, encoding_mask], encoding_mask

    @staticmethod
    def reconstruct(x_reduced, mask):
        x = np.zeros(len(mask), dtype=x_reduced.dtype)
        x[mask] = x_reduced
        return x

    @staticmethod
    def compute_required_radius(A, n_target, sample_count=10_000, safety_factor=1.05):
        """
        Estimates the Radius R needed to get n_target integer points
        with GCD=1 inside the cone Ax <= 0.
        """
        m, d = A.shape

        # 1. Estimate Fraction of Sphere Covered by Cone (Monte Carlo)
        # We ignore 'b' here and use b=0 because for large R,
        # the volume is dominated by the angular opening (Ax <= 0).

        # Generate random directions on unit sphere
        raw = np.random.normal(size=(sample_count, d))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        dirs = raw / norms

        # Check how many satisfy Ax <= 0
        # A: (m, d), dirs: (N, d) -> (N, m)
        projections = dirs @ A.T

        # Check if all constraints <= 0 for each direction
        inside_mask = np.all(projections <= 1e-9, axis=1)
        fraction = np.sum(inside_mask) / sample_count

        if fraction == 0:
            raise ValueError("Cone is too thin! Monte Carlo found 0 hits. "
                             "Cannot estimate R. You might need to pick R manually or increase sample_count.")

        print(f"Estimated Cone Fraction: {fraction:.4%}")

        # 2. Volume of Unit Ball in d-dimensions
        # V_unit = pi^(d/2) / gamma(d/2 + 1)
        vol_unit_ball = (np.pi ** (d / 2.0)) / gamma(d / 2.0 + 1.0)

        # 3. Density of Primitive Vectors (GCD=1)
        # Density approx 1/zeta(d)
        # For d=1, density is technically 0 in limit, but practical for geometry.
        prim_density = 1.0 / zeta(d) if d > 1 else 1.0

        # 4. Solve for R
        # N = Vol_Unit * R^d * Fraction * Density
        # R^d = N / (Vol_Unit * Fraction * Density)

        term = n_target / (vol_unit_ball * fraction * prim_density)
        R_est = term ** (1.0 / d)

        # Apply safety factor
        R_final = R_est * safety_factor

        return R_final, fraction * 100

    @staticmethod
    def compute_ball_radius(d, n_samples):
        """
        Computes the Radius R required for a d-dimensional ball to contain
        approximately n_samples integer points.

        Parameters:
        -----------
        d : int
            Number of dimensions.
        n_samples : int
            Target number of points.
        primitive : bool
            If True, adjusts for GCD=1 requirement (points must be primitive).
            If False, counts all integer points.

        Returns:
        --------
        float : The estimated Radius R.
        """
        # 1. Volume of a Unit Ball in d-dimensions
        # V_unit = pi^(d/2) / Gamma(d/2 + 1)
        vol_unit_ball = (np.pi ** (d / 2.0)) / gamma(d / 2.0 + 1.0)

        # 2. Density Adjustment
        # If we want GCD=1, the density of points is 1/zeta(d).
        # Note: zeta(1) is infinity, but physically 1D density is usually treated as 1 for counts
        density = 1.0 / zeta(d) if d > 1 else 1.0

        # 3. Solve for R
        # N = Vol_Unit * R^d * Density
        # R = ( N / (Vol_Unit * Density) ) ^ (1/d)

        term = n_samples / (vol_unit_ball * density)
        R = term ** (1.0 / d)

        return R

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


# --- Shared Numba Logic ---

@njit(cache=True)
def gcd_recursive(a, b):
    while b:
        a, b = b, a % b
    return a


@njit(cache=True)
def get_gcd_of_array(arr):
    """Calculates GCD of a vector. Returns 1 immediately if any pair gives 1."""
    d = len(arr)
    if d == 0: return 0
    result = abs(arr[0])
    for i in range(1, d):
        result = gcd_recursive(result, abs(arr[i]))
        if result == 1:
            return 1
    return result


@njit(cache=True)
def is_valid_integer_point(point, A, b, R_sq):
    """
    Checks 3 conditions:
    1. Inside Radius (L2)
    2. Inside Cone (Ax <= b)
    3. GCD == 1
    """
    # 1. Radius Check
    norm_sq = 0.0
    for x in point:
        norm_sq += x * x
    if norm_sq > R_sq or norm_sq == 0:  # Exclude origin
        return False

    # 2. Cone Check (Ax <= b)
    # Manual dot product for speed
    m, d = A.shape
    for i in range(m):
        dot = 0.0
        for j in range(d):
            dot += A[i, j] * point[j]
        if dot > b[i]:
            return False

    # 3. GCD Check (Primitive)
    if get_gcd_of_array(point) != 1:
        return False

    return True


@njit(cache=True)
def get_chrr_limits(idx, x, A_cols, b, current_Ax, R_sq):
    """
    Computes the valid range [t_min, t_max] for moving along axis 'idx'
    such that x + t*e_idx stays inside Ax<=b AND ||x|| <= R
    """
    t_min = -1e20 # arbitrary large
    t_max = 1e20

    # --- 1. Cone Constraints (Ax <= b) ---
    # Condition: A_row * (x + t*e_i) <= b_row
    #            (A_row*x) + t*A_row[i] <= b_row
    #            t * A_col_i[row] <= b_row - current_Ax[row]

    # We iterate over the pre-transposed A (A_cols) for cache efficiency
    col_data = A_cols[idx]
    m = len(b)

    for row in range(m):
        slope = col_data[row]
        # rem = b[row] - current_Ax[row] (but slightly safer to recompute if accumulating error?)
        # For CHRR speed, we usually use the tracked Ax.
        rem = b[row] - current_Ax[row]

        if slope > 1e-9:
            # t <= rem / slope
            t_max = min(t_max, rem / slope)
        elif slope < -1e-9:
            # t * neg <= rem  ->  t >= rem / neg
            t_min = max(t_min, rem / slope)
        else:
            # slope is 0. If constraint violated, line is invalid entirely.
            if rem < 0:
                return 1.0, -1.0 # Return empty interval

    # --- 2. Ball Constraint (||x + t*e_i||^2 <= R^2) ---
    # sum(x_j^2) - x_i^2 + (x_i + t)^2 <= R^2
    # (x_i + t)^2 <= R^2 - (current_norm_sq - x_i^2)

    current_norm_sq = 0.0
    for val in x:
        current_norm_sq += val*val

    rem_rad_sq = R_sq - (current_norm_sq - x[idx]**2)

    if rem_rad_sq < 0:
        return 1.0, -1.0 # Invalid

    limit_r = np.sqrt(rem_rad_sq)
    # -limit_r <= x_i + t <= limit_r
    # t >= -limit_r - x_i
    # t <= limit_r - x_i

    t_min = max(t_min, -limit_r - x[idx])
    t_max = min(t_max, limit_r - x[idx])

    return t_min, t_max


@njit(cache=True)
def chrr_walker(A, A_cols, b, R_sq, start_point, n_desired, thinning, buf_out):
    """
    A: (m, d)
    A_cols: (d, m) - Transposed A for fast column access
    """
    m, d = A.shape

    # Current State
    x = start_point.copy()
    current_Ax = np.dot(A, x)

    found = 0
    steps = 0

    # Temp buffer for integer check
    temp_int = np.zeros(d, dtype=np.int64)

    while found < n_desired:
        # 1. Coordinate Hit-and-Run Step
        # Pick random axis
        axis_idx = np.random.randint(0, d)

        # Get valid line segment
        t_min, t_max = get_chrr_limits(axis_idx, x, A_cols, b, current_Ax, R_sq)

        if t_max >= t_min:
            # Move to random point in interval
            t = np.random.uniform(t_min, t_max)

            # Update State
            x[axis_idx] += t
            # Update Ax efficiently: Ax_new = Ax_old + t * A_col_i
            # This is O(m)
            for row in range(m):
                current_Ax[row] += t * A_cols[axis_idx, row]

        steps += 1

        # 2. Harvesting (Thinning)
        # We only try to harvest every 'thinning' steps to ensure mixing
        if steps % thinning == 0:
            # Try to extract an integer point
            # Technique: Take continuous x, add Uniform(-0.5, 0.5), Round.
            # Then CHECK constraints.

            valid_harvest = True
            norm_sq = 0.0

            for k in range(d):
                # Add jitter + Round
                val = x[k] + np.random.uniform(-0.5, 0.5)
                ival = int(round(val))
                temp_int[k] = ival
                norm_sq += ival*ival

            # Check Radius
            if norm_sq > R_sq or norm_sq == 0:
                valid_harvest = False

            # Check Cone (Must re-check because rounding might push us out)
            if valid_harvest:
                for row in range(m):
                    dot = 0.0
                    for col in range(d):
                        dot += A[row, col] * temp_int[col]
                    if dot > b[row]:
                        valid_harvest = False
                        break

            # Check GCD
            if valid_harvest:
                if get_gcd_of_array(temp_int) != 1:
                    valid_harvest = False

            if valid_harvest:
                # Store
                buf_out[found, :] = temp_int[:]
                found += 1

    return steps


class CHRRSampler:
    def __init__(self, A, b, R, thinning=3):
        self.A = np.ascontiguousarray(A, dtype=np.float64)
        # We pre-transpose A for faster column access in the walker
        self.A_cols = np.ascontiguousarray(A.T, dtype=np.float64)
        self.b = np.ascontiguousarray(b, dtype=np.float64)
        self.R = float(R)
        self.R_sq = self.R**2
        self.thinning = thinning

    def find_start_point(self):
        """Finds a valid continuous point inside to start the chain."""
        # Simple Rejection Sampling Attempt (usually finds one instantly)
        # If the cone is SO thin this fails, we need LP, but let's assume this works.
        d = self.A.shape[1]
        for _ in range(10000):
            # Sample in small box around origin or uniform in R
            cand = np.random.uniform(-self.R/10, self.R/10, size=d)
            if np.linalg.norm(cand) > self.R: continue
            if np.all(self.A @ cand <= self.b):
                return cand
        raise RuntimeError("Could not find starting point for CHRR. Cone is too thin or closed.")

    def sample(self, n_samples):
        t0 = time.time()
        start_pt = self.find_start_point()

        # Buffer for Numba
        buf = np.zeros((n_samples, self.A.shape[1]), dtype=np.int64)

        # Run Walker
        # Note: CHRR naturally produces duplicates if the chain stays in the same
        # "integer cell" for multiple harvest steps.
        chrr_walker(self.A, self.A_cols, self.b, self.R_sq, start_pt, n_samples, self.thinning, buf)

        # Post-process for global Uniqueness
        # Because CHRR is a chain, we might need to run longer to get *unique* points
        # So we filter here.
        # unique_set = set(tuple(x) for x in buf)
        unique_set = list(tuple(x) for x in buf)

        # Convert back to array
        final_arr = np.array(list(unique_set))
        return final_arr, time.time() - t0


if __name__ == '__main__':
    a = np.array([[1, 2], [3, 4]])
    b = np.array([1, 1])
    x, y = sp.symbols('x y')
    shard = Shard(a, b, Position({x: 0.5, y: 0.5}), [x, y])
    print(shard.b_shifted)
