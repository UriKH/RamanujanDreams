from dataclasses import dataclass
from functools import cached_property

import sympy as sp
from typing import Tuple, Optional, List
import numpy as np


@dataclass
class Hyperplane:
    """
    Represents a hyperplane as a sympy expression.
    The expression might miss some symbols as the space the hyperplane lives in consists more axis than defined.
    """
    expr: sp.Expr
    symbols: Optional[List[sp.Basic]] = None

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = list(self.expr.free_symbols)
        if not self.expr.free_symbols.issubset(self.symbols):
            raise ValueError(
                f'Missing symbols in ordering. Expression contains {self.expr.free_symbols} but given {self.symbols}'
            )
        # self.linear_term, self.free_term = self.uniform()
        try:
            polys = {var: sp.Poly(self.expr, var) for var in self.expr.free_symbols}
            if any(poly.degree() > 1 for poly in polys.values()):
                raise sp.PolynomialError
        except sp.PolynomialError:
            raise ValueError(f'Expression is not linear')

        poly = sp.Poly(self.expr)
        self.sym_coef_map = {
            sym: poly.coeff_monomial(sym) if sym in self.expr.free_symbols else 0 for sym in self.symbols
        }
        self.linear_term: sp.Expr = sum([self.sym_coef_map[sym] * sym for sym in self.symbols if sym in self.expr.free_symbols])
        self.free_term = self.expr.subs({sym: 0 for sym in self.expr.free_symbols})

    # def uniform(self) -> Tuple[sp.Expr, sp.Expr]:
    #     """
    #     From a given linear expression convert to uniform format
    #     :return: The inequality [e.g.: by+cz+ax+K ---> reordering ---> ax+by+cz, -K]
    #     """
    #     symbols = [sym for sym in self.symbols if sym in self.expr.free_symbols]
    #
    #     try:
    #         polys = {var: sp.Poly(self.expr, var) for var in symbols}
    #         if any(poly.degree() > 1 for poly in polys.values()):
    #             raise sp.PolynomialError
    #     except sp.PolynomialError:
    #         raise ValueError(f'Expression is not linear')
    #
    #     k = self.expr.subs({sym: 0 for sym in symbols})
    #     uniform = sum([poly.coeff(sym) for sym, poly in polys.items() if sym in self.expr.free_symbols])
    #     # uniform = polys[symbols[0]]
    #     # for sym in symbols[1:]:
    #     #     uniform += polys[sym]
    #     return uniform, -k

    @cached_property
    def equation_like(self) -> Tuple[sp.Expr, sp.Expr]:
        """
        :return: lhs = rhs where lhs is the linear term and the rhs is the free trem
        """
        return self.linear_term, self.free_term

    @cached_property
    def vectors(self):
        linear = [self.sym_coef_map[sym] for sym in self.symbols]
        return np.array(linear), self.free_term

    @property
    def as_below_vector(self):
        """
        linear - free <= 0 ---> linear <= free ---> -linear >= -free
        """
        linear, free = self.vectors
        return -linear, -free

    @property
    def as_above_vector(self):
        """
        linear - free >= 0 ---> linear >= -free
        """
        linear, free = self.vectors
        return linear, -free


if __name__ == '__main__':
    x, y, z, a = sp.symbols('x y z a')
    expr = 2*x+4*z-2*y+5
    hp = Hyperplane(expr, [a, x, y, z])

    print(hp.equation_like)
    print(hp.vectors)
