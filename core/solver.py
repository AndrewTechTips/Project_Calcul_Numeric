import numpy as np
import sympy as sp


class ODESolver:
    """Math engine for solving first-order ODEs using Runge-Kutta methods."""

    def __init__(self, func_str: str, x0: float, y0: float, x_end: float, steps: int):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.x_end = float(x_end)
        self.steps = int(steps)

        # calculate step size (h)
        self.h = (self.x_end - self.x0) / self.steps

        self.x_sym, self.y_sym = sp.symbols("x y")
        try:
            # safely parse the math string into a callable numpy function
            self.expr = sp.sympify(func_str)
            self.f = sp.lambdify((self.x_sym, self.y_sym), self.expr, modules=["numpy"])
        except Exception as e:
            raise ValueError(f"Failed to parse equation: {e}")

    def get_x_grid(self) -> np.ndarray:
        """Returns the evenly spaced x-axis values."""
        return np.linspace(self.x0, self.x_end, self.steps + 1)

    def solve_euler(self) -> np.ndarray:
        """Standard Euler method (1st order)."""
        y_vals = np.zeros(self.steps + 1, dtype=np.float64)
        y_vals[0] = self.y0
        x_vals = self.get_x_grid()

        for i in range(self.steps):
            y_vals[i + 1] = y_vals[i] + self.h * self.f(x_vals[i], y_vals[i])
        return y_vals

    def solve_heun(self) -> np.ndarray:
        """Heun's method / Improved Euler (2nd order)."""
        y_vals = np.zeros(self.steps + 1, dtype=np.float64)
        y_vals[0] = self.y0
        x_vals = self.get_x_grid()

        for i in range(self.steps):
            # predict
            k1 = self.h * self.f(x_vals[i], y_vals[i])
            # correct
            k2 = self.h * self.f(x_vals[i] + self.h, y_vals[i] + k1)
            y_vals[i + 1] = y_vals[i] + 0.5 * (k1 + k2)
        return y_vals

    def solve_rk4(self) -> np.ndarray:
        """Classic Runge-Kutta (4th order)."""
        y_vals = np.zeros(self.steps + 1, dtype=np.float64)
        y_vals[0] = self.y0
        x_vals = self.get_x_grid()

        for i in range(self.steps):
            xi, yi = x_vals[i], y_vals[i]

            # calculate the 4 slopes
            k1 = self.h * self.f(xi, yi)
            k2 = self.h * self.f(xi + self.h / 2, yi + k1 / 2)
            k3 = self.h * self.f(xi + self.h / 2, yi + k2 / 2)
            k4 = self.h * self.f(xi + self.h, yi + k3)

            # weighted average of slopes
            y_vals[i + 1] = yi + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return y_vals
