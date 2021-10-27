from scipy.special import lambertw
import numpy as np
from scipy.optimize import fsolve, least_squares


class RPP:
    def __init__(self, k, min_price, max_price):
        """
        Here we assumed that k -> infinity so that we can buy or sell for infinitely many times.
        :param min_price:
        :param max_price:
        """
        self.k = k
        self.min_price, self.max_price = min_price, max_price
        self.phi = max_price / min_price

        if k == np.infty:
            self.r_star = (1 + lambertw((self.phi - 1) / np.e)).real
            self.s_star = (1 / (lambertw(- (self.phi - 1) / (np.e * self.phi)) + 1)).real
        else:
            guess = np.random.rand() * 100
            r_star_sol = least_squares(self.r_star_func, guess, bounds=([0, np.infty]))
            s_star_sol = least_squares(self.s_star_func, guess, bounds=([0, np.infty]))
            self.r_star = r_star_sol.x[0]
            self.s_star = s_star_sol.x[0]
            # while r_star_sol.cost > 1e-5:
            #     r_star_sol = least_squares(self.r_star_func, guess, bounds=([0, np.infty]))
            #     self.r_star = r_star_sol.x[0]
            # while s_star_sol.cost > 1e-5:
            #     s_star_sol = least_squares(self.s_star_func, guess, bounds=([0, np.infty]))
            #     self.s_star = s_star_sol.x[0]

        # print(f"r*: {self.r_star}, s*: {self.s_star}")
        r_star_error = self.r_star_func([self.r_star])
        s_star_error = self.s_star_func([self.s_star])
        assert abs(r_star_error) < 1e-5, f"Couldn't find a good solution for r*! Current error is {r_star_error}"
        assert abs(s_star_error) < 1e-5, f"Couldn't find a good solution for s*! Current error is {s_star_error}"

    def get_pi_max(self, i):
        return self.min_price * (1 + (self.r_star - 1) * np.power(1 + self.r_star / self.k, i - 1))

    def get_pi_min(self, i):
        return self.max_price * (1 - (1 - 1 / self.s_star) * np.power(1 + 1 / (self.k * self.s_star), i - 1))

    def r_star_func(self, z):
        return (self.phi - 1) / (z[0] - 1) - np.power(1 + z[0] / self.k, self.k)

    def s_star_func(self, z):
        return (1 - 1 / self.phi) / (1 - 1 / z[0]) - np.power(1 + 1 / (z[0] * self.k), self.k)
