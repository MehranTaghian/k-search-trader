from scipy.special import lambertw
import numpy as np
from scipy.optimize import fsolve


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
            guess = np.random.randn() * 100
            self.r_star = fsolve(self.r_star_func, guess)[0]
            self.s_star = fsolve(self.s_star_func, guess)[0]

        print(f"r*: {self.r_star}, s*: {self.s_star}")
        r_star_error = self.r_star_func([self.r_star])[0]
        s_star_error = self.s_star_func([self.s_star])[0]
        assert abs(r_star_error) < 1e-5, f"Couldn't find a good solution for r*! Current error is {r_star_error}"
        assert abs(s_star_error) < 1e-5, f"Couldn't find a good solution for s*! Current error is {s_star_error}"

    def get_pi_max(self, i):
        return self.min_price * (1 + (self.r_star - 1) * np.power(1 + self.r_star / self.k, i - 1))

    def get_pi_min(self, i):
        return self.max_price * (1 - (1 - 1 / self.s_star) * np.power(1 + 1 / (self.k * self.s_star), i - 1))

    def r_star_func(self, z):
        x = z[0]
        f = np.empty(1)
        f[0] = (self.phi - 1) / (x - 1) - np.power(1 + x / self.k, self.k)
        return f

    def s_star_func(self, z):
        x = z[0]
        f = np.empty(1)
        f[0] = (1 - 1 / self.phi) / (1 - 1 / x) - np.power(1 + 1 / (x * self.k), self.k)
        return f
