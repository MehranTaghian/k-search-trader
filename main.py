

class RPP:
    def __init__(self, min_price, max_price, ):
        """
        Here we assumed that k -> infinity so that we can buy or sell for infinitely many times.
        :param min_price:
        :param max_price:
        """

        self.phi = max_price / min_price

