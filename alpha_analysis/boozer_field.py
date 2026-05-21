class BoozerField():
    def __init__(self, B0, R0, a, kappa, delta, iota):
        self.B0 = B0
        self.R0 = R0
        self.a = a
        self.kappa = kappa
        self.delta = delta
        self.iota = iota

    def B(self, r, theta):
        """Calculate the magnetic field strength at a given point in Boozer coordinates."""
        # This is a simplified representation of the magnetic field in Boozer coordinates.
        # The actual implementation would depend on the specific form of the Boozer field.
        return self.B0 * (1 + (r / self.a) * (self.kappa * np.cos(theta) + self.delta * np.sin(theta)))

    def grad_B(self, r, theta):
        """Calculate the gradient of the magnetic field strength at a given point in Boozer coordinates."""
        # This is a placeholder for the actual gradient calculation.
        dB_dr = (self.B(r + 1e-5, theta) - self.B(r - 1e-5, theta)) / (2 * 1e-5)
        dB_dtheta = (self.B(r, theta + 1e-5) - self.B(r, theta - 1e-5)) / (2 * 1e-5)
        return np.array([dB_dr, dB_dtheta])