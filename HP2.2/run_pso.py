import numpy as np

rng = np.random.default_rng()


class PSO:
    def __init__(self, N, n, x_min, x_max, v_max, weight, w_min=0.35, beta=0.995):
        self.N = N  # number of particles
        self.n = n  # dimensionality
        self.x_min = np.asarray(x_min)
        self.x_max = np.asarray(x_max)
        self.v_max = v_max
        self.w = weight  # current inertia / weight
        self.w_min = w_min  # lower bound
        self.beta = beta  # decay factor (0<beta<1)
        r = rng.uniform(0, 1, size=(N, n))
        self.x = self.x_min + r * (self.x_max - self.x_min)

        self.v = -((self.x_max - self.x_min) / 2) + r * (self.x_max - self.x_min)

        # personal bests
        self.p_best_pos = self.x.copy()
        self.p_best_val = np.array(self.evaluate_swarm(self.x))
        # global best
        best_idx = int(np.argmin(self.p_best_val))
        self.g_best_pos = self.p_best_pos[best_idx].copy()
        self.g_best_val = float(self.p_best_val[best_idx])

    def evaluate_particle(self, particle):
        x1, x2 = particle[0], particle[1]
        term1 = (x1**2 + x2 - 11) ** 2
        term2 = (x1 + x2**2 - 7) ** 2

        f = term1 + term2

        return f

    def evaluate_swarm(self, swarm):
        return np.array([self.evaluate_particle(p) for p in swarm])

    def update_best_positions(self):
        # update personal bests
        for i in range(self.N):
            f = self.evaluate_particle(self.x[i])
            if f < self.p_best_val[i]:
                self.p_best_val[i] = f
                self.p_best_pos[i] = self.x[i].copy()
        # update global best
        idx = np.argmin(self.p_best_val)
        if self.p_best_val[idx] < self.g_best_val:
            self.g_best_val = self.p_best_val[idx]
            self.g_best_pos = self.p_best_pos[idx].copy()

    def update_velocity_and_position(self, c_1, c_2):
        for i in range(self.N):
            q = rng.uniform(0, 1, size=self.n)
            r = rng.uniform(0, 1, size=self.n)
            cognitive = c_1 * q * (self.p_best_pos[i] - self.x[i])
            social = c_2 * r * (self.g_best_pos - self.x[i])
            self.v[i] = self.w * self.v[i] + cognitive + social
            if self.v_max is not None:
                vmax = np.asarray(self.v_max)
                self.v[i] = np.clip(self.v[i], -vmax, vmax)
            self.x[i] += self.v[i]
        # decay weight after each iteration (use old w for this move)
        if self.w > self.w_min:
            self.w = max(self.w * self.beta, self.w_min)


if __name__ == "__main__":
    N = 50
    n = 2
    x_min = -5
    x_max = 5
    v_max = 4
    c1, c2 = 2, 2
    weight = 1.4  # initial w (>1)
    w_min = 0.35  # lower bound
    beta = 0.995  # decay factor close to 1
    iters = 200

    for _ in range(300):
        swarm = PSO(N, n, x_min, x_max, v_max, weight, w_min=w_min, beta=beta)
        for t in range(iters):
            swarm.update_best_positions()
            swarm.update_velocity_and_position(c1, c2)
        x1, x2 = swarm.g_best_pos
        f = swarm.g_best_val
        print(f"({x1:.6f}, {x2:.6f}, {f:.6f})")
