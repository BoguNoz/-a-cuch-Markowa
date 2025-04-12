import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt

# Proces Browna w 1D
def sample_brownian_motion_1d(num_steps, delta_t=1.0, sigma=1.0):
    x = torch.zeros(num_steps)
    x[0] = pyro.sample("X_0", dist.Normal(0., 1.))  # początkowy punkt

    for t in range(1, num_steps):
        increment = pyro.sample(f"delta_X_{t}", dist.Normal(0., sigma * delta_t**0.5))
        x[t] = x[t-1] + increment
    return x

# Proces Browna w 2D
def sample_brownian_motion_2d(num_steps, delta_t=1.0, sigma=1.0):
    x = torch.zeros(num_steps, 2)
    x[0] = pyro.sample("X_0", dist.Normal(torch.zeros(2), torch.ones(2)))

    for t in range(1, num_steps):
        increment = pyro.sample(f"delta_X_{t}", dist.Normal(torch.zeros(2), sigma * (delta_t**0.5) * torch.ones(2)))
        x[t] = x[t-1] + increment
    return x


def sample_brownian_result_showcase_1d(steps, delta_t=1.0, sigma=1.0):
    pyro.clear_param_store()
    chain = sample_brownian_motion_1d(steps, delta_t, sigma).detach().numpy()

    plt.plot(range(steps), chain, marker="o", linestyle="-", label="Brownian Motion 1D")
    plt.title("Proces Browna (1D)")
    plt.xlabel("t (czas)")
    plt.ylabel("x[t]")
    plt.grid(True)
    plt.legend()
    plt.show()


def sample_brownian_result_showcase_2d(steps, delta_t=1.0, sigma=1.0):
    pyro.clear_param_store()
    chain = sample_brownian_motion_2d(steps, delta_t, sigma).detach().numpy()

    plt.plot(chain[:, 0], chain[:, 1], marker="o", linestyle="-", label="Brownian Motion 2D")
    plt.title("Proces Browna (2D)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()
