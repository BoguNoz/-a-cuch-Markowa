import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt

# Próbkowanie łańcucha Markowa
def sample_markov_chain(num_steps):
    x = torch.zeros(num_steps)  # pusty tensor na próbki
    x[0] = pyro.sample("x_0", dist.Normal(0., 1.))
    for t in range(1, num_steps):
        x[t] = pyro.sample(f"x_{t}", dist.Normal(x[t-1], 1.))
    return x


def sample_markov_result_showcase_1d(steps):
    num_steps = steps  # długość łańcucha
    # pyro.set_rng_seed(0) # tworzenie stałego ziarna
    pyro.clear_param_store() # czyszczenie parametrów
    chain = sample_markov_chain(num_steps).detach().numpy()

    # Wykres
    plt.plot(range(num_steps), chain, marker="o", linestyle="-", label="x[t]")
    plt.title("Łańcuch Markowa (Random Walk)")
    plt.xlabel("t (czas)")
    plt.ylabel("x[t]")
    plt.grid(True)
    plt.legend()
    plt.show()