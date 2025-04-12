import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt

# Próbkowanie łańcucha Markowa dla 2D
def sample_markov_chain_2d(num_steps):
    # Tworzenie tensora 2D o wymiarach (num_steps, 2) - dla dwóch wymiarów
    x = torch.zeros(num_steps, 2)
    x[0] = pyro.sample("x_0", dist.Normal(torch.zeros(2), torch.ones(2)))

    for t in range(1, num_steps):
        # Losowanie nowego stanu z rozkładu Normalnego dla obu wymiarów
        x[t] = pyro.sample(f"x_{t}", dist.Normal(x[t-1], torch.ones(2)))
    return x

def sample_markov_result_showcase_2d(steps):
    num_steps = steps  # długość łańcucha
    pyro.clear_param_store()  # czyszczenie parametrów
    chain = sample_markov_chain_2d(num_steps).detach().numpy()

    # Wykres
    plt.plot(chain[:, 0], chain[:, 1], marker="o", linestyle="-", label="Ścieżka 2D")
    plt.title("Łańcuch Markowa (Random Walk) - 2D")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()

