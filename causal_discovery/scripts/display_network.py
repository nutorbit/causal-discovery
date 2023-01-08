import seaborn as sns
import matplotlib.pyplot as plt

from causal_discovery.algos.notears import NoTears
from causal_discovery.synthetic_dataset import SyntheticDataset
from causal_discovery.utils import display_network


def main():
    n, d = 1000, 20
    graph_type, degree, sem_type = 'erdos-renyi', 4, 'linear-gauss'
    noise_scale = 1.0
    dataset_type = 'linear'
    seed = 123

    dataset = SyntheticDataset(n, d, graph_type, degree, sem_type,
                               noise_scale, dataset_type, seed)
    
    model = NoTears(
        rho=1, 
        alpha=0.1, 
        l1_reg=0, 
        lr=1e-2
    )

    _ = model.learn(dataset.X)
    
    display_network(dataset.W, range(dataset.W.shape[1]), "./test.html", 0.7)
    


if __name__ == "__main__":
    main()
