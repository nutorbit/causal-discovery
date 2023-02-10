import seaborn as sns
import matplotlib.pyplot as plt

from causal_discovery.algos.notears import NoTears
from causal_discovery.synthetic_dataset import SyntheticDataset


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

    loss_hist = model.learn(dataset.X)
    
    plt.plot(loss_hist)
    plt.show()
    
    fig, ax = plt.subplots(ncols=2)
    sns.heatmap(dataset.W, ax=ax[0], cmap="PiYG", vmin=-2, vmax=2)
    sns.heatmap(model.get_result(), ax=ax[1], cmap="PiYG", vmin=-2, vmax=2)
    plt.show()


if __name__ == "__main__":
    main()
