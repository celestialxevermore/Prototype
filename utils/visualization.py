import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def visualize_gmm_step(gmm, source_embeddings, step, output_dir="visualizations"):
    """
    Visualize GMM's prior, means, covariances, and source-specific embeddings.
    """
    num_components = gmm.k
    colors = ['red', 'blue', 'green', 'purple', 'orange']  # Up to 5 sources
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack([X.ravel(), Y.ravel()])

    # Plot GMM contours
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(num_components):
        mean = gmm.means[i].cpu().detach().numpy()
        covariance = gmm.covariances[i].cpu().detach().numpy()

        # Multivariate Gaussian density
        pdf = multivariate_gaussian_density(xy, mean, covariance).reshape(100, 100)
        ax.contour(X, Y, pdf, levels=10, alpha=0.5, colors='gray')

    # Plot source embeddings
    for idx, embeddings in enumerate(source_embeddings):
        embeddings = embeddings.cpu().detach().numpy()
        ax.scatter(embeddings[:, 0], embeddings[:, 1], label=f"Source {idx + 1}", color=colors[idx % len(colors)], alpha=0.6)

    ax.set_title(f"GMM Visualization at Step {step}")
    ax.legend()
    plt.savefig(f"{output_dir}/gmm_step_{step}.png")
    plt.close()


def multivariate_gaussian_density(xy, mean, covariance):
    """
    Compute the density of a multivariate Gaussian at given points.
    """
    dim = mean.shape[0]
    covariance_inv = np.linalg.inv(covariance)
    covariance_det = np.linalg.det(covariance)
    norm_const = 1 / (np.sqrt((2 * np.pi) ** dim * covariance_det))

    diff = xy - mean
    exp_term = np.einsum('ij,jk,ik->i', diff, covariance_inv, diff)
    return norm_const * np.exp(-0.5 * exp_term)


def visualize_results(args, results, exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Loss 그래프
    ax1.plot(results["Full_results"]["Ours"]["Ours_train_full_losses"], label='Ours Train Loss')
    ax1.plot(results["Full_results"]["Ours"]['Ours_test_full_losses'], label='Ours Test Loss')
    #ax1.plot(results["Full_results"]["Ours_few"]["Ours_train_few_losses"], label='Ours (few) Train Loss')
    #ax1.plot(results["Full_results"]["Ours_few"]["Ours_test_few_losses"], label='Ours (few) Test Loss')
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss Curves - {args.source_dataset_name}')
    ax1.legend()
    ax1.grid(True)

    # Ours AUC 그래프
    ax2.plot(results["Full_results"]["Ours"]["Ours_train_full_auc"], label='Ours train AUC')
    ax2.plot(results["Full_results"]["Ours"]["Ours_test_full_auc"], label='Ours test AUC')
    #ax2.plot(results["Full_results"]["Ours_few"]["Ours_train_few_auc"], label='Ours (few) train AUC')
    #ax2.plot(results["Full_results"]["Ours_few"]["Ours_test_few_auc"], label='Ours (few) test AUC')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('AUC')
    ax2.set_title(f'Ours AUC Curves - {args.source_dataset_name}')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    metrics_plot_path = os.path.join(exp_dir, f"metrics_plot_{timestamp}.png")
    plt.savefig(metrics_plot_path)
    plt.close()

    print(f"Metrics plot saved as {metrics_plot_path}")
