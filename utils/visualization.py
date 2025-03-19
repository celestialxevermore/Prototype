import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pdb
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

    # few-shot이 8 이상일 때는 few-shot 결과만 시각화
    if args.few_shot >= 8:
        fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Few-shot의 Train vs Valid
        ax3.plot(results["Full_results"]["Ours_few"]["Ours_train_few_losses"], label='Train Loss')
        ax3.plot(results["Full_results"]["Ours_few"]["Ours_val_few_losses"], label='Valid Loss')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Loss')
        ax3.set_title('Few-shot: Train vs Valid Loss')
        ax3.legend()
        ax3.grid(True)

        ax4.plot(results["Full_results"]["Ours_few"]["Ours_train_few_auc"], label='Train AUC')
        ax4.plot(results["Full_results"]["Ours_few"]["Ours_val_few_auc"], label='Valid AUC')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('AUC')
        ax4.set_title('Few-shot: Train vs Valid AUC')
        ax4.legend()
        ax4.grid(True)

    # few-shot이 4일 때는 full과 few-shot 모두 시각화
    else:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Full dataset의 Train vs Valid
        ax1.plot(results["Full_results"]["Ours"]["Ours_train_full_losses"], label='Train Loss')
        ax1.plot(results["Full_results"]["Ours"]["Ours_val_full_losses"], label='Valid Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Full Dataset: Train vs Valid Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(results["Full_results"]["Ours"]["Ours_train_full_auc"], label='Train AUC')
        ax2.plot(results["Full_results"]["Ours"]["Ours_val_full_auc"], label='Valid AUC')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('AUC')
        ax2.set_title('Full Dataset: Train vs Valid AUC')
        ax2.legend()
        ax2.grid(True)
        
        # Few-shot의 Train vs Valid
        ax3.plot(results["Full_results"]["Ours_few"]["Ours_train_few_losses"], label='Train Loss')
        ax3.plot(results["Full_results"]["Ours_few"]["Ours_val_few_losses"], label='Valid Loss')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Loss')
        ax3.set_title('Few-shot: Train vs Valid Loss')
        ax3.legend()
        ax3.grid(True)

        ax4.plot(results["Full_results"]["Ours_few"]["Ours_train_few_auc"], label='Train AUC')
        ax4.plot(results["Full_results"]["Ours_few"]["Ours_val_few_auc"], label='Valid AUC')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('AUC')
        ax4.set_title('Few-shot: Train vs Valid AUC')
        ax4.legend()
        ax4.grid(True)

    plt.suptitle(f'Training Progress - {args.source_dataset_name} (K={args.few_shot})', y=1.02, fontsize=16)
    plt.tight_layout()
    metrics_plot_path = os.path.join(exp_dir, f"f{args.few_shot}_b{args.batch_size}_l{args.num_layers}_h{args.n_heads}_{timestamp}.png")
    plt.savefig(metrics_plot_path)
    plt.close()

    print(f"Metrics plot saved as {metrics_plot_path}")

# def visualize_results(args, results, exp_dir):
#     os.makedirs(exp_dir, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
#     #pdb.set_trace()
#     # Full dataset의 Train vs Valid
#     ax1.plot(results["Full_results"]["Ours"]["Ours_train_full_losses"], label='Train Loss')
#     ax1.plot(results["Full_results"]["Ours"]["Ours_val_full_losses"], label='Valid Loss')
#     ax1.set_xlabel('Epochs')
#     ax1.set_ylabel('Loss')
#     ax1.set_title('Full Dataset: Train vs Valid Loss')
#     ax1.legend()
#     ax1.grid(True)

#     ax2.plot(results["Full_results"]["Ours"]["Ours_train_full_auc"], label='Train AUC')
#     ax2.plot(results["Full_results"]["Ours"]["Ours_val_full_auc"], label='Valid AUC')
#     ax2.set_xlabel('Epochs')
#     ax2.set_ylabel('AUC')
#     ax2.set_title('Full Dataset: Train vs Valid AUC')
#     ax2.legend()
#     ax2.grid(True)

#     # Few-shot의 Train vs Valid
#     ax3.plot(results["Full_results"]["Ours_few"]["Ours_train_few_losses"], label='Train Loss')
#     ax3.plot(results["Full_results"]["Ours_few"]["Ours_val_few_losses"], label='Valid Loss')
#     ax3.set_xlabel('Epochs')
#     ax3.set_ylabel('Loss')
#     ax3.set_title('Few-shot: Train vs Valid Loss')
#     ax3.legend()
#     ax3.grid(True)

#     ax4.plot(results["Full_results"]["Ours_few"]["Ours_train_few_auc"], label='Train AUC')
#     ax4.plot(results["Full_results"]["Ours_few"]["Ours_val_few_auc"], label='Valid AUC')
#     ax4.set_xlabel('Epochs')
#     ax4.set_ylabel('AUC')
#     ax4.set_title('Few-shot: Train vs Valid AUC')
#     ax4.legend()
#     ax4.grid(True)

#     plt.suptitle(f'Training Progress - {args.source_dataset_name}', y=1.02, fontsize=16)
#     plt.tight_layout()
#     metrics_plot_path = os.path.join(exp_dir, f"metrics_plot_{timestamp}.png")
#     plt.savefig(metrics_plot_path)
#     plt.close()

#     print(f"Metrics plot saved as {metrics_plot_path}")