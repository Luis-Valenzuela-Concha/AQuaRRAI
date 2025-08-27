import matplotlib.pyplot as plt

def plot_set(set, include_filtered=False, include_metrics=False):
    columns = len(set.reconstructions) + 1
    rows = 1 if not include_filtered else 2

    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 7 * rows))

    if rows == 1:
        axes = axes.reshape(1, -1)
        

    font_size = 18

    axes[0, 0].imshow(set.ground_truth.data, cmap='inferno')
    axes[0, 0].set_title("Ground Truth", fontsize=font_size)
    axes[0, 0].axis('off')

    for i, recon in enumerate(set.reconstructions):
        axes[0, i+1].imshow(recon.data, cmap='inferno')

        metrics_text = ""
        if include_metrics:
            metrics_text = f"SSIM: {recon.metrics.ssim.value:.4f}\n"
            metrics_text += f"PSNR: {recon.metrics.psnr.value:.4f}\n"
            metrics_text += f"Residual RMS: {recon.metrics.residual_rms.value:.4f}"
            metrics_text += "\n\n"

        if recon.algorithm:
            algorithm = recon.algorithm.split('_')[:-1][1]
        axes[0, i+1].set_title(f"{metrics_text}{algorithm} ({recon.sim})", fontsize=font_size)
        axes[0, i+1].axis('off')

    if include_filtered:
        axes[1, 0].imshow(
            set.ground_truth.filtered_data if set.ground_truth.filtered_data is not None else set.ground_truth.data,
            cmap='inferno'
        )
        axes[1, 0].set_title("Ground Truth (Filtered)", fontsize=font_size)
        axes[1, 0].axis('off')

        for i, recon in enumerate(set.reconstructions):
            axes[1, i+1].imshow(
                recon.filtered_data if recon.filtered_data is not None else recon.data,
                cmap='inferno'
            )
            if recon.algorithm:
                algorithm = recon.algorithm.split('_')[:-1][1]
            axes[1, i+1].set_title(f"{algorithm} ({recon.sim})", fontsize=font_size)
            axes[1, i+1].axis('off')

    plt.suptitle(f"Object: {set.object_name}\n", fontsize=font_size + 6)
    plt.tight_layout()
    plt.show()