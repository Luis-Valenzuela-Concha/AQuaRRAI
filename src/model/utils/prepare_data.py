import numpy as np

def prepare_data_for_model(sets, include_filtered=True, grouped=True, score="ssim"):
    x_data = []
    y_score = []
    for set_data in sets:
        group = []
        group_scores = []
        for recon in set_data.reconstructions:
            if include_filtered and recon.filtered_data is not None:
                img = recon.filtered_data
            else:
                img = recon.data
            group.append(img)
            if score == "psnr":
                group_scores.append(recon.metrics.psnr.value)
            elif score == "ssim":
                group_scores.append(recon.metrics.ssim.value)

        if grouped:
            x_data.append(np.array(group))
            y_score.append(np.array(group_scores))
        else:
            x_data.extend(group)
            y_score.extend(group_scores)
    x_data = np.array(x_data)
    y_score = np.array(y_score)
    
    return x_data, y_score