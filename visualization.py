import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import random
from mpl_toolkits.axisartist.grid_finder import DictFormatter
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib import colors
import matplotlib
from matplotlib.colors import LinearSegmentedColormap


def draw_tracks(track_names, track_perf, predictions_full, eval_gt_full,
                test_seq, bin_size, num_regions, eval_infos, hic_keys,
                hic_track_size, predictions_hic, hic_output,
                num_hic_bins, fig_path):

    mats = []
    for h in range(len(hic_keys)):
        it = h * hic_track_size
        it2 = h * hic_track_size
        for i in range(len(predictions_hic)):
            mat_gt = recover_shape(hic_output[i][it:it + hic_track_size], num_hic_bins)
            mat_pred = recover_shape(predictions_hic[i][it2:it2 + hic_track_size], num_hic_bins)
            mats.append([mat_gt, mat_pred])

    print("Drawing tracks")
    pic_count = 0
    for it, track in enumerate(track_names):
        type = track[:track.find(".")]
        if type != "CAGE":
            continue
        if track_perf[track] < 0.7:
            continue
        for i in range(len(predictions_full)):
            fig = matplotlib.pyplot.figure(figsize=(20, 10))
            ax1 = setup_axes1(fig, 111, -45)

            matrix = np.tril(mats[i][0])

            # using the upper triangle matrix as mask
            red = ((1.0, 1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 1.0))
            cmap_red = LinearSegmentedColormap.from_list('Custom', red, 256)

            blue = ((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0))
            cmap_blue = LinearSegmentedColormap.from_list('Custom', blue, 256)

            sns.heatmap(mats[i][0], annot=False, mask=matrix, ax=ax1, cbar=False, alpha=0.5, cmap=cmap_blue)
            sns.heatmap(mats[i][1], annot=False, mask=matrix, ax=ax1, cbar=False, alpha=0.5, cmap=cmap_red)

            ####################################################
            max_val = np.max(eval_gt_full[i][it])
            tss_layer = test_seq[i]
            tss_marks = []
            for region in range(5000, 45000 + bin_size, bin_size):
                if np.sum(tss_layer[region:region + bin_size]) > 0:
                    tss_marks.append(max_val)
                else:
                    tss_marks.append(0)
            #####################################################
            vector1 = predictions_full[i][it]
            vector2 = eval_gt_full[i][it]
            x = range(num_regions)
            d1 = {'bin': x, 'expression': vector1}
            df1 = pd.DataFrame(d1)
            d2 = {'bin': x, 'expression': vector2}
            df2 = pd.DataFrame(d2)
            #####################################################
            d3 = {'bin': x, 'tss': tss_marks}
            df3 = pd.DataFrame(d3)
            #####################################################
            ax2 = fig.add_axes([0, 0.6, 1, 0.2])
            sns.lineplot(data=df1, x='bin', y='expression', ax=ax2)
            sns.lineplot(data=df2, x='bin', y='expression', ax=ax2)
            #####################################################
            sns.histplot(data=df3, x='bin', y='tss', fill=False, ax=ax2)
            #####################################################
            ax2.set_title(f"{eval_infos[i][0]}:{eval_infos[i][1]}")
            fig.tight_layout()
            plt.savefig(f"{fig_path}_{eval_infos[i][2]}_{track}.png")
            plt.close(fig)
            pic_count += 1
            if i > 20:
                break
        if pic_count > 100:
            break


def draw_regplots(track_names, track_perf, final_pred, eval_gt, fig_path):
    pic_count = 0
    print("Drawing gene regplot")
    for it, track in enumerate(track_names):
        type = track[:track.find(".")]
        if type != "CAGE":
            continue
        if track_perf[track] < 0.7:
            continue
        a = []
        b = []
        for gene in final_pred.keys():
            a.append(final_pred[gene][track])
            b.append(eval_gt[gene][track])

        fig, ax = plt.subplots(figsize=(6, 6))
        r, p = stats.spearmanr(a, b)

        sns.regplot(x=a, y=b,
                    ci=None, label="r = {0:.2f}; p = {1:.2e}".format(r, p)).legend(loc="best")

        ax.set(xlabel='Predicted', ylabel='Ground truth')
        plt.title("Gene expression prediction")
        fig.tight_layout()
        plt.savefig(f"{fig_path}_{track}_{track_perf[track]}.svg")
        plt.close(fig)
        pic_count += 1
        if pic_count > 200:
            break


def draw_attribution():
    # attribution
    # for c, cell in enumerate(cells):
    #     for i in range(1200, 1210, 1):
    #         baseline = tf.zeros(shape=(input_size, num_features))
    #         image = test_input_sequences[i].astype('float32')
    #         ig_attributions = attribution.integrated_gradients(our_model, baseline=baseline,
    #                                                            image=image,
    #                                                            target_class_idx=[mid_bin, c],
    #                                                            m_steps=40)
    #
    #         attribution_mask = tf.squeeze(ig_attributions).numpy()
    #         attribution_mask = (attribution_mask - np.min(attribution_mask)) / (
    #                     np.max(attribution_mask) - np.min(attribution_mask))
    #         attribution_mask = np.mean(attribution_mask, axis=-1, keepdims=True)
    #         attribution_mask[int(input_size / 2) - 2000 : int(input_size / 2) + 2000, :] = np.nan
    #         attribution_mask = skimage.measure.block_reduce(attribution_mask, (100, 1), np.mean)
    #         attribution_mask = np.transpose(attribution_mask)
    #
    #         fig, ax = plt.subplots(figsize=(60, 6))
    #         sns.heatmap(attribution_mask, linewidth=0.0, ax=ax)
    #         plt.tight_layout()
    #         plt.savefig(figures_folder + "/attribution/track_" + str(i + 1) + "_" + str(cell) + "_" + test_info[i] + ".jpg")
    #         plt.close(fig)
    return None


def recover_shape(v, size_X):
    v = np.asarray(v).flatten()
    end = int((size_X * size_X - size_X) / 2)
    v = v[:end]
    X = np.zeros((size_X, size_X))
    X[np.triu_indices(X.shape[0], k=1)] = v
    X = X + X.T
    return X


def setup_axes1(fig, rect, angle):
    tr = Affine2D().rotate_deg(angle).scale(1, .4)

    grid_helper = floating_axes.GridHelperCurveLinear(tr, extremes=(-1, 23, -1, 19))
    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)
    aux_ax = ax1.get_aux_axes(tr)

    for key in ax1.axis:
        ax1.axis[key].set_visible(False)

    return aux_ax