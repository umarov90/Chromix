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
import matplotlib.ticker as ticker


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
        if "response" in track:
            continue
        for i in range(len(predictions_full)):
            fig = matplotlib.pyplot.figure(figsize=(20, 10))
            ax1 = setup_axes1(fig, 111, -45)
            ax1.set_zorder(1)
            # ax11 = setup_axes1(fig, 111, -45)
            # ax11.set_zorder(0)
            mask = np.tril(np.ones_like(mats[i][0]))
            # mask2 = np.triu(np.ones_like(mats[i][0]))

            # using the upper triangle matrix as mask
            red = ((1.0, 1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 1.0))
            cmap_red = LinearSegmentedColormap.from_list('Custom', red, 256)

            blue = ((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0))
            cmap_blue = LinearSegmentedColormap.from_list('Custom', blue, 256)

            sns.heatmap(mats[i][0], annot=False, mask=mask, ax=ax1, cbar=False, alpha=0.5, cmap=cmap_blue)
            # sns.heatmap(mats[i][1], annot=False, mask=mask, ax=ax1, cbar=False, alpha=0.5, cmap=cmap_red)
            ####################################################
            max_val = np.max(eval_gt_full[i][it])
            tss_layer = test_seq[i][:, 4]
            tss_track = []
            tss_pos = []
            for region in range(0, 210000, bin_size):
                if np.sum(tss_layer[region:region + bin_size]) > 0:
                    tss_track.append(max_val)
                    tss_pos.append(region / bin_size)
                else:
                    tss_track.append(0)
            tss_names = []
            start = eval_infos[i][1] - 105000
            end = eval_infos[i][1] + 105001
            for info in eval_infos:
                if start < info[1] < end:
                    tss_names.append(info[2])
                # if start > info[1] + 105001:
                #     break
            #####################################################
            vector1 = np.pad(predictions_full[i][it], (25, 24), 'constant')
            vector2 = np.pad(eval_gt_full[i][it], (25, 24), 'constant')
            x = range(len(tss_track))
            d1 = {'bin': x, 'expression': vector1}
            df1 = pd.DataFrame(d1)
            d2 = {'bin': x, 'expression': vector2}
            df2 = pd.DataFrame(d2)
            #####################################################
            d3 = {'bin': x, 'expression': tss_track}
            df3 = pd.DataFrame(d3)
            #####################################################
            ax2 = fig.add_axes([0.1, 0.6, 0.8, 0.2])
            sns.lineplot(data=df1, x='bin', y='expression', ax=ax2)
            ax2.fill_between(x, vector1, alpha=0.5)
            sns.lineplot(data=df2, x='bin', y='expression', ax=ax2)
            #####################################################
            sns.barplot(data=df3, x='bin', y='expression', ax=ax2, color='green')
            # try:
            #     for ind, tp in enumerate(tss_pos):
            #         ax2.text(tp, 0, tss_names[ind], color="g")
            # except:
            #     pass
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(100))
            ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
            #####################################################
            ax2.set_title(f"{eval_infos[i][0]}:{eval_infos[i][1] - 105000}-{eval_infos[i][1] + 105001}")
            # fig.tight_layout()
            plt.savefig(f"{fig_path}_{eval_infos[i][2]}_{track}.png")
            plt.close(fig)
            pic_count += 1
        break
        #     if i > 20:
        #         break
        # if pic_count > 100:
        #     break


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
