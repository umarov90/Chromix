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
import matplotlib.gridspec as gridspec


def draw_tracks(p, track_names, predictions_full, eval_gt_full,
                test_seq, infos, hic_keys,
                predictions_hic, hic_output, fig_path):
    mats = {}
    for h in range(len(hic_keys)):
        it = h * p.hic_track_size
        it2 = h * p.hic_track_size
        for i in range(len(predictions_hic)):
            mat_gt = recover_shape(hic_output[i][it:it + p.hic_track_size], p.num_hic_bins)
            mat_pred = recover_shape(predictions_hic[i][it2:it2 + p.hic_track_size], p.num_hic_bins)
            mats.setdefault(h, []).append([mat_gt, mat_pred])

    gene_info = pd.read_csv("data/hg38.GENCODEv38.pc_lnc.gene.info.tsv", sep="\t", index_col=False)
    print("Drawing tracks")
    hic_to_draw = len(hic_keys) - 2 # 1 hic key
    types_to_draw = ["scEnd5", "scATAC"]
    types = []
    for it, track in enumerate(track_names):
        type = track[:track.find(".")]
        if type not in types_to_draw:
            continue
        if type in types:
            continue
        if "response" in track:
            continue
        r = list(range(len(predictions_full)))
        random.shuffle(r)
        r = r[:50]
        for i in r:
            fig = plt.figure(tight_layout=True, figsize=(14, 7))
            gs = gridspec.GridSpec(2, 2)

            ax00 = fig.add_subplot(gs[0, :])

            ax0 = fig.add_subplot(gs[1, 0])
            ax1 = fig.add_subplot(gs[1, 1])

            if len(hic_keys) > 0:
                mat_gt = mats[hic_to_draw][i][0]
                mat_pred = mats[hic_to_draw][i][1]

                sns.heatmap(mat_pred, linewidth=0.0, ax=ax0, square=True)
                ax0.set_title("Prediction")
                sns.heatmap(mat_gt, linewidth=0.0, ax=ax1, square=True)
                ax1.set_title("Ground truth")

            ####################################################
            max_val = np.max(eval_gt_full[i][it])
            tss_layer = test_seq[i][:, 4]
            tss_track = []
            tss_pos = []
            for region in range(0, 2*p.half_size, p.bin_size):
                if np.sum(tss_layer[region:region + p.bin_size]) > 0:
                    tss_track.append(max_val)
                    tss_pos.append(region / p.bin_size)
                else:
                    tss_track.append(0)
            tss_names = []
            start = infos[i][1] - p.half_size
            end = infos[i][1] + p.half_size + 1
            chrom = infos[i][0]
            # df["geneName"][(df["chrom"] == chrom) & (df["C"] == 900) & (df["C"] == 900)]
            for info in infos:
                if start < info[1] < end:
                    tss_names.append(gene_info.loc[gene_info['geneID'] == info[2], 'geneName'].iloc[0])
                # if start > info[1] + 105001:
                #     break
            #####################################################
            # print(f"tss layer {np.sum(tss_layer)}")
            # print(f"tss names {len(tss_names)}")
            vector1 = np.pad(predictions_full[i][it], (550, 549), 'constant')
            vector2 = np.pad(eval_gt_full[i][it], (550, 549), 'constant')
            x = range(len(tss_track))
            d1 = {'bin': x, 'expression': vector1}
            df1 = pd.DataFrame(d1)
            d2 = {'bin': x, 'expression': vector2}
            df2 = pd.DataFrame(d2)
            #####################################################
            d3 = {'bin': x, 'expression': tss_track}
            df3 = pd.DataFrame(d3)
            #####################################################
            sns.lineplot(data=df1, x='bin', y='expression', ax=ax00)
            ax00.fill_between(x, vector1, alpha=0.5)
            sns.lineplot(data=df2, x='bin', y='expression', ax=ax00)
            #####################################################
            sns.barplot(data=df3, x='bin', y='expression', ax=ax00, color='green')
            try:
                for ind, tp in enumerate(tss_pos):
                    md = ind % 5
                    ax00.text(tp, (max_val/5) * md, tss_names[ind], color="g")
            except:
                pass
            ax00.xaxis.set_major_locator(ticker.MultipleLocator(100))
            ax00.xaxis.set_major_formatter(ticker.ScalarFormatter())
            #####################################################
            ax00.set_title(f"{infos[i][0]}:{infos[i][1] - p.half_size}-{infos[i][1] + p.half_size + 1}")
            # fig.tight_layout()
            plt.savefig(f"{fig_path}{infos[i][2]}_{track}.png")
            plt.close(fig)
        types.append(type)
        if len(types) == len(types_to_draw):
            break


def draw_regplots(track_names, track_perf, final_pred, eval_gt, fig_path):
    pic_count = 0
    print("Drawing gene regplot")
    for it, track in enumerate(track_names):
        type = track[:track.find(".")]
        if type not in ["CAGE"]:
            continue
        if track_perf[track] < 0.85:
            continue

        a = []
        b = []
        for gene in eval_gt.keys():
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
    # baseline = tf.zeros(shape=(input_size, num_features))
    # image = ns.astype('float32')
    # ig_attributions = attribution.integrated_gradients(our_model, baseline=baseline,
    #                                                    image=image,
    #                                                    target_class_idx=[mid_bin, track_to_use],
    #                                                    m_steps=40)
    #
    # attribution_mask = tf.squeeze(ig_attributions).numpy()
    # attribution_mask = (attribution_mask - np.min(attribution_mask)) / (
    #         np.max(attribution_mask) - np.min(attribution_mask))
    # attribution_mask = np.mean(attribution_mask, axis=-1, keepdims=True)
    # # attribution_mask[int(input_size / 2) - 1000 : int(input_size / 2) + 1000, :] = np.nan
    # print(attribution_mask.shape)
    # attribution_mask = attribution_mask[int(input_size / 2) - 1000: int(input_size / 2) + 1000, :4]
    # # attribution_mask = measure.block_reduce(attribution_mask, (100, 1), np.mean)
    # print(attribution_mask.shape)
    # attribution_mask = np.transpose(attribution_mask)
    # print(attribution_mask.shape)
    # fig, ax = plt.subplots(figsize=(60, 6))
    # sns.heatmap(attribution_mask, linewidth=0.0, ax=ax)
    # plt.tight_layout()
    # plt.savefig(f"temp/{chrn}.{chrp}.attribution.png")
    # plt.close(fig)
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
