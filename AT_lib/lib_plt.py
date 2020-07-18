import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def interpolate_points(coord_init, coord_end, nb_samples):
    """
    Utility function for interpolation between two points.
    :param coord_init:
    :param coord_end:
    :param nb_samples:
    :return:
    """
    interpol = np.zeros([nb_samples, 2])
    for alpha in range(nb_samples):
        interpol[alpha, :] = (alpha / (nb_samples - 1)) * (coord_end - coord_init) + coord_init
    return interpol


def plot_samples(samples, latent_codes, labels,
                 epoch, nrows=1,
                 titles=None, size=1, latent_ticks=False, cmap='gray'):
    """
        Plots the given samples as well as the latent codes (if the latent dimension is <3).
    :param samples:
    :param latent_codes:
    :param labels:
    :param epoch:
    :param runNB:
    :param nrows:
    :param title:
    :param name_prefix:
    :param size:
    :param latent_ticks:
    :return:
    """
    assert titles is None or len(titles) == len(samples)

    Z_fix = np.array([[1., 0.],
                      [-0.5, 0.8660254],
                      [-0.5, -0.8660254]])

    ncols = np.ceil(len(samples) / nrows)
    if latent_codes.shape[1] <= 3 and ncols * nrows < len(samples) + 1:
        ncols += 1

    fig = plt.figure(figsize=(ncols * size, nrows * size))

    no_ticks = dict(left=False, bottom=False, labelleft=False, labelbottom=False)
    title = ''
    if latent_codes.shape[1] <= 3:
        if latent_codes.shape[1] == 3:
            ax = fig.add_subplot(nrows, ncols, 1, projection='3d')
            if epoch is not None:
                title = 'Epoch {0}   -  {1}'.format(epoch, title)

        else:
            ax = fig.add_subplot(nrows, ncols, 1)
            if epoch is not None:
                ax.set_ylabel('Epoch {}'.format(epoch))
            ax.set_aspect('equal')
            # ax.plot(Z_fix[0, :], Z_fix[1, :], 'r-')
            # ax.plot(Z_fix[1, :], Z_fix[2, :], 'r-')
            # ax.plot(Z_fix[2, :], Z_fix[0, :], 'r-')
        colors = ["blue", "orange", "green", "purple", "black"]
        label_str = ["HAP", "SAD", "SUR", "ANG", "DIS"]
        for lab in np.unique(labels):
            coords = latent_codes[labels == lab, ...]
            # from ipdb import set_trace
            # set_trace()
            ax.scatter(coords[:, 0], coords[:, 1],
                       s=3, c=np.repeat(colors[lab], coords.shape[0]), label=label_str[lab], alpha=0.5,
                       marker='o')
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
        # ax.scatter(*[latent_codes[:, i] for i in range(latent_codes.shape[1])],
        #            s=5, c=labels, label=np.unique(labels), alpha=0.5)
        ax.legend(fontsize=2)

        if not latent_ticks:
            ax.set_xlim(latent_codes.min() - .1, latent_codes.max() + .1)
            ax.set_ylim(latent_codes.min() - .1, latent_codes.max() + .1)
            ax.tick_params(axis='both', which='both', **no_ticks)
        else:
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.xaxis.label.set_size(5)
            ax.yaxis.label.set_size(5)
            ax.tick_params(axis='both', which='both', labelsize=4)
        offset = 1
    else:
        offset = 0

    i = 0
    for index, sample in enumerate(samples):
        ax = fig.add_subplot(nrows, ncols, offset + index + 1)
        if latent_codes.shape[1] > 3 and index == 0 and epoch is not None:
            ax.set_ylabel('Epoch {}'.format(epoch))
        ax.imshow(sample, cmap=cmap)
        if titles is not None:
            ax.set_title(titles[i], ha='center', va='center', alpha=.8, size=7)
            i += 1

        # ax.tick_params(axis='both', which='both', **no_ticks)
        ax.axis('off')

    fig.suptitle(title, fontsize=8)
    return fig


def plot_video_img(image_point, latent_point, label_point, z_fixed_, labels_z_fixed, df_emotions):
    """

    :param image_point:
    :param latent_point:
    :param label_point:
    :param z_fixed_:
    :param labels_z_fixed:
    :param df_emotions:
    :return:
    """

    def get_color(arr): return sns.color_palette()[np.argmax(arr)]

    f = plt.figure(figsize=(15, 5))
    ax = f.add_subplot(131)
    ax2 = f.add_subplot(132)
    ax3 = f.add_subplot(133)

    # plot simplex
    t1 = plt.Polygon(z_fixed_, facecolor="whitesmoke", edgecolor="black", lw=1, zorder=0)
    ax.add_patch(t1)
    ax.set_xlim((-0.75, 1.25))
    ax.set_ylim((-1, 1))
    ax.axis('off')
    ax.set_aspect('equal')

    ax.scatter(x=z_fixed_[0, 0], y=z_fixed_[0, 1], color=get_color(labels_z_fixed[0]), marker='*', s=80)
    ax.annotate("AT1", z_fixed_[0, :] + np.array([0.05, 0]))
    ax.scatter(x=z_fixed_[1, 0], y=z_fixed_[1, 1], color=get_color(labels_z_fixed[1]), marker='*', s=80)
    ax.annotate("AT2", z_fixed_[1, :] + np.array([0.05, 0]))
    ax.scatter(x=z_fixed_[2, 0], y=z_fixed_[2, 1], color=get_color(labels_z_fixed[2]), marker='*', s=80)
    ax.annotate("AT3", z_fixed_[2, :] + np.array([0.05, -0.08]))

    # plot current point
    ax.scatter(x=latent_point[0], y=latent_point[1], color=get_color(label_point), s=150)

    # remove axis from image subplot & make a frame corresponding to maximum label
    for spine in ax2.spines.values():
        spine.set_edgecolor(get_color(label_point))
        spine.set_linewidth(4)

    ax2.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        labelleft=False,
        left=False)
    # plot image
    ax2.imshow(image_point, cmap="gray")

    # create barplot of labels
    sns.barplot(data=df_emotions, x='emotion', y='level', ax=ax3)
    ax3.set_ylim((1, 5))

    f.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
    plt.close(f)
    return image


def grid_plot(imgs_sampled, n_perAxis, px=28, py=28, figSize=16, channels=1, title=""):
    """
    Plot images on archetype grid
    :param imgs_sampled:
    :param n_perAxis:
    :param px:
    :param py:
    :param figSize:
    :return:
    """
    if channels == 1:
        imgMatrix = np.ones([n_perAxis * px, n_perAxis * py])
        cmap = "gray"
    else:
        imgMatrix = np.ones([n_perAxis * px, n_perAxis * py, channels])
        cmap = None
    nbCols = np.zeros(n_perAxis)
    cnt = 0
    imgC = 0
    for i in range(n_perAxis, 0, -1):
        nbCols[cnt] = i
        cnt += 1
    nbCols = nbCols.astype(int)
    # print(nbCols)

    for i in range(n_perAxis):
        ccc = 0
        for j in range(nbCols[i]):
            if (j == 0):
                imgMatrix[i * px:(i + 1) * px, ccc * py:(ccc + 1) * py] = imgs_sampled[imgC, :, :]
            else:
                imgMatrix[int(0.5 * j * px + i * px): int(0.5 * j * px + (i + 1) * px),
                ccc * py:(ccc + 1) * py] = imgs_sampled[imgC, :, :]

            ccc += 1
            imgC += 1

    fig = plt.figure(figsize=(figSize, figSize))
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    no_ticks = dict(left='off', bottom='off', labelleft='off', labelbottom='off')
    ax.tick_params(axis='both', which='both', **no_ticks)
    ax.set_title(title, fontsize=1.6 * figSize)
    # zzzM[ zzzM < -0.1 ] = 0 # HACK !!
    ax.imshow(imgMatrix, cmap=cmap)
    return fig
