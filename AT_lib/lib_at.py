import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp


def centralize_matrix(M):
    """
    See https://en.wikipedia.org/wiki/Centering_matrix
    :param M:
    :return:
    """
    n, p = M.shape
    Q = np.full((n, n), -1 / n)
    Q = Q + np.eye(n)
    M = np.dot(np.dot(Q, M), Q)
    return M


def greedy_min_distance(z_f, z_true_mu):
    """
    Calculates the mean L2 Distance between found Archetypes and the true Archetypes (in latent space).
    1. Select the 2 vector with smallest pairwise distance
    2. Calculate the euclidean distance
    3. Remove the 2 vectors and jump to 1.
    :param z_f:
    :param z_true_mu:
    :return: mean loss
    """
    loss = []
    dist = sp.spatial.distance.cdist(z_f, z_true_mu)
    for i in range(z_f.shape[0]):
        z_fixed_idx, z_true_idx = np.unravel_index(dist.argmin(), dist.shape)
        loss.append(dist[z_fixed_idx, z_true_idx])
        dist = np.delete(np.delete(dist, z_fixed_idx, 0), z_true_idx, 1)
    return loss


def create_z_fix(dim_latent_space):
    """
    Creates Coordinates of the Simplex spanned by the Archetypes.

    The simplex will have its centroid at 0.
    The sum of the vertices will be zero.
    The distance of each vertex from the origin will be 1.
    The length of each edge will be constant.
    The dot product of the vectors defining any two vertices will be - 1 / M.
    This also means the angle subtended by the vectors from the origin
    to any two distinct vertices will be arccos ( - 1 / M ).

    :param dim_latent_space:
    :return:
    """

    z_fixed_t = np.zeros([dim_latent_space, dim_latent_space + 1])

    for k in range(0, dim_latent_space):
        s = 0.0
        for i in range(0, k):
            s = s + z_fixed_t[i, k] ** 2

        z_fixed_t[k, k] = np.sqrt(1.0 - s)

        for j in range(k + 1, dim_latent_space + 1):
            s = 0.0
            for i in range(0, k):
                s = s + z_fixed_t[i, k] * z_fixed_t[i, j]

            z_fixed_t[k, j] = (-1.0 / float(dim_latent_space) - s) / z_fixed_t[k, k]
            z_fixed = np.transpose(z_fixed_t)
    return z_fixed


def barycentric_coords(n_per_axis=5):
    """
    Creates coordinates for the traversal of 3 Archetypes (i.e. creates the a weights)
    :param n_per_axis:
    :return: [weights, n_perAxis]; weights has shape (?, 3)
    """
    weights = np.zeros([int((n_per_axis * (n_per_axis + 1)) / 2), 3])

    offset = np.sqrt(3 / 4) / (n_per_axis - 1)
    A = np.array([[1.5, 0, 0], [np.sqrt(3) / 2, np.sqrt(3), 0], [1, 1, 1, ]])
    cnt = 0
    innerCnt = 0
    for i in np.linspace(0, 1.5, n_per_axis):
        startX = i
        startY = cnt * offset

        if n_per_axis - cnt != 1:
            stpY = (np.sqrt(3) - 2 * startY) / (n_per_axis - cnt - 1)
        else:
            stpY = 1
        for j in range(1, n_per_axis - cnt + 1):
            P_x = startX
            P_y = startY + (j - 1) * stpY
            b = np.array([P_x, P_y, 1])
            sol = solve(A, b)

            out = np.abs(np.around(sol, 6))
            weights[innerCnt, :] = out
            innerCnt += 1
        cnt += 1

    return [weights, n_per_axis]


def num_to_pixel_archetypes2d(z_fixed,
                              imgPixel_X,
                              imgPixel_Y,
                              stpX, stpY,
                              originX, originY,
                              sampleMargin,
                              nImg):
    n = z_fixed.shape[0]
    z_fixed_px = np.empty(z_fixed.shape)

    for i in range(n):
        z_x = z_fixed[i, 0]
        z_y = z_fixed[i, 1]
        dist_z_x = np.abs(z_x - originX)
        dist_z_y = np.abs(z_y - originY)
        z_fixed_px[i, 0] = round(dist_z_x * (imgPixel_X / stpX))
        z_fixed_px[i, 1] = (2 * sampleMargin + nImg) * imgPixel_Y - round((dist_z_y * (imgPixel_Y / stpY)))

    return z_fixed_px


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    return True


def create_hinton_plot(z_sampled, weightMatrix, mixture_sampled, figSize=8):
    """
    :param z_sampled: np.array of shape (3, 28, 28)
    :param weightMatrix:  np.array of shape (4, 3)
    :param mixture_sampled: np.array of shape (4, 28, 28)
    :param figSize:
    :return:
    """
    fig = plt.figure(figsize=(figSize, figSize))

    title1 = plt.subplot2grid((10, 5), (0, 0), colspan=1, aspect=.5)
    plt.xticks(())
    plt.yticks(())
    title1.axis('off')
    plt.text(0.5, 0.5, 'Archetypes:  ', ha='center', va='center', size=2.4 * figSize, alpha=.5)

    at1 = plt.subplot2grid((10, 5), (0, 1), colspan=1, rowspan=2, aspect='equal')
    plt.xticks(())
    plt.yticks(())
    for spine in at1.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(2)
    at1.imshow(z_sampled[0], cmap='gray')
    plt.title("Archetype 1", ha='center', va='center', size=1.4 * figSize, alpha=.8)

    at2 = plt.subplot2grid((10, 5), (0, 2), colspan=1, rowspan=2, aspect='equal')
    plt.xticks(())
    plt.yticks(())
    for spine in at2.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(2)
    at2.imshow(z_sampled[1], cmap='gray')
    plt.title("Archetype 2", ha='center', va='center', size=1.4 * figSize, alpha=.8)

    at3 = plt.subplot2grid((10, 5), (0, 3), colspan=1, rowspan=2, aspect='equal')
    plt.xticks(())
    plt.yticks(())
    for spine in at3.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(2)
    plt.title("Archetype 3", ha='center', va='center', size=1.4 * figSize, alpha=.8)
    at3.imshow(z_sampled[2], cmap='gray')

    title2 = plt.subplot2grid((10, 5), (1, 4), colspan=1, rowspan=1, aspect=.5)
    plt.xticks(())
    plt.yticks(())
    title2.axis('off')
    plt.text(0.5, 0.5, 'Mixtures:', ha='center', va='center', size=2.4 * figSize, alpha=.5)

    title3 = plt.subplot2grid((10, 5), (2, 0), colspan=1, rowspan=1, aspect=.5)
    plt.xticks(())
    plt.yticks(())
    title3.axis('off')
    plt.text(0.5, 0.5, 'Weights:', ha='center', va='center', size=2.4 * figSize, alpha=.5)

    hintonPlot = plt.subplot2grid((10, 5), (2, 1), colspan=3, rowspan=8)
    plt.xticks(())
    plt.yticks(())
    hinton(np.transpose(weightMatrix))

    mix1 = plt.subplot2grid((10, 5), (2, 4), colspan=1, rowspan=2, aspect='equal')
    plt.xticks(())
    plt.yticks(())
    for spine in mix1.spines.values():
        spine.set_edgecolor('lightgreen')
        spine.set_linewidth(3)
    plt.ylabel("Mixture 1", ha='center', va='center', size=1.4 * figSize, alpha=.8)
    mix1.imshow(mixture_sampled[0], cmap='gray')

    mix2 = plt.subplot2grid((10, 5), (4, 4), colspan=1, rowspan=2, aspect='equal')
    plt.xticks(())
    plt.yticks(())
    for spine in mix2.spines.values():
        spine.set_edgecolor('lightgreen')
        spine.set_linewidth(3)
    plt.ylabel("Mixture 2", ha='center', va='center', size=1.4 * figSize, alpha=.8)
    mix2.imshow(mixture_sampled[1], cmap='gray')

    mix3 = plt.subplot2grid((10, 5), (6, 4), colspan=1, rowspan=2, aspect='equal')
    plt.xticks(())
    plt.yticks(())
    for spine in mix3.spines.values():
        spine.set_edgecolor('lightgreen')
        spine.set_linewidth(3)
    plt.ylabel("Mixture 3", ha='center', va='center', size=1.4 * figSize, alpha=.8)
    mix3.imshow(mixture_sampled[2], cmap='gray')

    mix4 = plt.subplot2grid((10, 5), (8, 4), colspan=1, rowspan=2, aspect='equal')
    plt.xticks(())
    plt.yticks(())
    for spine in mix4.spines.values():
        spine.set_edgecolor('lightgreen')
        spine.set_linewidth(3)
    plt.ylabel("Mixture 4", ha='center', va='center', size=1.4 * figSize, alpha=.8)
    mix4.imshow(mixture_sampled[3], cmap='gray')

    # plt.tight_layout()
    # plt.savefig('latent_B.png')
    # plt.show()
    # plt.close()
    return fig


def generate_sample_path(A, B, z_fixed, nbSamples=10):
    p = z_fixed.shape[1]
    coord_init = np.dot(A, z_fixed)
    coord_end = np.dot(B, z_fixed)

    # generate samples between init end start coords
    # start at A*z_fixed and go to B*z_fixed
    # remember: A,B are weight vectors, i.e. entries sum to one
    # generate samples between init end start coords
    samplePath = np.zeros([nbSamples, p])
    for alpha in range(nbSamples):
        samplePath[alpha, :] = (alpha / (nbSamples - 1)) * (coord_end - coord_init) + coord_init

    return samplePath


def plot_sample_path(samplePath_imgs, nbRow=5, nbCol=6, figSize=10, data_shape=[128, 128]):
    fig, axes = plt.subplots(nbRow, nbCol, figsize=(figSize, figSize))
    # fig.suptitle("Linear Interpolation between Archetypes (l. to r.)", fontsize=2.4 * figSize)
    # plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=1.8, wspace=0, hspace=0.3)
    # fig1.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
    cnt = 0
    for i in range(nbRow):
        for j in range(nbCol):
            # axes1[i,j].set_title('label = {}'.format("this is a title"))
            image = samplePath_imgs[cnt].reshape(data_shape[0],
                                                 data_shape[1])  # not necessary to reshape if ndim is set to 2
            cnt = cnt + 1

            # axes[0, 0].set_title('Archetype 1', fontsize=1.4 * figSize, fontweight='bold', color='Red')
            axes[0, 0].spines['top'].set_color('red')
            axes[0, 0].spines['top'].set_linewidth(2)
            axes[0, 0].spines['bottom'].set_color('red')
            axes[0, 0].spines['bottom'].set_linewidth(2)
            axes[0, 0].spines['right'].set_color('red')
            axes[0, 0].spines['right'].set_linewidth(2)
            axes[0, 0].spines['left'].set_color('red')
            axes[0, 0].spines['left'].set_linewidth(2)

            # axes[-1, -1].set_title('Archetype 2', fontsize=1.4 * figSize, fontweight='bold', color='red')
            axes[-1, -1].spines['top'].set_color('red')
            axes[-1, -1].spines['top'].set_linewidth(2)
            axes[-1, -1].spines['bottom'].set_color('red')
            axes[-1, -1].spines['bottom'].set_linewidth(2)
            axes[-1, -1].spines['right'].set_color('red')
            axes[-1, -1].spines['right'].set_linewidth(2)
            axes[-1, -1].spines['left'].set_color('red')
            axes[-1, -1].spines['left'].set_linewidth(2)

            axes[i, j].imshow(image, cmap="gray")  # Greys_r #spectral)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    fig.tight_layout()
    return fig
