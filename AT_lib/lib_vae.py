import tensorflow as tf
import numpy as np

tfd = tf.contrib.distributions


def build_prior(dim_latentspace):
    """
    Creates N(0,1) Multivariate Normal prior.
    :param dim_latentspace:
    :return: mvn_diag
    """
    mu = tf.zeros(dim_latentspace)
    rho = tf.ones(dim_latentspace)
    mvn_diag = tfd.MultivariateNormalDiag(mu, rho)
    return mvn_diag


def build_encoder(data, dim_latentspace, z_fixed):
    """
    Basic Encoder Network for DAA.
    Has to be used with tf.make_template for variable sharing.
    :param data:
    :param dim_latentspace:
    :param z_fixed:
    :return:z_predicted, mu_t, sigma_t, t
    """
    nAT = dim_latentspace + 1

    x = tf.layers.flatten(data)
    net = tf.layers.dense(x, 200, tf.nn.relu)
    params = tf.layers.dense(net, 100)
    net, sigma = params[:, :50], params[:, 50:]

    # Weight Matrices
    weights_A = tf.layers.dense(net, nAT, tf.nn.softmax)
    weights_B_t = tf.layers.dense(net, nAT)
    weights_B = tf.nn.softmax(tf.transpose(weights_B_t), 1)

    # latent space parametrization
    mu_t = tf.matmul(weights_A, z_fixed)
    sigma_t = tf.layers.dense(sigma, dim_latentspace, tf.nn.softplus)
    t = tfd.MultivariateNormalDiag(mu_t, sigma_t)

    # predicted archetypes
    z_predicted = tf.matmul(weights_B, mu_t)

    return z_predicted, mu_t, sigma_t, t


def build_encoder_convs(data, dim_latentspace, z_fixed, x_shape=[128, 128]):
    """
    Encoder Network with Convolutions.
    :param data:
    :param dim_latentspace:
    :param z_fixed:
    :return:z_predicted, mu_t, sigma_t, t
    """
    n_at = dim_latentspace + 1
    activation = tf.nn.relu
    net = tf.layers.conv2d(tf.reshape(data, [-1] + x_shape + [1]), filters=64, kernel_size=4, strides=2,
                           padding='same',
                           activation=activation)
    net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
    net = tf.layers.conv2d(net, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
    net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
    net = tf.layers.conv2d(net, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
    net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
    net = tf.layers.flatten(net)

    params = tf.layers.dense(net, 100)
    net, sigma = params[:, :50], params[:, 50:]
    # net = params
    # sigma = tf.layers.dense(sigma, dim_latentSpace)

    # Weight Matrices
    weights_A = tf.layers.dense(net, n_at, tf.nn.softmax)
    weights_B_t = tf.layers.dense(net, n_at)
    weights_B = tf.nn.softmax(tf.transpose(weights_B_t), 1)

    # latent space parametrization
    mu_t = tf.matmul(weights_A, z_fixed)

    sigma_t = tf.layers.dense(sigma, dim_latentspace, tf.nn.softplus)
    t = tfd.MultivariateNormalDiag(mu_t, sigma_t)

    # predicted archetypes
    z_predicted = tf.matmul(weights_B, mu_t)

    return z_predicted, mu_t, sigma_t, t


def dirichlet_prior(dim_latentspace, alpha=1.0):
    """
    Wrapper for tfd.Dirichlet(alpha).
    :param dim_latentspace:
    :param alpha:
    :return:
    """
    nATs = dim_latentspace + 1
    alpha = [alpha] * nATs
    dist = tfd.Dirichlet(alpha)
    return dist


def build_decoder_jaffe_two_labels(latent_code, data_shape):
    """
    Wrapper for build_decoder_jaffe with num_labels=2
    :param latent_code:
    :param data_shape:
    :return:
    """
    return build_decoder_jaffe(latent_code=latent_code, data_shape=data_shape, num_labels=2)


def build_decoder_jaffe(latent_code, data_shape, num_labels=5):
    """
    Decoder for jaffe data
    :param data_shape: e.g. [128, 128].
    :param num_labels: Number of labels to be predicted.
    :return: x_hat, side_info
    """
    activation = tf.nn.relu
    units = 49
    x = tf.layers.dense(latent_code, units=units, activation=activation)
    x = tf.layers.dense(x, units=units, activation=activation)
    recovered_size = int(np.sqrt(units))
    x = tf.reshape(x, [-1, recovered_size, recovered_size, 1])

    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)

    x = tf.contrib.layers.flatten(x)
    x = tf.layers.dense(x, units=np.prod(data_shape), activation=None)

    x = tf.layers.dense(x, units=np.prod(data_shape), activation=tf.nn.sigmoid)
    x_hat = tf.reshape(x, shape=[-1] + data_shape)
    x_hat = tfd.Normal(x_hat, 1.0)
    x_hat = tfd.Independent(x_hat, 2)

    side_info = tf.layers.dense(latent_code, 200, tf.nn.relu)
    side_info = tf.layers.dense(side_info, num_labels, tf.nn.sigmoid) * 5

    return x_hat, side_info


def build_encoder_convs_vae(data, dim_latentspace, z_fixed, x_shape=[128, 128]):
    """
    https://github.com/mmeendez8/Autoencoder/blob/master/autoencoder.py
    :param data:
    :param dim_latentspace:
    :param z_fixed:
    :param x_shape:
    :return:z_predicted, mu_t, sigma_t, t
    """
    activation = tf.nn.relu
    net = tf.layers.conv2d(tf.reshape(data, [-1] + x_shape + [1]), filters=64, kernel_size=4, strides=2,
                           padding='same',
                           activation=activation)
    net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
    net = tf.layers.conv2d(net, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
    net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
    net = tf.layers.conv2d(net, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
    net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
    net = tf.layers.flatten(net)

    params = tf.layers.dense(net, 100)
    net, sigma = params[:, :50], params[:, 50:]

    sigma_t = tf.layers.dense(sigma, dim_latentspace, tf.nn.softplus)
    mu_t = tf.layers.dense(net, dim_latentspace)
    t = tfd.MultivariateNormalDiag(mu_t, sigma_t)

    z_predicted = None

    return z_predicted, mu_t, sigma_t, t
