"""
Contains the Network Architectures.
"""
import tensorflow as tf
import numpy as np

tfd = tf.contrib.distributions


def share_variables(func):
    """
    Wrapper for tf.make_template as decorator.
    :param func:
    :return:
    """
    return tf.make_template(func.__name__, func, create_scope_now_=True)


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


def build_encoder_basic(dim_latentspace, z_fixed):
    """
    Basic DAA Encoder Architecture. Not actually used but more for showcasing the implementation.
    :param dim_latentspace:
    :param z_fixed:
    :return:z_predicted, mu_t, sigma_t, t
    """

    @share_variables
    def encoder(data):
        nAT = dim_latentspace + 1

        x = tf.layers.flatten(data)
        net = tf.layers.dense(x, 200, tf.nn.relu)
        net = tf.layers.dense(net, 100)
        mean_branch, var_branch = net[:, :50], net[:, 50:]

        # Weight Matrices
        weights_A = tf.layers.dense(mean_branch, nAT, tf.nn.softmax)
        weights_B_t = tf.layers.dense(mean_branch, nAT)
        weights_B = tf.nn.softmax(tf.transpose(weights_B_t), 1)

        # latent space parametrization
        mu_t = tf.matmul(weights_A, z_fixed)
        sigma_t = tf.layers.dense(var_branch, dim_latentspace, tf.nn.softplus)
        t = tfd.MultivariateNormalDiag(mu_t, sigma_t)

        # predicted archetypes
        z_predicted = tf.matmul(weights_B, mu_t)

        return {"z_predicted": z_predicted, "mu": mu_t, "sigma": sigma_t, "p": t}

    return encoder


def build_encoder_convs(dim_latentspace, z_fixed, x_shape):
    """
    Returns a function to create encoders with shared variables.
    :param dim_latentspace:
    :param z_fixed:
    :param x_shape:
    :return:
    """

    @share_variables
    def encoder(data):
        nAT = dim_latentspace + 1
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

        # Weight Matrices
        weights_A = tf.layers.dense(net, nAT, tf.nn.softmax)
        weights_B_t = tf.layers.dense(net, nAT)
        weights_B = tf.nn.softmax(tf.transpose(weights_B_t), 1)

        # latent space parametrization as linear combination of archetypes
        mu_t = tf.matmul(weights_A, z_fixed)

        sigma_t = tf.layers.dense(sigma, dim_latentspace, tf.nn.softplus)
        t = tfd.MultivariateNormalDiag(mu_t, sigma_t)

        # predicted archetypes
        z_predicted = tf.matmul(weights_B, mu_t)

        return {"z_predicted": z_predicted, "mu": mu_t, "sigma": sigma_t, "p": t}

    return encoder


def dirichlet_prior(dim_latentspace, alpha=1.0):
    """

    :param dim_latentspace:
    :param alpha:
    :return:
    """
    nATs = dim_latentspace + 1
    alpha = [alpha] * nATs
    dist = tfd.Dirichlet(alpha)
    return dist


def build_decoder(data_shape, num_labels, trainable_var=True):
    """
     Builds Decoder for jaffe data
    :param data_shape:
    :param num_labels:
    :param trainable_var: Make the variance of the decoder trainable.
    :return:
    """
    var = tf.Variable(initial_value=1.0, trainable=trainable_var)

    @share_variables
    def decoder(latent_code):
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
        x = tf.layers.dense(x, units=np.prod(data_shape), activation=activation)

        x = tf.layers.dense(x, units=np.prod(data_shape), activation=tf.nn.sigmoid)
        x_hat = tf.reshape(x, shape=[-1] + data_shape)
        x_hat = tfd.Normal(x_hat, var)
        x_hat = tfd.Independent(x_hat, 2)

        side_info = tf.layers.dense(latent_code, 200, tf.nn.relu)
        side_info = tf.layers.dense(side_info, num_labels, tf.nn.sigmoid) * 5

        return {"x_hat": x_hat, "side_info": side_info}

    return decoder


def build_encoder_vae(dim_latentspace, x_shape):
    """
    Vanilla VAE

    e.g. https://github.com/mmeendez8/Autoencoder/blob/master/autoencoder.py
    :param dim_latentspace:
    :param x_shape:
    :return:z_predicted, mu_t, sigma_t, t
    """

    @share_variables
    def encoder(data):
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

        return {"mu": mu_t, "sigma": sigma_t, "p": t}

    return encoder
