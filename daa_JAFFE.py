#!/usr/bin/env python
# util
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
from itertools import compress
from pathlib import Path
import imageio

# scipy stack + tf
import numpy as np
import pandas as pd
import tensorflow as tf

# custom libs
from AT_lib import lib_vae, lib_at, lib_plt
from jaffe import jaffe_data

from ipdb import set_trace

tfd = tf.contrib.distributions


# If error message "Could not connect to any X display." is issued, uncomment the following line:
# os.environ['QT_QPA_PLATFORM']='offscreen'

def main():
    def build_loss():
        """
        Build all the required losses for the Deep Archetype Model.
        :return: archetype_loss, class_loss, likelihood, divergence, elbo
        """
        likelihood = tf.reduce_sum(x_hat.log_prob(data))
        if args.dir_prior:
            q_sample = t_posterior.sample(50)
            divergence = tf.reduce_mean(encoded_z_data["p"].log_prob(q_sample) - prior.log_prob(q_sample))
        else:
            divergence = tf.reduce_mean(tfd.kl_divergence(t_posterior, prior))

        if not args.vae:
            archetype_loss = tf.losses.mean_squared_error(z_predicted, z_fixed)
        else:
            archetype_loss = tf.constant(0, dtype=tf.float32)
        # Sideinformation Reconstruction loss
        class_loss = args.class_loss_factor * tf.losses.mean_squared_error(predictions=y_hat,
                                                                           labels=side_information)

        elbo = tf.reduce_mean(
            args.recon_loss_factor * likelihood
            - args.class_loss_factor * class_loss
            - args.at_loss_factor * archetype_loss
            - kl_loss_factor * divergence
        )

        return archetype_loss, class_loss, likelihood, divergence, elbo

    ####################################################################################################################
    # ########################################### Data #################################################################

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # # Japanese Face Expressions

    X, Y, jaffe_meta_data = jaffe_data.get_jaffe_data(csv_path=JAFFE_CSV_P, image_path=JAFFE_IMGS_DIR, crop=True)
    Y = Y[..., :args.num_labels]

    train_data = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(buffer_size=180).batch(args.batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((X, Y)).batch(args.batch_size)

    data_shape = list(X.shape[1:])

    all_imgs_ext = np.vstack((X, X[:args.batch_size]))

    # ########################################### Data Placeholders ####################################################
    data = tf.placeholder(tf.float32, [None] + data_shape, 'data')
    side_information = tf.placeholder(tf.float32, [None, args.num_labels], 'labels')
    latent_code = tf.placeholder(tf.float32, [None, args.dim_latentspace], 'latent_code')

    kl_loss_factor = tf.Variable(args.kl_loss_factor, dtype='float32', trainable=False)

    assert data_shape == [data.shape[i].value for i in
                          range(1, len(data.shape))], "Specified data shape does not coincide with data placeholder."

    # ########################################### Model Setup ##########################################################
    z_fixed_ = lib_at.create_z_fix(args.dim_latentspace)
    z_fixed = tf.cast(z_fixed_, tf.float32)
    if not args.vae:
        encoder_net = lib_vae.build_encoder_convs(dim_latentspace=args.dim_latentspace, z_fixed=z_fixed,
                                                  x_shape=[128, 128])
    else:
        encoder_net = lib_vae.build_encoder_vae(dim_latentspace=args.dim_latentspace, x_shape=[128, 128])

    decoder = lib_vae.build_decoder(data_shape=data_shape, num_labels=args.num_labels, trainable_var=args.trainable_var)

    if args.dir_prior:
        prior_dir = tfd.Dirichlet([1.] * nAT)
        prior = tfd.MultivariateNormalDiag(tf.matmul(prior_dir.sample(args.batch_size), z_fixed),
                                           tf.ones(args.dim_latentspace))
    else:
        prior = lib_vae.build_prior(args.dim_latentspace)

    encoded_z_data = encoder_net(data)
    try:
        z_predicted, mu_t, sigma_t, t_posterior = [encoded_z_data[key] for key in ["z_predicted", "mu", "sigma", "p"]]
    except KeyError:
        assert args.vae
        mu_t, sigma_t, t_posterior = [encoded_z_data[key] for key in ["mu", "sigma", "p"]]
    decoded_post_sample = decoder(t_posterior.sample())
    x_hat, y_hat = decoded_post_sample["x_hat"], decoded_post_sample["side_info"]
    latent_decoded = decoder(latent_code)
    latent_decoded_x, latent_decoded_y = latent_decoded["x_hat"], latent_decoded["side_info"]

    # Build the loss
    archetype_loss, sideinfo_loss, likelihood, kl_divergence, elbo = build_loss()

    # Build the optimizer
    optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(-elbo)

    # reconstruction of random samples from a dirichlet
    dirichlet_prior = lib_vae.dirichlet_prior(dim_latentspace=args.dim_latentspace, alpha=0.7)
    num_prior_samples = tf.placeholder(tf.int32, (), 'num_prior_samples')

    if args.vae:
        samples_prior = prior.sample(num_prior_samples, seed=113)
        samples_decoded = decoder(samples_prior)

    else:
        samples_prior = dirichlet_prior.sample(num_prior_samples, seed=113)
        samples_decoded = decoder(tf.matmul(samples_prior, z_fixed))

    # Specify what is to be logged.
    tf.summary.scalar(name='elbo', tensor=elbo)
    tf.summary.scalar(name='archetype_loss', tensor=archetype_loss)
    tf.summary.scalar(name='sideinfo_loss', tensor=sideinfo_loss)
    tf.summary.scalar(name='likelihood', tensor=likelihood)
    tf.summary.scalar(name='kl_divergence', tensor=kl_divergence)

    hyperparameters = [tf.convert_to_tensor([k, str(v)]) for k, v in vars(args).items()]
    tf.summary.text('hyperparameters', tf.stack(hyperparameters))

    summary_op = tf.summary.merge_all()

    ####################################################################################################################
    # ########################################### Plots ################################################################

    def plot_latent_traversal(filename=None, title="", traversal_steps_per_dir=15, z_dims=None):
        """
        Plot latent traversal across the simplex.
        :return:
        """
        traversal_weights, _ = lib_at.barycentric_coords(n_per_axis=traversal_steps_per_dir)
        z_f = z_fixed_.copy()

        if z_f.shape[0] > 3:
            # Just subset of the dimensions for the traversal if we have more than 3 Archetypes
            if z_dims is None:
                z_dims = [0, 1, 2]
            z_f = z_f[z_dims, :]
        elif z_fixed.shape[0] < 3:
            # If 2 archetypes only, just use a subset of the weights..
            traversal_weights = traversal_weights[:traversal_steps_per_dir, 1:]

        traversal_latents = np.dot(traversal_weights, z_f)
        imgs_traversal = sess.run(latent_decoded_x.mean(), feed_dict={latent_code: traversal_latents})
        fig = lib_plt.grid_plot(imgs_traversal,
                                px=data_shape[0],
                                py=data_shape[1],
                                figSize=16, title=title, n_perAxis=traversal_steps_per_dir)
        if filename is None:
            filename = 'latent_traversal_final.png'
        fig.savefig(FINAL_RESULTS_DIR / filename, dpi=600)
        plt.close(fig)

    def plot_z_fixed(path=None):
        """
        Plot at the archetypes Z_fixed.
        :param path: target path for the file. Defaults to FINAL_RESULTS_DIR/'Z_fixed_final.png'
        :return:
        """

        samples_z_fixed = sess.run(latent_decoded_x.mean(), {latent_code: z_fixed_})

        latent_code_train = None
        for i in range(num_mb_its_per_epoch):
            min_idx = i * args.batch_size
            max_idx = min((i + 1) * args.batch_size, X.shape[0])

            tmp = sess.run(mu_t, feed_dict={data: X[min_idx:max_idx],
                                            side_information: Y[min_idx:max_idx]})
            latent_code_train = np.vstack((latent_code_train, tmp)) if latent_code_train is not None else tmp

        assert latent_code_train.shape[0] == X.shape[0]
        fig_zfixed = lib_plt.plot_samples(samples=samples_z_fixed, latent_codes=latent_code_train,
                                          labels=np.argmax(Y, axis=1),
                                          epoch=None, titles=[f"Archetype {i + 1}" for i in range(nAT)])
        # set_trace()
        fig_zfixed.savefig(path, dpi=300)
        plt.close(fig_zfixed)

    def plot_random_samples(path):
        """
        plot random samples drawn from a dirichlet.
        :param path:
        :return:
        """
        # plot random samples (drawn from a dirichlet)
        tensors_rsample = [samples_decoded["x_hat"].mean(),
                           samples_decoded["side_info"],
                           samples_prior]

        rnd_samples_img, rnd_samples_labels, rnd_samples_latents, = sess.run(
            tensors_rsample, feed_dict={num_prior_samples: 49})
        fig_rsamples = lib_plt.plot_samples(samples=rnd_samples_img, latent_codes=rnd_samples_latents,
                                            labels=np.argmax(rnd_samples_labels, axis=1), nrows=5,
                                            epoch=epoch)
        fig_rsamples.savefig(path, dpi=600)
        plt.close(fig_rsamples)

    def plot_hinton(weight_target=0.65):
        """
        Generate Hinton Plots.
        First plots average face (all archetypes weighted equally) and then compares it to weighting
        each of the archetypes with the weight_target parameter (and the others with (1-weight_target)/2 respectively.
        :param weight_target: float between 0 and 1
        :return:
        """
        assert 0 < weight_target < 1
        other_weights = (1 - weight_target) / 2
        weights = np.array([[1.0 / 3] * 3,
                            [other_weights, other_weights, weight_target],
                            [other_weights, weight_target, other_weights],
                            [weight_target, other_weights, other_weights]]).astype(np.float32)
        latent_coords = weights @ z_fixed_
        imgs_hinton, labels_hinton = sess.run([latent_decoded_x.mean(), latent_decoded_y],
                                              feed_dict={latent_code: latent_coords})
        samples_z_f = sess.run(latent_decoded_x.mean(), feed_dict={latent_code: z_fixed_})

        fig = lib_at.create_hinton_plot(samples_z_f, weights, imgs_hinton, figSize=8)
        fig.savefig(FINAL_RESULTS_DIR / 'hinton{weight}.png'.format(weight=weight_target))
        plt.close(fig)

    def plot_interpolation(start_img_str, end_img_str, nb_samples=9, nb_rows=3, nb_cols=3):
        """
        Plot interpolation between two Images.
        :param start_img_str: String of jaffe Image
        :param end_img_str: String of jaffe Image
        :param nb_samples: Number of samples on interpolation
        :param nb_rows: Number of rows in plot
        :param nb_cols: Number of cols in plot
        :return:
        """
        assert nb_rows * nb_cols >= nb_samples, "Please make sure to have enough rows/cols in the plot."
        idx_img1 = jaffe_meta_data[jaffe_meta_data["PIC"] == start_img_str].index[0]
        idx_img2 = jaffe_meta_data[jaffe_meta_data["PIC"] == end_img_str].index[0]

        img_1 = X[[idx_img1], :]
        img_2 = X[[idx_img2], :]

        latent_path_interpol = lib_plt.interpolate_points(coord_init=sess.run(mu_t, {data: img_1}),
                                                          coord_end=sess.run(mu_t, {data: img_2}),
                                                          nb_samples=nb_samples)

        imgs_f32, labels = sess.run([latent_decoded_x.mean(), latent_decoded_y],
                                    feed_dict={latent_code: latent_path_interpol})
        df_labels = pd.DataFrame(labels, columns=df.columns[2:])
        df_labels.to_csv(
            FINAL_RESULTS_DIR / "interpolation_{start}_to_{end}_labels.csv".format(start=start_img_str,
                                                                                   end=end_img_str),
            index=False)
        fig = lib_at.plot_sample_path(samplePath_imgs=imgs_f32, nbRow=nb_rows, nbCol=nb_cols, figSize=10)
        fig.savefig(FINAL_RESULTS_DIR / "interpolation_{start}_to_{end}.png".format(start=start_img_str,
                                                                                    end=end_img_str))

    def create_gif(nb_samples_per_side=20, fps=2):
        """
        Create the GIF of traversing the sides of the latent simplex.

        :param nb_samples_per_side:
        :param fps:
        :return:
        """
        labels_z_fixed = sess.run(latent_decoded_y, {latent_code: z_fixed_})
        img_l = []
        for side in range(3):
            latent_path_interpol = lib_plt.interpolate_points(coord_init=z_fixed_[side, :],
                                                              coord_end=z_fixed_[(side + 1) % 3, :],
                                                              nb_samples=nb_samples_per_side)

            imgs_path, labels_path = sess.run([latent_decoded_x.mean(), latent_decoded_y],
                                              feed_dict={latent_code: latent_path_interpol})

            df_labels = pd.DataFrame(labels_path, columns=["HAP", "SAD", "SUR", "ANG", "DIS"])
            df_labels["id"] = df_labels.index
            df_melted = pd.melt(df_labels, id_vars='id', var_name='emotion', value_name='level')

            for i in range(latent_path_interpol.shape[0]):
                fig = lib_plt.plot_video_img(imgs_path[i, :, :], latent_path_interpol[i, :],
                                             labels_path[i, :], z_fixed_, labels_z_fixed,
                                             df_emotions=df_melted[df_melted.id == i])
                img_l.append(fig)

        imageio.mimsave(FINAL_RESULTS_DIR / 'animation_{nb_p_side}.gif'.format(nb_p_side=nb_samples_per_side),
                        img_l, fps=fps)

    def create_latent_df():
        """
        Create pandas DF with the latent mean coordinates + labels of the data.
        :return: Dataframe  pd.DataFrame(array_all, columns=['ldim0', 'ldim1', 'HAP', 'SAD', 'SUR', 'ANG', 'DIS'])
        """
        test_pos_mean = None
        for i in range(all_imgs_ext.shape[0] // args.batch_size):
            min_idx = i * args.batch_size
            max_idx = (i + 1) * args.batch_size

            tmp_mu = sess.run(mu_t, feed_dict={data: X[min_idx:max_idx],
                                               side_information: Y[min_idx:max_idx]})
            test_pos_mean = np.vstack((test_pos_mean, tmp_mu)) if test_pos_mean is not None else tmp_mu

        test_pos_mean = test_pos_mean[:X.shape[0]]

        array_all = np.hstack((test_pos_mean, Y))
        cols_dims = [f'ldim{i}' for i in range(args.dim_latentspace)]
        df = pd.DataFrame(array_all, columns=cols_dims + ['HAP', 'SAD', 'SUR', 'ANG', 'DIS'])
        return df

    ####################################################################################################################
    # ########################################### Training Loop ########################################################
    num_mb_its_per_epoch = int(np.ceil(X.shape[0] / args.batch_size))

    saver = tf.train.Saver()
    step = 0
    sess.run(tf.global_variables_initializer())
    cur_kl_factor = 5000
    if not args.test_model:
        writer = tf.summary.FileWriter(logdir=TENSORBOARD_DIR, graph=sess.graph)
        for epoch in range(args.n_epochs):
            train_iterator = train_data.make_one_shot_iterator().get_next()
            for b in range(num_mb_its_per_epoch):
                mb_x, mb_y = sess.run(train_iterator)
                feed_train = {data: mb_x, side_information: mb_y, kl_loss_factor: cur_kl_factor}
                sess.run(optimizer, feed_dict=feed_train)
                step += 1

            if epoch % args.test_frequency_epochs == 0:
                cur_kl_factor = max(cur_kl_factor / args.kl_decrease_factor, args.kl_loss_factor)
                print(f"Current KL Loss Factor: {cur_kl_factor}")
                # evaluate metrics on some images; NOTE that this is no real test set
                tensors_test = [summary_op, elbo,
                                likelihood,
                                kl_divergence, archetype_loss, sideinfo_loss]
                test_iterator = test_data.make_one_shot_iterator().get_next()

                test_total_loss, test_likelihood, test_kl, test_atl, test_sideinfol = 0, 0, 0, 0, 0
                for b in range(num_mb_its_per_epoch):
                    mb_x, mb_y = sess.run(test_iterator)
                    feed_test = {data: mb_x, side_information: mb_y, kl_loss_factor: cur_kl_factor}
                    summary, test_total_loss_, test_likelihood_, test_kl_, test_atl_, test_sideinfol_ = sess.run(
                        tensors_test,
                        feed_test)
                    writer.add_summary(summary, global_step=step)

                    test_total_loss += test_total_loss_
                    test_likelihood += test_likelihood_
                    test_kl += test_kl_
                    test_atl += test_atl_
                    test_sideinfol += test_sideinfol_


                test_total_loss /= num_mb_its_per_epoch
                test_likelihood /= num_mb_its_per_epoch
                test_kl /= num_mb_its_per_epoch
                test_atl /= num_mb_its_per_epoch
                test_sideinfol /= num_mb_its_per_epoch

                print(str(args.runNB) + '\nEpoch ' + str(epoch) + ':\n', 'Total Loss:', test_total_loss,
                      '\n Likelihood:', np.mean(test_likelihood),
                      '\n Divergence:', np.mean(test_kl),
                      '\n Archetype Loss:', test_atl,
                      '\n Label Loss:', np.mean(test_sideinfol) / args.class_loss_factor,
                      )

                # reconstruction from the location of the fixed archetypes
                plot_z_fixed(IMGS_DIR / f'Z_fixed_epoch{epoch}.png')
                # plot_random_samples(IMGS_DIR / f'random_sample_epoch{epoch}.png')

            if epoch % args.save_each == 0 and epoch > 0:
                saver.save(sess, save_path=SAVED_MODELS_DIR / "save", global_step=epoch)

        saver.save(sess, save_path=SAVED_MODELS_DIR / "save", global_step=args.n_epochs)
        print("Model Trained!")
        print("Tensorboard Path: {}".format(TENSORBOARD_DIR))
        print("Saved Model Path: {}".format(SAVED_MODELS_DIR))

        # create folder for inference results in the folder of the most recently trained model
        if not FINAL_RESULTS_DIR.exists():
            os.mkdir(FINAL_RESULTS_DIR)

        plot_z_fixed(FINAL_RESULTS_DIR / 'Z_fixed_final.png')

        df = create_latent_df()
        df.to_csv(FINAL_RESULTS_DIR / "latent_codes.csv", index=False)

        plot_latent_traversal()
    else:
        # Load already trained model
        print("Loading Model from {0}".format(SAVED_MODELS_DIR))
        saver.restore(sess, save_path=tf.train.latest_checkpoint(SAVED_MODELS_DIR))
        # create folder for inference results in the folder of the most recently trained model
        os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)

        ################################################################################################################
        # save latent codes
        df = create_latent_df()
        df.to_csv(FINAL_RESULTS_DIR / "latent_codes.csv", index=False)

        ################################################################################################################
        # Plots
        print("Creating plots in '{0}'".format(FINAL_RESULTS_DIR))
        plot_latent_traversal()
        plot_z_fixed()
        plot_hinton(weight_target=0.65)
        plot_interpolation(start_img_str="YM.HA3", end_img_str="MK.SA3")

        # create_gif(nb_samples_per_side=100, fps=9)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Logging Settings
    parser.add_argument('--runNB', type=str, default="1")
    parser.add_argument('--results-path', type=str, default='./Results/JAFFE')
    parser.add_argument('--test-frequency-epochs', type=int, default=5)
    parser.add_argument('--save_each', type=int, default=10000)

    # NN settings
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--n-epochs', type=int, default=5001)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--dim-latentspace', type=int, default=2,
                        help="Number of Archetypes = Latent Space Dimension + 1")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_labels', type=int, default=5)
    parser.add_argument('--trainable-var', dest='trainable_var', action='store_true', default=False,
                        help="Learn variance of decoder. If false, set to constant '1.0'.")

    # DAA loss: weights
    parser.add_argument('--at-loss-factor', type=float, default=80.0)
    parser.add_argument('--class-loss-factor', type=float, default=80.0)
    parser.add_argument('--recon-loss-factor', type=float, default=1.0)
    parser.add_argument('--kl-loss-factor', type=float, default=40.0)
    parser.add_argument('--kl-decrease-factor', type=float, default=1.5)

    # loading already existing model
    parser.add_argument('--test-model', dest='test_model', action='store_false', default=False)
    parser.add_argument('--model-substr', type=str, default=None)

    # Different settings for the prior
    parser.add_argument('--dir-prior', dest='dir_prior', action='store_true', default=False,
                        help="Use the dirichlet + Gauss noise prior instead of a standard normal.")
    parser.add_argument('--vae', dest='vae', action='store_true', default=False,
                        help="Train standard vae instead of AT.")

    args = parser.parse_args()
    print(args)

    assert 0 < args.num_labels <= 5, "Choose up to 5 labels."

    nAT = args.dim_latentspace + 1

    # GPU target
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # all the path/directory stuff
    CUR_DIR = Path(__file__).resolve().parent
    RESULTS_DIR = CUR_DIR / 'Results/JAFFE'
    if not args.test_model:
        # create new model directory
        MODEL_DIR = RESULTS_DIR / "{time}_{run_name}_{dim_lspace}_{mb_size}_{n_epochs}".format(
            time=datetime.now().replace(second=0, microsecond=0),
            run_name=args.runNB, dim_lspace=args.dim_latentspace, mb_size=args.batch_size, n_epochs=args.n_epochs)
    else:
        # get latest trained model matching to args.model_substr
        all_results = os.listdir(RESULTS_DIR)
        if args.model_substr is not None:
            idx = [args.model_substr in res for res in all_results]
            all_results = list(compress(all_results, idx))
        all_results.sort()
        MODEL_DIR = RESULTS_DIR / all_results[-1]

    FINAL_RESULTS_DIR = MODEL_DIR / 'final_results/'
    TENSORBOARD_DIR = MODEL_DIR / 'Tensorboard'
    IMGS_DIR = MODEL_DIR / 'imgs'
    SAVED_MODELS_DIR = MODEL_DIR / 'Saved_models/'
    VIDEO_IMGS_DIR = FINAL_RESULTS_DIR / "video_imgs"

    JAFFE_CSV_P = CUR_DIR / 'jaffe/labels.csv'
    JAFFE_IMGS_DIR = CUR_DIR / 'jaffe/images'

    if not args.test_model:
        for path in [TENSORBOARD_DIR, SAVED_MODELS_DIR, IMGS_DIR]:
            os.makedirs(path, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    nAT = args.dim_latentspace + 1
    main()
