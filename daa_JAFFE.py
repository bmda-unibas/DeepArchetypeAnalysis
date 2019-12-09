#!/usr/bin/env python
# util
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
from itertools import compress
from pathlib import Path
import shutil
import imageio

# scipy stack + tf
import numpy as np
import pandas as pd
import tensorflow as tf

# custom libs
from AT_lib import lib_vae, lib_at, lib_plt
from jaffe import jaffe_data

tfd = tf.contrib.distributions


def main():
    def build_loss():
        """
        Build all the required losses for the Deep Archetype Model.
        :return: archetype_loss, class_loss, likelihood, divergence, elbo
        """
        likelihood = tf.reduce_mean(x_hat.log_prob(data))
        divergence = tf.reduce_mean(tfd.kl_divergence(t_posterior, prior))

        archetype_loss = tf.losses.mean_squared_error(z_predicted, z_fixed)

        # Sideinformation Reconstruction Loss
        sideinfo_loss = args.class_loss_factor * tf.losses.mean_squared_error(predictions=y_hat,
                                                                              labels=side_information)

        elbo = tf.reduce_mean(
            args.recon_loss_factor * likelihood
            - args.class_loss_factor * sideinfo_loss
            - args.at_loss_factor * archetype_loss
            - args.kl_loss_factor * divergence
        )

        return archetype_loss, sideinfo_loss, likelihood, divergence, elbo

    ####################################################################################################################
    # ########################################### Data #################################################################

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # # Japanese Face Expressions

    num_total_samples, n_labels, data_shape = jaffe_data.get_jaffe_dimensions(JAFFE_CSV_P, crop=True)
    n_batches = int(num_total_samples / args.batch_size)

    # which two labels to use, if we don't use all of them
    selected_labels = [1, 4]

    def get_next_batch(batch_size):
        """
        Helper function for getting mini batches.
        :param batch_size:
        :return: mb_x, mb_y
        """
        mb_x, mb_y = jaffe_data.get_jaffe_batch(batch_size=batch_size,
                                                csv=JAFFE_CSV_P,
                                                image_path=JAFFE_IMGS_DIR, crop=True)
        if args.two_labels_only:
            mb_y = mb_y[:, selected_labels].reshape(-1, len(selected_labels))
        return mb_x, mb_y

    all_images, all_labels = jaffe_data.get_jaffe_batch(batch_size=180,
                                                        csv=JAFFE_CSV_P,
                                                        image_path=JAFFE_IMGS_DIR, crop=True, shuffle=False)
    if args.two_labels_only:
        all_labels = all_labels[:, selected_labels].reshape(-1, len(selected_labels))
    jaffe_meta_data = pd.read_csv(JAFFE_CSV_P, header=0, delimiter=" ")

    ####################################################################################################################
    # ########################################### Data Placeholders ####################################################
    data = tf.placeholder(tf.float32, [None] + data_shape, 'data')
    num_labels = 5
    if args.two_labels_only:
        num_labels = len(selected_labels)
    labels = tf.placeholder(tf.float32, [None, num_labels], 'labels')
    latent_code = tf.placeholder(tf.float32, [None, args.dim_latentspace], 'latent_code')
    side_information = labels

    assert data_shape == [data.shape[i].value for i in
                          range(1, len(data.shape))], "Specified data shape does not coincide with data placeholder."

    ####################################################################################################################
    # ########################################### Model Setup ##########################################################
    encoder = tf.make_template('encoder', lib_vae.build_encoder_convs)
    if not args.two_labels_only:
        decoder = tf.make_template('decoder', lib_vae.build_decoder_jaffe)
    else:
        decoder = tf.make_template('decoder', lib_vae.build_decoder_jaffe_two_labels)
    # initialization of fixed archetype positions
    z_fixed_ = lib_at.create_z_fix(args.dim_latentspace)
    z_fixed = tf.cast(z_fixed_, tf.float32)

    prior = lib_vae.build_prior(dim_latentspace=args.dim_latentspace)
    z_predicted, mu_t, sigma_t, t_posterior = encoder(data, args.dim_latentspace, z_fixed)
    x_hat, y_hat = decoder(t_posterior.sample(), data_shape)
    latent_decoded_x, latent_decoded_y = decoder(latent_code, data_shape)

    # Build the loss
    archetype_loss, sideinfo_loss, likelihood, kl_divergence, elbo = build_loss()

    # Build the optimizer
    optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(-elbo)

    # reconstruction of random samples from a dirichlet
    dirichlet_prior = lib_vae.dirichlet_prior(dim_latentspace=args.dim_latentspace, alpha=0.7)
    num_prior_samples = tf.placeholder(tf.int32, (), 'num_prior_samples')
    samples_dirichlet_prior = dirichlet_prior.sample(num_prior_samples, seed=113)
    samples_dirichlet_decoded = decoder(tf.matmul(samples_dirichlet_prior, z_fixed), data_shape)
    rand_samples_img_tf = samples_dirichlet_decoded[0].mean()
    rand_samples_labels_tf = samples_dirichlet_decoded[1]

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

    def plot_z_fixed():
        """
        Plot at the archetypes Z_fixed.
        :return:
        """
        samples_z_fixed = sess.run(latent_decoded_x.mean(), {latent_code: z_fixed_})
        latent_code_test = sess.run(mu_t, feed_dict={data: all_images, labels: all_labels})

        fig_zfixed = lib_plt.plot_samples(samples=samples_z_fixed, latent_codes=latent_code_test,
                                          labels=all_labels[:, 0],
                                          epoch=None, titles=["Archetype 1", "Archetype 2", "Archetype 3"])
        fig_zfixed.savefig(FINAL_RESULTS_DIR / 'Z_fixed_final.png', dpi=600)
        plt.close(fig_zfixed)

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

        img_1 = all_images[[idx_img1], :]
        img_2 = all_images[[idx_img2], :]

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

    ####################################################################################################################
    # ########################################### Training Loop ########################################################

    saver = tf.train.Saver()
    step = 0
    sess.run(tf.global_variables_initializer())
    if args.train_model:
        writer = tf.summary.FileWriter(logdir=TENSORBOARD_DIR, graph=sess.graph)
        for epoch in range(args.n_epochs):
            for b in range(n_batches):
                mb_x, mb_y = get_next_batch(args.batch_size)

                feed_train = {data: mb_x, labels: mb_y}
                sess.run(optimizer, feed_dict=feed_train)

                tensors_train = [elbo,
                                 archetype_loss,
                                 sideinfo_loss,
                                 likelihood,
                                 kl_divergence]
                # tot_loss_train, at_loss_train, sideinfo_loss_train, likelihood_train, divergence_train = \
                sess.run(tensors_train, feed_train)
                step += 1

            if epoch % args.test_frequency_epochs == 0:
                # evaluate metrics on all images; NOTE that this is no real test set but just all JAFFE images
                tensors_test = [summary_op, elbo,
                                likelihood,
                                kl_divergence, archetype_loss, sideinfo_loss]
                feed_test = {data: all_images, labels: all_labels}
                summary, test_total_loss, test_likelihood, test_kl, test_atl, test_sideinfol = sess.run(tensors_test,
                                                                                                        feed_test)

                writer.add_summary(summary, global_step=step)
                print(str(args.runNB) + '\nEpoch ' + str(epoch) + ':\n', 'Total Loss:', test_total_loss,
                      '\n Likelihood:', np.mean(test_likelihood),
                      '\n Divergence:', np.mean(test_kl),
                      '\n Archetype Loss:', test_atl,
                      '\n Label Loss:', np.mean(test_sideinfol),
                      )

                ########################################################################################################
                # draw some random samples & get the reconstruction from the location of the fixed archetypes
                ########################################################################################################
                tensors_rsample = [rand_samples_img_tf,
                                   rand_samples_labels_tf,
                                   samples_dirichlet_prior]

                rnd_samples_img, rnd_samples_labels, rnd_samples_latents, = sess.run(
                    tensors_rsample, feed_dict={num_prior_samples: 49})
                samples_z_f, labels_z_f = sess.run([latent_decoded_x.mean(), latent_decoded_y],
                                                   feed_dict={latent_code: z_fixed_})
                test_pos_mu = sess.run(mu_t, feed_test)

                # plot z fixed
                if test_pos_mu.shape[1] == 1:
                    # if only 2 archetypes, set y values to zero for plotting
                    test_pos_mu = np.hstack((test_pos_mu, np.zeros_like(test_pos_mu)))
                fig_zfixed = lib_plt.plot_samples(samples=samples_z_f, latent_codes=test_pos_mu,
                                                  labels=all_labels[:, 0],
                                                  epoch=epoch)
                fig_zfixed.savefig(IMGS_DIR / 'Z_fixed_epoch{epoch}.png'.format(epoch=epoch), dpi=600)
                plt.close(fig_zfixed)

                # plot random samples (drawn from a dirichlet)
                fig_rsamples = lib_plt.plot_samples(samples=rnd_samples_img, latent_codes=rnd_samples_latents,
                                                    labels=rnd_samples_labels[:, 0], nrows=5,
                                                    epoch=epoch)
                fig_rsamples.savefig(IMGS_DIR / 'random_sample_epoch{epoch}.png'.format(epoch=epoch), dpi=600)
                plt.close(fig_rsamples)

            if epoch % args.save_each == 0:
                saver.save(sess, save_path=SAVED_MODELS_DIR, global_step=epoch)

        saver.save(sess, save_path=SAVED_MODELS_DIR, global_step=args.n_epochs)
        print("Model Trained!")
        print("Tensorboard Path: {}".format(TENSORBOARD_DIR))
        print("Saved Model Path: {}".format(SAVED_MODELS_DIR))

        # create folder for inference results in the folder of the most recently trained model
        if not FINAL_RESULTS_DIR.exists():
            os.mkdir(FINAL_RESULTS_DIR)

        plot_latent_traversal()
        plot_z_fixed()
    else:
        # Load already trained model
        print("Loading Model from {0}".format(SAVED_MODELS_DIR))
        saver.restore(sess, save_path=tf.train.latest_checkpoint(SAVED_MODELS_DIR))
        # create folder for inference results in the folder of the most recently trained model
        os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)

        ################################################################################################################
        # save latent codes
        feed_test = {data: all_images, labels: all_labels}

        test_pos_mean, test_pos_sigma = sess.run([mu_t, sigma_t], feed_test)
        array_all = np.hstack((test_pos_mean, all_labels))
        df = pd.DataFrame(array_all, columns=['ldim0', 'ldim1', 'HAP', 'SAD', 'SUR', 'ANG', 'DIS'])
        df.to_csv(FINAL_RESULTS_DIR / "latent_codes.csv", index=False)

        ################################################################################################################
        # Plots
        print("Creating plots in '{0}'".format(FINAL_RESULTS_DIR))
        plot_latent_traversal()
        plot_z_fixed()
        plot_hinton(weight_target=0.65)
        plot_interpolation(start_img_str="YM.HA3", end_img_str="MK.SA3")

        create_gif(nb_samples_per_side=100, fps=9)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Logging Settings
    parser.add_argument('--runNB', type=str, default="1")
    parser.add_argument('--test-frequency-epochs', type=int, default=50)
    parser.add_argument('--save_each', type=int, default=1000)

    # NN settings
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--dim-latentspace', type=int, default=2)
    parser.add_argument('--n-epochs', type=int, default=201)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--at-loss-factor', type=float, default=80.0)
    parser.add_argument('--class-loss-factor', type=float, default=80.0)
    parser.add_argument('--recon-loss-factor', type=float, default=1.0)
    parser.add_argument('--kl-loss-factor', type=float, default=40.0)

    # synthetic data settings
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=None)

    # call --test-model if you don't want it in train mode
    parser.add_argument('--test-model', dest='train_model', action='store_false')
    parser.set_defaults(train_model=True)
    parser.add_argument('--model-substr', type=str, default=None)
    parser.add_argument('--two-labels-only', dest='two_labels_only', action='store_true',
                        help="Only use two labels (Sadness and Disgust).")
    parser.set_defaults(two_labels_only=False)

    args = parser.parse_args()
    print(args)

    # GPU target
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # all the path/directory stuff
    CUR_DIR = Path(__file__).resolve().parent
    RESULTS_DIR = CUR_DIR / 'Results/JAFFE'
    if args.train_model:
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

    if args.train_model:
        for path in [TENSORBOARD_DIR, SAVED_MODELS_DIR, IMGS_DIR]:
            os.makedirs(path, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    nAT = args.dim_latentspace + 1
    main()
