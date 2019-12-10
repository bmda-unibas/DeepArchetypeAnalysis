import pandas as pd
import numpy as np
import os.path
import imageio

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_jaffe_dimensions(csv, crop):
    """

    :param csv:
    :param crop:
    :return:n_samples, n_labels, image_shape
    """
    jaffe_meta_data = pd.read_csv(csv, header=0, delimiter=" ")
    n_samples = jaffe_meta_data.shape[0]
    # n_labels = jaffe_meta_data.shape[1]
    n_labels = 5

    if crop:
        image_shape = [128, 128]
    else:
        image_shape = [256, 256]
    return n_samples, n_labels, image_shape


def get_jaffe_batch(batch_size, csv, image_path, crop=True, shuffle=True):
    """
    Return one mb
    :param batch_size:
    :param csv:
    :param image_path:
    :param crop:
    :return:image_data, jaffe_meta_data_sampled.as_matrix(['HAP', 'SAD', 'SUR', 'ANG', 'DIS'])
    """
    jaffe_meta_data = pd.read_csv(csv, header=0, delimiter=" ")
    jaffe_meta_data["PIC"] = jaffe_meta_data["PIC"].str.replace("-", ".")
    if shuffle:
        jaffe_meta_data_sampled = jaffe_meta_data.sample(n=batch_size)
    else:
        jaffe_meta_data_sampled = jaffe_meta_data

    image_data = []
    for image_name, number in zip(jaffe_meta_data_sampled["PIC"], jaffe_meta_data_sampled["#"]):
        image_file = "%s.%d.tiff" % (image_name, number)
        image_file_path = os.path.join(image_path, image_file)
        image = imageio.imread(image_file_path)
        if (crop):
            image = image[94:222, 64:192]
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        if (crop):
            size = 128
        else:
            size = 258
        image_data.append(image.reshape(size * size))

    image_data = (np.array(image_data) / 255).reshape(-1, size, size)
    return image_data, jaffe_meta_data_sampled.as_matrix(['HAP', 'SAD', 'SUR', 'ANG', 'DIS'])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img, img_labels = get_jaffe_batch(batch_size=12, csv=dir_path + '/labels.csv',
                                      image_path=dir_path + '/images', crop=True)
    for i in range(3):
        imgplot = plt.imshow(img[i, :].reshape([128, -1]))
        plt.show()
