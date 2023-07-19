import math

import matplotlib.pyplot as plt
from PIL import Image
import liniar_regresion as tf


# [2]
def display_images(images, labels, num_columns=5):
    num_rows = math.ceil(len(images) / num_columns)
    fig = plt.figure(figsize=(num_columns * 3, num_rows * 3))

    for i, image_path in enumerate(images):
        ax = fig.add_subplot(num_rows, num_columns, i + 1)
        ax.axis('off')

        image = Image.open(image_path)
        ax.imshow(image)
        ax.set_title(labels[i])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example usage
    # image_paths = ['./pictures/flower_photos/daisy/100080576_f52e8ee070_n.jpg']
    # labels = ['daisy']
    #display_images(image_paths, labels)

    node2 = tf.constant(6, tf.float32)

    ses = tf.Session()
    print(ses.run[node1, node2])

    ses.close()
