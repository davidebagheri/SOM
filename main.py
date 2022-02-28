import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser

from som import SOM
from decay import ExpDecay

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--img_path", type=str, help="path to image to quantize")
    parser.add_argument("-c", "--n_colors", type=int, help="number of color in the quantization", default=16)
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs in the training process", default=200)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size in the training process", default=30)
    args = parser.parse_args()

    # Load image
    img = Image.open(args.img_path)
    img = np.asarray(img)
    h, w, c = img.shape

    # Training params
    epochs = args.epochs
    batch_size = args.batch_size

    # Init
    n_colors = args.n_colors
    som = SOM(3, n_colors, 0, 255)
    lr = ExpDecay(init_value=0.5, decay_step=10, decay_rate=0.8)
    radius = 10

    for epoch in range(epochs):
        # Gather a batch and train
        for _ in range(batch_size):
            row = np.random.randint(0, h)
            col = np.random.randint(0, w)
            sample = img[row, col, :]

            som.train_step(sample, radius, lr.get_value(epoch))

    # Reconstruct quantized image and save
    out_img = np.zeros(img.shape)
    for i in range(h):
        for j in range(w):
            color = img[i,j,:]
            new_color = som.getClosestCenter(color)
            out_img[i,j,:] = new_color

    Image.fromarray(np.uint8(out_img)).save("out.jpg")