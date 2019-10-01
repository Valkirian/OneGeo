import cv2
import matplotlib.pyplot as plt


def show_channel_weights(image, title):

    hls_channels = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HLS))
    means = [ ch.mean() for ch in hls_channels ]

    for label, ch, mean in zip("HLS", hls_channels, means):
        fig = plt.figure()
        ax = plt.imshow(ch/mean - 1)
        ax.set_cmap('coolwarm')
        plt.title("Channel {} (mean = {})".format(label, int(round(mean))))
        plt.colorbar()
        fig.canvas.set_window_title(title)

    #Need to call plt.show() or be in interactive mode (plt.ion) for the plots
    # to show up
