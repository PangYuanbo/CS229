from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    Height, Width, Channels = image.shape
    centroids_init = np.zeros((num_clusters, Channels), dtype=np.float32)
    for i in range(num_clusters):
        # Randomly select a pixel from the image
        random_row = random.randint(0, Height - 1)
        random_col = random.randint(0, Width - 1)
        # Assign the RGB values of the selected pixel to the centroid
        centroids_init[i] = image[random_row, random_col]
        # Ensure the centroid is a float32 type
        centroids_init[i] = centroids_init[i].astype(np.float32)
    # Ensure centroids are unique by checking for duplicates
    # *** END YOUR CODE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
        # Usually expected to converge long before `max_iter` iterations
                # Initialize `dist` vector to keep track of distance to every centroid
                # Loop over all centroids and store distances in `dist`
                # Find closest centroid and update `new_centroids`
        # Update `new_centroids`
    Height, Width, Channels = image.shape
    for iteration in range(max_iter):
        if iteration % print_every == 0:
            print(f'Iteration {iteration}/{max_iter}')
        centroids_number=np.zeros(centroids.shape[0], dtype=np.float32)
        new_centroids=np.zeros(centroids.shape, dtype=np.float32)
        for h in range(Height):
            for w in range(Width):
                pixel = image[h, w]
                dist=np.zeros(centroids.shape[0], dtype=np.float32)
                for i in range(centroids.shape[0]):
                    # Calculate the Euclidean distance between the pixel and each centroid
                    dist[i] = np.linalg.norm(pixel - centroids[i])
                # Find the index of the closest centroid
                closest_centroid_index = np.argmin(dist)

                new_centroids[closest_centroid_index]+= pixel
                centroids_number[closest_centroid_index] += 1
        # Avoid division by zero
        for i in range(centroids.shape[0]):
            if centroids_number[i] > 0:
                new_centroids[i] /= centroids_number[i]
            else:
                # If no pixels were assigned to this centroid, keep it unchanged
                new_centroids[i] = centroids[i]
        centroids=new_centroids.astype(np.float32)


    # *** END YOUR CODE ***

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
            # Initialize `dist` vector to keep track of distance to every centroid
            # Loop over all centroids and store distances in `dist`
            # Find closest centroid and update pixel value in `image`
    Height, Width, Channels = image.shape
    for h in range(Height):
        for w in range(Width):
            pixel = image[h, w]
            dist = np.zeros(centroids.shape[0], dtype=np.float32)
            for i in range (centroids.shape[0]):
                dist[i] = np.linalg.norm(pixel - centroids[i])
            closest_centroid_index = np.argmin(dist)
            image[h, w] = centroids[closest_centroid_index]
    image = image.astype(np.float32)  # Ensure the image is in float32 format
    # *** END YOUR CODE ***

    return image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
