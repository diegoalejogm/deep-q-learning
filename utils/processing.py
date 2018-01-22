import PIL.Image
import numpy as np
import torch
from torch.autograd import Variable


# TODO: Process only one image


def phi_map(image_list, as_var=True):
    # Frame Skipping size
    k = len(image_list)

    im_tuple = tuple()
    for i in range(k):
        # Load single image as PIL and convert to Luminance
        im = PIL.Image.fromarray(image_list[i]).convert('L')
        # Resize image
        im = im.resize((84, 84), PIL.Image.ANTIALIAS)
        # Transform to numpy array
        im = np.array(im) / 255.
        # Add processed image to tuple
        im_tuple += (im,)

    # Return tensor of processed images
    arr = tuple_to_numpy(im_tuple)

    # Convert to Variable
    if as_var:
        arr = Variable(torch.from_numpy(arr)).float().cuda()
    return arr


def tuple_to_numpy(im_tuple):
    # Stack tuple of 2D images as 3D np array
    arr = np.dstack(im_tuple)
    # Move depth axis to first index: (height, width, depth) to (depth, height, width)
    arr = np.moveaxis(arr, -1, 0)
    # Make arr 4D by adding dimension at first index
    arr = np.expand_dims(arr, 0)
    return arr
