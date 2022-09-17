# Returns output of the height, width and output size based on input shape,
# mask shape, number of masks, stride and padding. When defining shape for
# input or mask specify it as (height,width, n_channels)
def calc_output_dim(input_shape, mask_shape, n_masks, stride, padding):
    output_h = int(calc_dim(input_shape[0], mask_shape[0], padding, stride))
    output_w = int(calc_dim(input_shape[1], mask_shape[1], padding, stride))

    return output_h, output_w, output_h * output_w * n_masks


# Returns output size for specific dimension, like height or width in case of
# 2D arrays, based on input dimension, mask dimension, padding and stride
def calc_dim(input_dim, mask_dim, padding, stride):
    return 1 + ((input_dim - mask_dim + 2 * padding) / stride)

input_shape = (600, 500, 25)
mask_shape = (11, 11, 3)
n_masks = 20
stride = 2
padding = 1
#print('calculate output dimension')
out_h, out_w, out_size = calc_output_dim(input_shape, mask_shape, n_masks,
                                         stride, padding)
#print('output shape is: ' + str(out_h) + 'x' + str(out_w) + 'x' +
#      str(n_masks) + '=' + str(out_size))

#print('convolution with 40 masks of size 5x5x3 with stride=1, padding=0')
input_shape = (200, 200, 3)
mask_shape = (5, 5, 3)
n_masks = 40
stride = 1
padding = 0
out_h, out_w, out_size = calc_output_dim(input_shape, mask_shape, n_masks,
                                         stride, padding)
#print('output shape is: ' + str(out_h) + 'x' + str(out_w) + 'x' +
#      str(n_masks) + '=' + str(out_size))

#print('pooling with 2x2 pooling regions stride=2')
input_shape = (out_h, out_w, 3)
mask_shape = (2, 2, 3)
stride = 2

out_h, out_w, out_size = calc_output_dim(input_shape, mask_shape, n_masks,
                                         stride, padding)
#print('output shape is: ' + str(out_h) + 'x' + str(out_w) + 'x' +
#      str(n_masks) + '=' + str(out_size))

#print('convolution with 80 masks of size 4x4 with stride=2, padding=1')
input_shape = (out_h, out_w, 3)
mask_shape = (4, 4, 3)
n_masks = 80
padding = 1

out_h, out_w, out_size = calc_output_dim(input_shape, mask_shape, n_masks,
                                         stride, padding)
#print('output shape is: ' + str(out_h) + 'x' + str(out_w) + 'x' +
#      str(n_masks) + '=' + str(out_size))

#print('1x1 convolution with 20 masks')
input_shape = (out_h, out_w, 3)
mask_shape = (1, 1, 3)
n_masks = 20
# this was not clearly stated in description, but from solution looks
# like they meant to set padding and stride to default values
stride = 1
padding = 0

out_h, out_w, out_size = calc_output_dim(input_shape, mask_shape, n_masks,
                                         stride, padding)
#print('output shape is: ' + str(out_h) + 'x' + str(out_w) + 'x' +
#      str(n_masks) + '=' + str(out_size))

