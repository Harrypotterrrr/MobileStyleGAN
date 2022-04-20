import numpy as np
import tensorflow as tf


def _shape(tf_expr, dim_idx):
    if tf_expr.shape.rank is not None:
        dim = tf_expr.shape[dim_idx]
        if dim is not None:
            return dim
    return tf.shape(tf_expr)[dim_idx]

# reference: https://github.com/NVlabs/stylegan2/blob/master/dnnlib/tflib/ops/upfirdn_2d.py
def upfirdn2d_cpu(x, k, upx, upy, downx, downy, padx0, padx1, pady0, pady1):
    """Slow reference implementation of `upfirdn_2d()` using standard TensorFlow ops."""

    x = tf.convert_to_tensor(x)
    k = np.asarray(k, dtype=np.float32)
    assert x.shape.rank == 4
    inH = x.shape[1]
    inW = x.shape[2]
    minorDim = _shape(x, 3)
    kernelH, kernelW = k.shape
    assert inW >= 1 and inH >= 1
    assert kernelW >= 1 and kernelH >= 1
    assert isinstance(upx, int) and isinstance(upy, int)
    assert isinstance(downx, int) and isinstance(downy, int)
    assert isinstance(padx0, int) and isinstance(padx1, int)
    assert isinstance(pady0, int) and isinstance(pady1, int)

    # Upsample (insert zeros).
    x = tf.reshape(x, [-1, inH, 1, inW, 1, minorDim])
    x = tf.pad(x, [[0, 0], [0, 0], [0, upy - 1], [0, 0], [0, upx - 1], [0, 0]])
    x = tf.reshape(x, [-1, inH * upy, inW * upx, minorDim])

    # Pad (crop if negative).
    x = tf.pad(x, [[0, 0], [max(pady0, 0), max(pady1, 0)], [max(padx0, 0), max(padx1, 0)], [0, 0]])
    x = x[:, max(-pady0, 0) : x.shape[1] - max(-pady1, 0), max(-padx0, 0) : x.shape[2] - max(-padx1, 0), :]

    # Convolve with filter.
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, [-1, 1, inH * upy + pady0 + pady1, inW * upx + padx0 + padx1])
    w = tf.constant(k[::-1, ::-1, np.newaxis, np.newaxis], dtype=x.dtype)
    x = tf.transpose(x, [0, 2, 3, 1])
    # x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID', data_format='NCHW')
    x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID', data_format='NHWC')
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, [-1, minorDim, inH * upy + pady0 + pady1 - kernelH + 1, inW * upx + padx0 + padx1 - kernelW + 1])
    x = tf.transpose(x, [0, 2, 3, 1])
    print(x.shape)

    # Downsample (throw away pixels).
    return x[:, ::downy, ::downx, :]

def _upfirdn2d_cpu(
    x, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):

    _, channel, in_h, in_w = x.shape
    print(x.shape)
    x = tf.reshape(x, [-1, in_h, in_w, 1])
    print(x.shape)

    _, in_h, in_w, minor = x.shape
    kernel_h, kernel_w = kernel.shape

    out = tf.reshape(x, [-1, in_h, 1, in_w, 1, minor])
    # out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = tf.pad(out, [[0,0], [0,0], [0,up_y-1], [0,0], [0,up_x-1], [0,0]])
    out = tf.reshape(out, [-1, in_h * up_y, in_w * up_x, minor])

    # out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = tf.pad(out, [[0, 0], [max(pad_y0, 0), max(pad_y1, 0)], [max(pad_x0, 0), max(pad_x1, 0)], [0, 0]])
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = tf.transpose(out, [0, 3, 1, 2])
    out = tf.reshape(
        out,
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    # w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    w = tf.reverse(kernel, [0, 1])
    w = tf.reshape(w, [1, 1, kernel_h, kernel_w])
    w = tf.transpose(w, [2, 3, 0, 1])
    print(w.shape)
    print(out.shape)
    out = tf.transpose(out, [0, 2, 3, 1]) # transpose to NHWC
    out = tf.nn.conv2d(out, w, strides=1, padding='SAME')
    print(minor * (in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1) * (in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1))
    print(tf.size(out))
    out = tf.reshape(
        out, [
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    ])
    out = tf.transpose(out, [0, 2, 3, 1])
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return tf.reshape(out, [-1, channel, out_h, out_w])

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    # TODO CUDA
    if len(tf.config.list_physical_devices('GPU')) != 0:
        # out = UpFirDn2d.apply(
        #     input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
        # )
        out = None
        raise NotImplemented
    else:
        out = upfirdn2d_cpu(
            input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
        )

    return out
