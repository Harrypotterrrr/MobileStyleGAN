import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import layers, initializers


def _SFB2D(low, highs, g0_row, g1_row, g0_col, g1_col, mode):
    mode = int_to_mode(mode)

    lh, hl, hh = torch.unbind(highs, dim=2)
    lo = sfb1d(low, lh, g0_col, g1_col, mode=mode, dim=2)
    hi = sfb1d(hl, hh, g0_col, g1_col, mode=mode, dim=2)
    y = sfb1d(lo, hi, g0_row, g1_row, mode=mode, dim=3)

    return y

# TODO
class DWTInverse(keras.Model):
    """ Performs a 2d DWT Inverse reconstruction of an image
    Args:
        wave (str or pywt.Wavelet): Which wavelet to use
        C: deprecated, will be removed in future
    """
    def __init__(self, wave='db1', mode='zero', trace_model=False):
        super(DWTInverse, self).__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]
        # Prepare the filters
        filts = prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_col', filts[0])
        self.register_buffer('g1_col', filts[1])
        self.register_buffer('g0_row', filts[2])
        self.register_buffer('g1_row', filts[3])
        self.mode = mode
        self.trace_model = trace_model

    def call(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward
        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        mode = mode_to_int(self.mode)

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                                ll.shape[-1], device=ll.device)

            # 'Unpad' added dimensions
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]
            if not self.trace_model:
                ll = SFB2D.apply(ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row, mode)
            else:
                ll = _SFB2D(ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row, mode)
        return ll

class IDWTUpsample(keras.Model):

    def __init__(
        self,
        channels_in,
        style_dim,
    ):
        super(IDWTUpsample, self).__init__()
        self.channels = channels_in // 4
        assert self.channels * 4 == channels_in
        # upsample
        self.idwt = DWTInverse(mode='zero', wave='db1')
        # modulation
        self.modulation = layers.Dense(
            channels_in,
            input_shape=style_dim,
            bias_initializer=initializers.Constant(1.0),
        )
        self.modulation.bias.data.fill_(1.0)

    def call(self, x, style):
        b, _, h, w = x.size()
        x = self.modulation(style).reshape(b, -1, 1, 1) * x
        low = x[:, :self.channels]
        high = x[:, self.channels:]
        high = high.reshape(b, self.channels, 3, h, w)
        x = self.idwt((low, [high]))
        return x

# Reference: https://github.com/SarderLab/tfWavelet
def cdf97():
    Lo_D = np.array([0.026748757411, -0.016864118443, -0.078223266529, 0.266864118443,\
                            0.602949018236, 0.266864118443, -0.078223266529,-0.016864118443, \
                            0.026748757411])
    Hi_D = np.array([0.091271763114, -0.057543526229,-0.591271763114,1.11508705,\
                            -0.591271763114,-0.057543526229,0.091271763114,0,0])
    Lo_R = np.array([-0.091271763114,-0.057543526229,0.591271763114,1.11508705,\
                             0.591271763114 ,-0.057543526229,-0.091271763114,0,0])
    Hi_R = np.array([0.026748757411,0.016864118443,-0.078223266529,-0.266864118443,\
                             0.602949018236,-0.266864118443,-0.078223266529,0.016864118443,\
                             0.026748757411])
    cdf97 = {
            'Lo_D':Lo_D,
            'Hi_D':Hi_D,
            'Lo_R':Lo_R,
            'Hi_R':Hi_R
            }
    return cdf97


### Create an DWT object 2d discrete wavelet transform
class tfWavelet(object):
    """
    class that provides a 2D DWT implimentation in TensorFlow.
    uses the Mallat Algorithm for 2d decomposition
    """

    def __init__(self, wavelet=cdf97()):
        # calculate the pad length from filter length
        self.tap_length = len(wavelet['Lo_D'])
        self.pad_length = self.tap_length/2

        ### parse wavelet taps
        # low pass tap for decomposition
        self.Lo_D = tf.convert_to_tensor(wavelet['Lo_D'],dtype=tf.float32)
        # High pass tap for decomposition
        self.Hi_D = tf.convert_to_tensor(wavelet['Hi_D'],dtype=tf.float32)
        # low pass tap for reconstruction
        self.Lo_R = tf.convert_to_tensor(wavelet['Lo_R'],dtype=tf.float32)
        # High pass tap for reconstruction
        self.Hi_R = tf.convert_to_tensor(wavelet['Hi_R'],dtype=tf.float32)

        # capture coeff shape for multi_DWT
        self.coeff_shapes = []


    def multi_DWT(self, data):
        """
        multi level 2D DWT decomposition
        INPUT DATA FORMAT:
        [batch x dim1 x dim2 x channels]
        RETURNS:
        approx: [batch, dim1, dim2, channels]
        detail: [batch, dim1, dim2, [cH(channels 1 to n),cV,cD], level]
        NOTE:
        to make the detail tensor constant size, the higher level
        detail coeffs are resized ()to size of the first level
        """

        # track shape of coeffs
        batch_num, dim1_num, dim2_num, channel_num = data.get_shape().as_list()
        self.coeff_shapes = []

        # first DWT decomposition
        approx, detail = self.DWT(data)
        detail = tf.concat(tf.split(detail,3,axis=-1), axis=-2)

        # track shape
        _, out_dim1, out_dim2, _, _ = detail.get_shape().as_list()
        self.coeff_shapes.insert(0,[out_dim1,out_dim2])

        # further recursive DWT decomposition
        while min(dim1_num, dim2_num) >= (2*self.tap_length-self.pad_length)*2 :
            approx, detail_ = self.DWT(approx)
            detail_ = tf.squeeze(tf.concat(tf.split(detail_,3,axis=-1), axis=-2))

            # track shape
            _, dim1_num, dim2_num, _ = approx.get_shape().as_list()
            self.coeff_shapes.insert(0,[dim1_num,dim2_num])

            # upscale
            detail_ = tf.image.resize_bicubic(detail_, [out_dim1,out_dim2], align_corners=True)

            detail_ = tf.expand_dims(detail_, axis=-1)
            detail = tf.concat([detail_, detail], axis=-1)

        return approx, detail


    def multi_iDWT(self, approx, detail):
        """
        multi level 2D DWT reconstruction
        INPUT DATA FORMAT:
        approx: [batch, dim1, dim2, channels]
        detail: [batch, dim1, dim2, [cH(channels 1 to n),cV,cD], level]
        RETURNS:
        [batch x dim1 x dim2 x channels]
        """

        batch_num, dim1_num, dim2_num, channel_num, levels = detail.get_shape().as_list()

        # recursive iDWT reconstruction
        for i,coeff_shape in enumerate(self.coeff_shapes):
            # get detail level
            detail_ = detail[:,:,:,:,i]

            # downscale
            detail_ = tf.image.resize_bicubic(detail_, coeff_shape, align_corners=True)

            detail_ = tf.stack(tf.split(detail_,3,axis=-1),axis=-1)
            approx = self.iDWT(approx, detail_)

        return approx


    def DWT(self, data):
        """
        single level 2D DWT decomposition
        INPUT DATA FORMAT:
        [batch x dim1 x dim2 x channels]
        RETURNS:
        approx: [batch, dim1, dim2, channels]
        detail: [batch, dim1, dim2, channels, [cH,cV,cD]]
        """

        # pad data
        paddings = [[0,0],[self.pad_length,self.pad_length], \
                        [self.pad_length,self.pad_length],[0,0]]
        data = tf.pad(data, paddings, 'SYMMETRIC')

        # horizontal decomposition
        data_Lo = self.conv_axis_and_downsample(data, 1, self.Lo_D)
        data_Hi = self.conv_axis_and_downsample(data, 1, self.Hi_D)

        # vertical decomposition
        cA = self.conv_axis_and_downsample(data_Lo, 0, self.Lo_D)
        cH = self.conv_axis_and_downsample(data_Lo, 0, self.Hi_D)
        cV = self.conv_axis_and_downsample(data_Hi, 0, self.Lo_D)
        cD = self.conv_axis_and_downsample(data_Hi, 0, self.Hi_D)

        # pack coeffs
        approx = cA
        detail = tf.stack([cH,cV,cD], axis=-1)
        return approx, detail


    def iDWT(self, approx, detail):
        """
        single level 2D inverse DWT reconstruction
        INPUT DATA FORMAT:
        approx: [batch, dim1, dim2, channels]
        detail: [batch, dim1, dim2, channels, [cH,cV,cD]]
        RETURNS:
        [batch x dim1 x dim2 x channels]
        """
        # check shapes match
        _,dim1,dim2,_,_ = detail.get_shape().as_list()
        _,dim1_a,dim2_a,_ = approx.get_shape().as_list()
        if dim1 != dim1_a:
            approx = approx[:,1:1+dim1,:,:]
        if dim2 != dim2_a:
            approx = approx[:,:,1:1+dim2,:]

        # unpack coeffs
        cA = approx
        cH,cV,cD = tf.unstack(detail, axis=-1)

        # vertical decomposition
        data_Lo = self.upsample_and_conv_axis(cA, 0, self.Lo_R) + \
                        self.upsample_and_conv_axis(cH, 0, self.Hi_R)
        data_Hi = self.upsample_and_conv_axis(cV, 0, self.Lo_R) + \
                        self.upsample_and_conv_axis(cD, 0, self.Hi_R)

        # horizontal decomposition
        data = self.upsample_and_conv_axis(data_Lo, 1, self.Lo_R) + \
                    self.upsample_and_conv_axis(data_Hi, 1, self.Hi_R)

        # unpad data
        data = data[:,self.pad_length:-self.pad_length,self.pad_length:-self.pad_length,:]
        return data


    def conv_axis_and_downsample(self, data, axis, filter):
        """
        1D convolution followed by decimation along one axis
        implemented as a 2D separable convolution and matmul
        INPUT DATA FORMAT:
        [batch x dim1 x dim2 x channels]
        """

        assert axis==0 or axis==1, "axis must be 0 or 1"
        batch_num, dim1_num, dim2_num, channel_num = data.get_shape().as_list()

        if axis == 0:
            # pad dim1
            data_pad = tf.pad(data, [[0,0],[self.pad_length,self.pad_length],[0,0], \
                                [0,0]], 'REFLECT')
            # make filter 2d
            filter = tf.expand_dims(filter,axis=1)

        else:
            # pad dim 2
            data_pad = tf.pad(data, [[0,0],[0,0],[self.pad_length,self.pad_length], \
                                [0,0]], 'REFLECT')
            # make filter 2d
            filter = tf.expand_dims(filter,axis=0)

        ### make 4D depthwise for separable conv
        d_filt = tf.stack([filter for i in range(channel_num)], axis=-1)
        d_filt = tf.expand_dims(d_filt,axis=-1)

        ### convolution
        data_conv = tf.nn.depthwise_conv2d(data_pad,d_filt,[1,1,1,1],'VALID')

        ### downsample via matrix multiplication
        if axis == 0:
            size = dim1_num
            downsample_mat = tf.eye(size)
            downsample_mat = tf.strided_slice(downsample_mat,[0,0],[size,size],[1,2])
            downsample_mat = tf.reverse(downsample_mat,[0])
            downsample_mat = tf.reverse(downsample_mat,[1])

            def matmul0(input):
                input_T = tf.transpose(input) # channels first
                output_T = tf.map_fn(lambda i: tf.linalg.matmul(i, downsample_mat), input_T, parallel_iterations=40)
                return tf.transpose(output_T)
            data_downsampled = tf.map_fn(lambda i: matmul0(i), data_conv, parallel_iterations=40)

        else:
            size = dim2_num
            downsample_mat = tf.eye(size)
            downsample_mat = tf.strided_slice(downsample_mat,[0,0],[size,size],[2,1])
            downsample_mat = tf.reverse(downsample_mat,[0])
            downsample_mat = tf.reverse(downsample_mat,[1])

            def matmul1(input):
                input_T = tf.transpose(input) # channels first
                output_T = tf.map_fn(lambda i: tf.linalg.matmul(downsample_mat, i), input_T, parallel_iterations=40)
                return tf.transpose(output_T)
            data_downsampled = tf.map_fn(lambda i: matmul1(i), data_conv, parallel_iterations=40)

        return data_downsampled


    def upsample_and_conv_axis(self, data, axis, filter):
        """
        1D convolution followed by expansion by zero striping along one axis
        INPUT DATA FORMAT:
        [batch x dim1 x dim2 x channels]
        """

        batch_num, dim1_num, dim2_num, channel_num = data.get_shape().as_list()
        assert axis==0 or axis==1, "axis must be 0 or 1"

        ### upsample via matrix multiplication
        if axis == 0:
            size = dim1_num*2
            upsample_mat = tf.eye(size)
            upsample_mat = tf.strided_slice(upsample_mat,[0,0],[size,size],[2,1])

            def matmul0(input):
                input_T = tf.transpose(input) # channels first
                output_T = tf.map_fn(lambda i: tf.linalg.matmul(i, upsample_mat), input_T, parallel_iterations=40)
                return tf.transpose(output_T)
            data_upsampled = tf.map_fn(lambda i: matmul0(i), data, parallel_iterations=40)

        else:
            size = dim2_num*2
            upsample_mat = tf.eye(size)
            upsample_mat = tf.strided_slice(upsample_mat,[0,0],[size,size],[1,2])

            def matmul1(input):
                input_T = tf.transpose(input) # channels first
                output_T = tf.map_fn(lambda i: tf.linalg.matmul(upsample_mat, i), input_T, parallel_iterations=40)
                return tf.transpose(output_T)
            data_upsampled = tf.map_fn(lambda i: matmul1(i), data, parallel_iterations=40)

        ### pad and make filters 2D
        if axis == 0:
            # pad data
            data_pad = tf.pad(data_upsampled, [[0,0],[self.pad_length,self.pad_length],[0,0], \
                                        [0,0]], 'REFLECT')
            # make filter 2d
            filter = tf.expand_dims(filter,axis=1)
        else:
            # pad data
            data_pad = tf.pad(data_upsampled, [[0,0],[0,0],[self.pad_length,self.pad_length], \
                                        [0,0]], 'REFLECT')
            # make filter 2d
            filter = tf.expand_dims(filter,axis=0)

        ### make 4D depthwise filters for separable conv
        d_filt = tf.stack([filter for i in range(channel_num)], axis=-1)
        d_filt = tf.expand_dims(d_filt,axis=-1)

        ### convolution
        data_conv = tf.nn.depthwise_conv2d(data_pad,d_filt,[1,1,1,1],'VALID')
        return data_conv