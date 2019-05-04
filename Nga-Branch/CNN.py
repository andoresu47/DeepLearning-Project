import numpy as np
#from fast_layers import *
from cs231n.im2col import *

class ThreeLayerConvNet(object):
    """
    A Three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, reg=0.0, weight_scale=0.1, 
                 dtype=np.float32, batchnorm=False):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        - batchnorm: if use batch normalization
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.weight_scale = weight_scale
        self.batchnorm = batchnorm
        self.bn_params = {}
        
        self.conv_param = {'stride': 1}
        self.pool_param = {'pool_height': 2, 
                           'pool_width': 2, 
                           'stride': 2}
        
        self.init_sets()
    
    
    def init_sets(self):
        """
        Initialize weights and biases for the K-layer convolutional network
        """
        
        C,H_input,W_input = self.input_dim
        
        # Convolutional layer
        # Xavier initialization sqrt(1/(k x k)) at the first layer and He for the next (after ReLu)
        sig = 1.0/(np.sqrt(C)*self.filter_size)    
        self.params['W1'] = self.weight_scale*sig*np.random.randn(self.num_filters,C,self.filter_size,self.filter_size)
        self.params['b1'] = np.zeros(self.num_filters)
        
        if self.batchnorm:
            bn_param = {'mode': 'train',
                        'running_mean': np.zeros(self.num_filters),
                        'running_var': np.zeros(self.num_filters)}
            self.bn_params['bn_param1'] = bn_param
            self.params['gamma1'] = np.ones(self.num_filters)
            self.params['beta1'] = np.zeros(self.num_filters)

        # He initialize for Hidden affine layer
        # Note that the width and height are preserved after the convolutional layer
        # 2x2 max pool makes the width and height reduce by half
        d_in = self.num_filters*(H_input//2)*(W_input//2)       
        self.params['W2'] = self.weight_scale*np.sqrt(2.0/d_in)*np.random.randn(d_in,self.hidden_dim)  
        self.params['b2'] = np.zeros(self.hidden_dim)

        if self.batchnorm:
            bn_param = {'mode': 'train',
                        'running_mean': np.zeros(self.hidden_dim),
                        'running_var': np.zeros(self.hidden_dim)}
            self.bn_params['bn_param2'] = bn_param
            self.params['gamma2'] = np.ones(self.hidden_dim)
            self.params['beta2'] = np.zeros(self.hidden_dim)
                
        # He Initialize for Output affine layer
        self.params['W3'] = self.weight_scale*np.sqrt(2.0/self.hidden_dim)*np.random.randn(self.hidden_dim,self.num_classes)
        self.params['b3'] = np.zeros(self.num_classes)
        
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)
            

    def loss(self, X, y=None, logit_distill=None, temperature=1.0, alpha=0.5):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        - y=None: for testing
        - logit_distill: distill knowledge from big model, default is not doing distillation
        - temperature: using fot distilling
        - alpha: control parameter for distilling, alpha = 1 corresponds to unlabel training
        """
        X = X.astype(self.dtype)
        
        mode = 'test' if y is None else 'train'

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = self.filter_size
        conv_param = self.conv_param

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = self.pool_param
        
        if self.batchnorm:
            for key, bn_param in self.bn_params.items():
                bn_param['mode'] = mode
            gamma1, beta1 = self.params['gamma1'], self.params['beta1']
            bn_param1 = self.bn_params['bn_param1']
            gamma2, beta2 = self.params['gamma2'], self.params['beta2']
            bn_param2 = self.bn_params['bn_param2']
                
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
        # Forward pass
        conv_param['pad'] = (filter_size - 1) // 2
            
        if self.batchnorm:             
            conv_out, cache_conv = conv_bn_relu_pool_forward(X, W1, b1, gamma1, beta1, conv_param, pool_param, bn_param1)
        else:
            conv_out, cache_conv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
            
        if self.batchnorm:
            affine_out, cache_affine = affine_bn_relu_forward(conv_out, W2, b2, gamma2, beta2, bn_param2)
        else:
            affine_out, cache_affine = affine_relu_forward(conv_out, W2, b2)
            
        scores, cache_scores = affine_forward(affine_out, W3, b3)

        if y is None:
            return scores

        # Backward pass
        loss, grads = 0, {}
        
        # Computing of the loss
        loss, dscores = softmax_loss(scores, y)
        if logit_distill is not None:
            # Compute loss with introducing soft target from big model
            loss_soft, dscores_soft = softmax_distill(scores, logit_distill, temperature)
            loss = (1-alpha)*loss + alpha*temperature**2*loss_soft
            dscores = (1-alpha)*dscores + alpha*temperature**2*dscores_soft
            
        loss += 0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))
                
        dout_affine, dW3, db3 = affine_backward(dscores, cache_scores)
        
        if self.batchnorm:
            dout_conv, dW2, db2, dgamma2, dbeta2 = affine_bn_relu_backward(dout_affine, cache_affine)
        else:
            dout_conv, dW2, db2 = affine_relu_backward(dout_affine, cache_affine)

        if self.batchnorm:
            dx, dW1, db1, dgamma1, dbeta1 = conv_bn_relu_pool_backward(dout_conv, cache_conv)
        else:
            dx, dW1, db1 = conv_relu_pool_backward(dout_conv, cache_conv)
         
        
        grads['W1']= dW1 + self.reg*W1
        grads['W2']= dW2 + self.reg*W2
        grads['W3']= dW3 + self.reg*W3
        grads['b1']= db1
        grads['b2']= db2
        grads['b3']= db3
        if self.batchnorm:
            grads['gamma1'] = dgamma1
            grads['gamma2'] = dgamma2
            grads['beta1'] = dbeta1
            grads['beta2'] = dbeta2
        

        return loss, grads



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """   
    N = x.shape[0]
    x_reshape = x.reshape(N,-1)
    
    out = x_reshape.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
   
    N = x.shape[0]
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(N,-1).T.dot(dout)
    db = np.sum(dout,axis=0)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """

    out = np.maximum(0,x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache
    
    mask = x>0
    dx = dout*mask

    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    if mode == 'train':       
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_norm = (x-sample_mean)/np.sqrt(sample_var+eps)  # avoid divide by 0
        out = x_norm*gamma + beta
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        cache = [x, sample_mean, sample_var, eps, gamma]
    elif mode == 'test':
        x_norm = (x-running_mean)/np.sqrt(running_var+eps)  # avoid divide by 0
        out = x_norm*gamma + beta
        
        cache = [x, running_mean, running_var, eps, gamma]
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    
    x, mean, var, eps, gamma = cache
    N = x.shape[0]
    std = np.sqrt(var+eps)
    
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(dout*(x-mean)/std, axis=0)
    
    x_hat = (x-mean)/std
    dxhat = dout*gamma
    
    dx = (N*dxhat - x_hat*np.sum(dxhat*x_hat, axis=0) - np.sum(dxhat, axis=0))/(N*std)

    return dx, dgamma, dbeta


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = 1 + (H + 2 * pad - HH) // stride  # Use `//` for python3
    W_out = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, H_out, W_out))

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)

    for n in range(N):
        for f in range(F):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    out[n, f, h_out, w_out] = np.sum(
                        x_pad[n, :, h_out*stride:h_out*stride+HH, w_out*stride:w_out*stride+WW]*w[f, :]) + b[f]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)

    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            db[f] += np.sum(dout[n, f])
            for h_out in range(H_out):
                for w_out in range(W_out):
                    dw[f] += x_pad[n, :, h_out*stride:h_out*stride+HH, w_out*stride:w_out*stride+WW] * \
                    dout[n, f, h_out, w_out]
                    dx_pad[n, :, h_out*stride:h_out*stride+HH, w_out*stride:w_out*stride+WW] += w[f] * \
                    dout[n, f, h_out, w_out]

    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]

    return dx, dw, db


def conv_forward_im2col(x, w, b, conv_param):
    """
    A fast implementation of the forward pass for a convolutional layer
    based on im2col and col2im.
    """
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    # Create output
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    x_cols = im2col_indices(x, filter_height, filter_width, pad, stride)
    #x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((num_filters, -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache



def conv_backward_im2col(dout, cache):
    """
    A fast implementation of the backward pass for a convolutional layer
    based on im2col and col2im.
    """
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    db = np.sum(dout, axis=(0, 2, 3))

    num_filters, _, filter_height, filter_width = w.shape
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
    dx = col2im_indices(dx_cols, x.shape, filter_height, filter_width, pad, stride)
    #dx = col2im_cython(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
    #                   filter_height, filter_width, pad, stride)

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None

    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
        for h_out in range(H_out):
            for w_out in range(W_out):
                out[n, :, h_out, w_out] = np.max(x[n, :, h_out*stride:h_out*stride+pool_height,
                    w_out*stride:w_out*stride+pool_width], axis=(-1, -2)) # axis can also be (1, 2)

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None

    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    # Find the index (row, col) of the max value
                    # Ref: examples of https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.argmax.html
                    ind = np.unravel_index(np.argmax(x[n, c, h*stride:h*stride+pool_height,
                        w*stride:w*stride+pool_width], axis=None), (pool_height, pool_width))
    
                    dx[n, c, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width][ind] = \
                    dout[n, c, h, w]
    return dx



def max_pool_forward_im2col(x, pool_param):
    """
    An implementation of the forward pass for max pooling based on im2col.

    This isn't much faster than the naive version, so it should be avoided if
    possible.
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    assert (H - pool_height) % stride == 0, 'Invalid height'
    assert (W - pool_width) % stride == 0, 'Invalid width'

    out_height = (H - pool_height) // stride + 1
    out_width = (W - pool_width) // stride + 1

    x_split = x.reshape(N * C, 1, H, W)
    x_cols = im2col_indices(x_split, pool_height, pool_width, padding=0, stride=stride)
    x_cols_argmax = np.argmax(x_cols, axis=0)
    x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
    out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

    cache = (x, x_cols, x_cols_argmax, pool_param)
    return out, cache


def max_pool_backward_im2col(dout, cache):
    """
    An implementation of the backward pass for max pooling based on im2col.

    This isn't much faster than the naive version, so it should be avoided if
    possible.
    """
    x, x_cols, x_cols_argmax, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
    dx_cols = np.zeros_like(x_cols)
    dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
    dx = col2im_indices(dx_cols, (N * C, 1, H, W), pool_height, pool_width,
                padding=0, stride=stride)
    dx = dx.reshape(x.shape)

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """

    N, C, H, W = x.shape

    # Reshape x to N*H*W * C to call batch normalization
    x_new = np.reshape(np.transpose(x, (0, 2, 3, 1)), (-1, C))

    out, cache = batchnorm_forward(x_new, gamma, beta, bn_param)
    
    # Reshape out to (N, C, H, W)
    out = np.transpose(np.reshape(out, (N, H, W, C)), (0, 3, 1, 2))

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    N, C, H, W = dout.shape

    # Reshape dout to N*H*W * C to call batch normalization
    dout_new = np.reshape(np.transpose(dout, (0, 2, 3, 1)), (-1, C))

    dx, dgamma, dbeta = batchnorm_backward_alt(dout_new, cache)

    # Reshape dx to (N, C, H, W)
    dx = np.transpose(np.reshape(dx, (N, H, W, C)), (0, 3, 1, 2))

    return dx, dgamma, dbeta


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta : Weight for the batch norm regularization
    - bn_params : Contain variable use to batch norml, running_mean and var
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """

    a, fc_cache = affine_forward(x, w, b)
    an, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (fc_cache, bn_cache, relu_cache)

    return out, cache


def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache

    da = relu_backward(dout, relu_cache)
    dan, dgamma, dbeta = batchnorm_backward_alt(da, bn_cache)
    dx, dw, db = affine_backward(dan, fc_cache)

    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_im2col(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_im2col(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_im2col(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_im2col(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_pool_forward(x, w, b, gamma, beta, conv_param, pool_param, bn_param):
    """
    Convenience layer that performs a convolution, spatial
    batchnorm, a ReLU, and a pool.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_im2col(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    s, relu_cache = relu_forward(an)
    out, pool_cache = max_pool_forward_im2col(s, pool_param)

    cache = (conv_cache, bn_cache, relu_cache, pool_cache)

    return out, cache


def conv_bn_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, bn_cache, relu_cache, pool_cache = cache

    ds = max_pool_backward_im2col(dout, pool_cache)
    dan = relu_backward(ds, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_im2col(da, conv_cache)

    return dx, dw, db, dgamma, dbeta


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)  # use shift to avoid x is too large to do exponential
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

    
def softmax_distill(x, logit, temperature=1.0):
    x /= temperature
    logit /= temperature
    shifted_logits = x - np.max(x, axis=1, keepdims=True)  # use shift to avoid x is too large to do exponential
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    
    shifted_logits_soft = logit - np.max(logit, axis=1, keepdims=True)
    Z_soft = np.sum(np.exp(shifted_logits_soft), axis=1, keepdims=True)
    log_probs_soft = shifted_logits_soft - np.log(Z_soft)
    probs_soft = np.exp(log_probs_soft)
    
    N = x.shape[0]
    loss = -np.sum(probs_soft*log_probs) / N
    
    dx = (probs - probs_soft)/temperature/N
    return loss, dx
