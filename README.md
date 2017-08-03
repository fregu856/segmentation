# segmentation

Tensorflow implementaton of ENet (https://arxiv.org/pdf/1606.02147.pdf) based on the official Torch implementation (https://github.com/e-lab/ENet-training), the Keras implementation by PavlosMelissinos (https://github.com/PavlosMelissinos/enet-keras) and the Tensorflow implementaton by kwotsin (https://github.com/kwotsin/TensorFlow-ENet), trained on the Cityscapes dataset (https://www.cityscapes-dataset.com/).


****

First, I got the error "No gradient defined for operation 'MaxPoolWithArgmax_1' (op type: MaxPoolWithArgmax)". To fix this, I had to add the following code:  
@ops.RegisterGradient("MaxPoolWithArgmax")  
def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):  
  return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],  
                                               grad,  
                                               op.outputs[1],  
                                               op.get_attr("ksize"),  
                                               op.get_attr("strides"),  
                                               padding=op.get_attr("padding"))    
to the file /usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/nn_grad.py

                                              
