# segmentation

Tensorflow implementaton of ENet (https://arxiv.org/pdf/1606.02147.pdf) based on the official Torch implementation (https://github.com/e-lab/ENet-training), the Keras implementation by PavlosMelissinos (https://github.com/PavlosMelissinos/enet-keras) and the Tensorflow implementaton by kwotsin (https://github.com/kwotsin/TensorFlow-ENet), trained on the Cityscapes dataset (https://www.cityscapes-dataset.com/).


****

First, I got the error "No gradient defined for operation 'MaxPoolWithArgmax_1' (op type: MaxPoolWithArgmax)". To fix this, I had to add the following code:  
@ops.RegisterGradient("MaxPoolWithArgmax")  
def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):  
  return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0], grad, op.outputs[1], op.get_attr("ksize"), op.get_attr("strides"), padding=op.get_attr("padding"))     
to the file /usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/nn_grad.py

                                              
****

number of nonroads in pretrain_train before balancing: 100448
number of roads in pretrain_train before balancing: 34280
number of nonroads in pretrain_train after balancing: 34280
number of roads in pretrain_train after balancing: 34280
number of pretrain_train imgs: 68560

number of nonroads in pretrain_val before balancing: 16643
number of roads in pretrain_val before balancing: 6110
number of nonroads in pretrain_val after balancing: 2000
number of roads in pretrain_val after balancing: 2000
number of pretrain_val imgs: 4000

number of epochs of pretraining: XXXXXXXXXXXX

plot of train/val loss in pretraining: XXXXXXXXXXXX



*****

zip -r demo_0.zip cityscapes/leftImg8bit/demoVideo/stuttgart_00

scp amrkri@kent.s2.chalmers.se:/mnt/data/demo_0.zip demo_0.zip

