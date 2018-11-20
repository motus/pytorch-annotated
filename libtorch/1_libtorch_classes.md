
# The origin of PyTorch methods

At the top of high-level libtorch C++ API is the `at::Tensor` class. It provides a comprehensive set
of methods for tensor storage management, data initialization, auto-differentiation, as well as all
basic tensor operations. It is defined at
[ATen/core/Tensor.h](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/Tensor.h)

All actual work is delegated from `Tensor` to the classes `TensorImpl` and `Type`.


Here's how the `Type` hierarchy looks like for CUDA:

![libtorch Type hierarchy](structat_1_1Type__inherit__graph.png)
