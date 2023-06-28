# batched-pytorch-ksvd
Library for Batched version of KSVD written in PyTorch that runs on GPU

This library is a batched version of the pytorch KSVD code https://github.com/nel215/ksvd

https://github.com/nel215/ksvd only works with 2d matrix **(A, B)**. This repo efficiently extends it to **(Batch, A, B)**. 

`example_command.sh` contains example of how to run the batched KSVD on GPU using pytorch.

Running `example_command.sh` computes the coefficient & basis of the given weight matrix & sparsity and saves it with `--save_name`.npy extension. 

This library was written as part of work at MSR India.
 
