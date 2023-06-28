import argparse
from torch_ksvd_2d import TorchApproximateKSVD
import torch
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser(description="Generate heatmap")
    parser.add_argument(
        "--sp",
        type=int,
        default=0,
        help="Example index",
    )

    parser.add_argument(
        "--basis",
        type=int,
        default=0,
        help="Example index",
    )

    parser.add_argument(
        "--d_path",
        type=str,
        default=None,
        help="Example index",
    )

    parser.add_argument(
        "--c_path",
        type=str,
        default=None,
        help="Example index",
    )

    parser.add_argument(
        "--w_path",
        type=str,
        default=0,
        help="Example index",
        required=True
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default="save",
        required=True
    )

    return parser.parse_args()

args = parse_args()




device = "cuda:0"

W = torch.from_numpy(np.load(args.w_path)).to(torch.float32).to(device)

print(W.shape)

sp,head_size = args.sp, W.shape[2]
C, D = None, None

c_path, d_path = args.c_path, args.d_path

if (c_path is not None):
    C = torch.from_numpy(np.load(c_path)).to(device)
if (d_path is not None):
    D = torch.from_numpy(np.load(d_path)).to(device)


num_comp = args.basis

ksvd = TorchApproximateKSVD(
        num_basis=num_comp,
        max_iter=4,
        coef_sparsity=sp,
        name=args.save_name,
        logger=None,
        device=device,
        shouldQ=False,
        head_size=head_size,
    )

ksvd.fit_external(
    X = W,
    D_init = D,
    coefficients_init=C
)
