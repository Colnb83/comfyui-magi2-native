# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""``UnfoldNd`` with a fallback for environments without the ``unfoldNd`` package.

Default: re-export ``unfoldNd.UnfoldNd`` (unchanged behaviour).

If the package is not installed, a stand-in covers the only case the data
proxies use: 3-D patches with ``stride == kernel_size``, no padding and no
dilation. That is a reshape; the output layout is the one ``torch.nn.Unfold``
defines: ``(N, C * kt * kh * kw, L)``, column index ordered (c, kt, kh, kw),
patches ordered (t, h, w). Trailing input that does not fill a patch is
dropped, as unfold does.
"""

import torch
import torch.nn as nn

try:
    from unfoldNd import UnfoldNd  # noqa: F401
except ImportError:

    class UnfoldNd(nn.Module):
        def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
            super().__init__()
            kernel_size = tuple(int(k) for k in kernel_size)
            stride = tuple(int(s) for s in stride) if not isinstance(stride, int) else (stride,) * len(kernel_size)
            if len(kernel_size) != 3 or stride != kernel_size or dilation != 1 or padding != 0:
                raise NotImplementedError(
                    "the built-in UnfoldNd stand-in only supports 3-D patches with "
                    "stride == kernel_size, padding 0, dilation 1; install unfoldNd"
                )
            self.kernel_size = kernel_size

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            kt, kh, kw = self.kernel_size
            n, c, t, h, w = x.shape
            nt, nh, nw = t // kt, h // kh, w // kw
            x = x[:, :, : nt * kt, : nh * kh, : nw * kw]
            x = x.reshape(n, c, nt, kt, nh, kh, nw, kw)
            x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
            return x.reshape(n, c * kt * kh * kw, nt * nh * nw)


__all__ = ["UnfoldNd"]
