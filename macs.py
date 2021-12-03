from model.seg_model3 import Seg_tran
import torch
from thop import profile, clever_format

model = Seg_tran(
        num_classes=6,
        h=512,
        w=512,
        k1=8,
        k2=8,
        dim=256,
        depth=[2, 2, 1],
        heads=8,
        dim_head=32,
        ratio=4,
        attn_drop=0.5,
        proj_drop=0.5,
)

x = torch.rand(1, 3, 512, 512)

output = profile(model, (x,))
macs, params = clever_format(output)
print(macs, params)