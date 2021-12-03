import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvEmbed(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super(ConvEmbed, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LocalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., k1=7, k2=7):
        super(LocalAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.k1, self.k2 = k1, k2

    def forward(self, x):
        B, G, N, C = x.shape
        # h_group, w_group = H // self.k1, W // self.k2
        # total_groups = h_group * w_group
        # x = x.reshape(B, h_group, self.k1, w_group, self.k2, C).permute(0, 3, 1, 2, 4, 5)
        qkv = self.qkv(x).reshape(B, G, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn @ v  # b g h n m
        x = attn.transpose(2, 3).reshape(B, G, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class local_net(nn.Module):
    def __init__(self,
                 dim,
                 heads,
                 depth,
                 k1,
                 k2,
                 ratio=4,
                 attn_drop=0,
                 proj_drop=0):
        super(local_net, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,
                        LocalAttention(dim, num_heads=heads, attn_drop=attn_drop, proj_drop=proj_drop, k1=k1, k2=k2)),
                PreNorm(dim, FeedForward(dim, dim * ratio, dropout=proj_drop)),
            ]))

    def forward(self, x):
        for attn, ffn in self.layers:
            x = attn(x) + x
            x = ffn(x) + x
        return x

class global_net(nn.Module):
    def __init__(self,
                 dim,
                 heads,
                 depth,
                 ratio=4,
                 attn_drop=0,
                 proj_drop=0):
        super(global_net, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,
                        GlobalAttention(dim, num_heads=heads, attn_drop=attn_drop, proj_drop=proj_drop)),
                PreNorm(dim, FeedForward(dim, dim * ratio, dropout=proj_drop)),
            ]))

    def forward(self, x):
        for attn, ffn in self.layers:
            x = attn(x) + x
            x = ffn(x) + x
        return x

class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, cls):
        B, N, C = cls.shape
        # h_group, w_group = H // self.k1, W // self.k2
        # total_groups = h_group * w_group
        # x = x.reshape(B, h_group, self.k1, w_group, self.k2, C).permute(0, 3, 1, 2, 4, 5)
        qkv = self.qkv(cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn @ v
        cls = attn.transpose(1, 2).reshape(B, N, C)
        cls = self.proj(cls)
        cls = self.proj_drop(cls)
        return cls


class Project(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Project, self).__init__()
        self.net = nn.Sequential(
            nn.GELU(),
            # nn.LayerNorm(out_channel),
            nn.Linear(in_channel, out_channel),
        )

    def forward(self, x):
        return self.net(x)


class Seg_tran(nn.Module):
    def __init__(self,
                 num_classes,
                 h,
                 w,
                 k1,
                 k2,
                 dim,
                 depth,
                 heads,
                 dim_head=64,
                 ratio=4,
                 attn_drop=0,
                 proj_drop=0,
                 ):
        super(Seg_tran, self).__init__()
        gdim = dim // 2
        ghead = heads // 2

        self.k1, self.k2 = k1, k2
        self.d1, self.d2 = 4, 4
        self.layers = nn.ModuleList()
        self.h0, self.w0 = h // k1 // self.d1, w // k2 // self.d2
        self.cls = nn.Parameter(torch.randn(1, k1*k2, gdim))
        # self.pos_embed = nn.Parameter(0.1*torch.zeros(1, 100, 225, dim))

        self.x_embed = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, dim, 3, 2, 1),
            nn.GELU(),
            # nn.BatchNorm2d(dim),
        )

        for dp in depth:
            self.layers.append(nn.ModuleList([
                Project(gdim, dim),
                local_net(dim, heads, dp, k1, k2, ratio, attn_drop, proj_drop),
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                Project(dim, gdim),
                global_net(gdim, ghead, 1, ratio, attn_drop, proj_drop)
            ]))

        self.cls_head = nn.Sequential(
            nn.ConvTranspose2d(dim+gdim, dim+gdim, 4, 2, 1),
            nn.BatchNorm2d(dim+gdim),
            nn.GELU(),
            nn.Conv2d(dim+gdim, dim+gdim, 3, 1, 1),
            nn.BatchNorm2d(dim+gdim),
            nn.GELU(),
            # nn.BatchNorm2d(dim+gdim),
            nn.Conv2d(dim+gdim, num_classes, 1, bias=False)
        )


    def forward(self, x):
        b, c, h, w = x.shape

        local_h, local_w = self.k1 * self.d1, self.k2 * self.d2

        total_windows = h // local_h * w // local_w

        # cls = x.reshape(b, c, h // self.k1, self.k1, w // self.k2, self.k2).permute(0, 2, 4, 3, 5, 1)\
        #     .contiguous().view(b, total_windows, -1)

        # x = x.reshape(b, c, h // local_h, self.k1, self.d1, w // local_w, self.k2, self.d2).permute(0, 2, 5, 3, 6, 4, 7,
        #                                                                                             1) \
        #     .contiguous().view(b, total_windows, -1, self.d1 * self.d2 * c)

        x = self.x_embed(x)

        x = rearrange(x, 'b c (h0 h1) (w0 w1) -> b (h0 w0) (h1 w1) c', h0=self.k1, w0=self.k2)

        cls = self.cls.repeat(b, 1, 1)
        for down, ln, dc, up, gn in self.layers:
            cls = down(cls)
            #print(cls.shape, x.shape)
            x = torch.cat((cls.unsqueeze(2), x), 2)
            x = ln(x)
            cls, x = x[:, :, 0], x[:, :, 1:]
            x = rearrange(x, 'b (h0 w0) (h1 w1) d -> b d (h0 h1) (w0 w1)', h0=self.k1, h1=h // local_h)
            x = dc(x)
            x = rearrange(x, 'b d (h0 h1) (w0 w1)-> b (h0 w0) (h1 w1) d ', h0=self.k1, w0=self.k2)
            cls = up(cls)
            cls = gn(cls)
            # cls = gf(cls) + cls
        x = rearrange(x, 'b (h0 w0) (h1 w1) d -> b d (h0 h1) (w0 w1)', h0=self.k1, h1=h // local_h)
        cls = rearrange(cls, 'b (h w) d -> b d h w', h=self.k1)
        cls = F.interpolate(cls, size=(h//4, w//4), mode='bilinear', align_corners=True)
        x = torch.cat((x, cls), 1)
        x = self.cls_head(x)

        return x

if __name__ == '__main__':
    from thop import profile, clever_format
    model = Seg_tran(
        num_classes=6,
        h=256,
        w=256,
        k1=8,
        k2=8,
        dim=128,
        depth=5,
        heads=2,
        dim_head=64,
        ratio=4,
        # attn_drop=0.1,
        # proj_drop=0.1,
    )
    x = torch.rand(1, 3, 256, 256)
    print(model(x).shape)
    macs, params = profile(model, (x,))
    macs, params = clever_format((macs, params))
    print(f'macs:{macs}, params:{params}')

    # print(model(x)[1].shape)


