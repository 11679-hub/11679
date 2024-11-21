import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import torch_dct as dct
from einops import rearrange
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
from timm.layers.helpers import to_2tuple
import numpy as np
import math

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class FEFM(nn.Module):
    def __init__(self, dim, bias,depth):
        super(FEFM, self).__init__()
        self.num_heads=dim//16
        #print(depth)
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.to_hidden = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.to_hidden_nir = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.to_hidden_dw_nir = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim , dim , kernel_size=3, stride=1, padding=1, groups=dim , bias=bias)
        self.special=nn.Conv2d(dim , dim , kernel_size=3, stride=1, padding=1, groups=dim , bias=bias)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim , dim, kernel_size=1, bias=bias)
        self.project_middle = nn.Conv2d(dim , dim, kernel_size=1, bias=bias)
        self.pool1= nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False))
        self.pool2= nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False))
        self.norm = LayerNorm(dim , LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x,nir):
        b,c,h,w=x.shape
        hidden = self.to_hidden(x)
        nir_hidden=self.to_hidden_nir(nir)
        q=self.to_hidden_dw(hidden)
        k, v = self.to_hidden_dw_nir(nir_hidden).chunk(2, dim=1)
        #print(q.shape,k.shape)
        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        
        q_fft = dct.dct_2d(q_patch.float())
        k_fft = dct.dct_2d(k_patch.float())
        #print(q_fft.shape)
        out1 = q_fft * k_fft
        #print(out1.shape)
        q_fft = rearrange(q_fft, 'b (head c) h w patch1 patch2-> b head c (h w patch1 patch2)', head=self.num_heads)
        k_fft = rearrange(k_fft, 'b (head c) h w patch1 patch2-> b head c (h w patch1 patch2)', head=self.num_heads)
        out1 = rearrange(out1, 'b (head c) h w patch1 patch2-> b head c (h w patch1 patch2)', head=self.num_heads)
        q_fft = torch.nn.functional.normalize(q_fft, dim=-1)
        k_fft = torch.nn.functional.normalize(k_fft, dim=-1)
        attn = (q_fft @ k_fft.transpose(-2, -1)) * self.temperature
        #print(attn.shape)
        # attn=attn.real
        #print(attn.shape)
        attn = attn.softmax(dim=-1)
        out = (attn @ out1)
        #print(out.shape)
        out = rearrange(out, 'b head c (h w patch1 patch2) -> b (head c) h w patch1 patch2', head=self.num_heads, h=h//self.patch_size, w=w//self.patch_size,patch1=self.patch_size,
                            patch2=self.patch_size)

        out = dct.idct_2d(out)
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                patch2=self.patch_size)
        out=self.project_middle(out)
        # out = self.norm(out)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        #print(lambda_full)

        output = self.pool1(q * out)+self.pool2(v-lambda_full*(v*out))
        output = self.project_out(output)



        return output
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
##########################################################################
class DropPath(nn.Dropout):
    def forward(self, inputs):
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        mask = torch.ones(shape).cuda()
        if self.training:
            mask = F.dropout(mask, self.p, training=True)
            return inputs * mask
        else:
            return inputs
class TransformerBlock(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction,act,bias, LayerNorm_type='WithBias',depth=None):
        super(TransformerBlock, self).__init__()
        self.dim=n_feat
        ffn_expand_dim = int(n_feat * 4)
        self.ffn = DFFN(n_feat, hidden_features=ffn_expand_dim)
        drop_path=0.1
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.ffn=FeedForward(n_feat, 2.66, bias)
        self.norm1 = LayerNorm(n_feat, LayerNorm_type)
        self.norm2 = LayerNorm(n_feat, LayerNorm_type)
        self.attn = FEFM(n_feat, bias,depth)
        
        self.norm3 = LayerNorm(n_feat, LayerNorm_type)
        #self.ffn = DFFN(dim, ffn_expansion_factor, bias)
        self.mix=FDSM(n_feat)
    def forward(self, img):
        #(img.shape)
        #print(self.dim)
        x, y = torch.split(img, split_size_or_sections=self.dim,dim=1)
        
        x_mix,y_mix=self.mix(self.norm1(x),self.norm2(y))
        mid = x + self.drop_path(self.attn(x_mix,y_mix))
        x=self.drop_path(self.ffn(self.norm3(mid)))+mid
        out=torch.cat((x,y),dim=1)

        return out

class DFFN(nn.Module):
    def __init__(self, dim, hidden_features=None, bias=False):
        super(DFFN, self).__init__()

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias
    
def resize_complex_weight(origin_weight, new_h, new_w):
    h, w, num_heads = origin_weight.shape[0:3]  # size, w, c, 2
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
    new_weight = torch.nn.functional.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    ).permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)
    return new_weight
class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
 
class FDSM(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.conv_rgb = nn.Conv2d(c, c, 1, 1, 0, groups=c,bias=False)
        self.conv_nir = nn.Conv2d(c, c, 1, 1, 0, groups=c,bias=False)
        self.softmax=nn.SiLU()
        self.pool= nn.Sequential(nn.Conv2d(c, c, 1, 1, 0, bias=False),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False),
            LayerNorm(c, LayerNorm_type='WithBias'),
            nn.PReLU())
        self.fc_ab = nn.Sequential(
            nn.Conv2d(c, c*2, 1, 1, 0, bias=False))
        self.dynamic_rgb=DynamicFilter(c)
        self.dynamic_nir=DynamicFilter(c)
    def forward(self, rgb, nir):
        feat_1 = self.conv_rgb(rgb)
        feat_2 = self.conv_nir(nir)
        feat_sum = feat_1 + feat_2
        s = self.pool(feat_sum)
        z = s
        ab = self.fc_ab(z)
        B, C, H, W = ab.shape
        ab=ab.view(B,2, C//2,H,W)
        ab=self.softmax(ab)
        a = ab[:,0,...]
        b = ab[:,1,...]
        feat_1 = self.dynamic_rgb(feat_1.permute(0, 2, 3, 1),a.permute(0, 2, 3, 1))
        feat_2 = self.dynamic_nir(feat_2.permute(0, 2, 3, 1),b.permute(0, 2, 3, 1))

        return feat_1.permute(0, 3, 1, 2), feat_2.permute(0, 3, 1, 2)
class DynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=1, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=30, weight_resize=True,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        #print(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x,y):
        B, H, W, _ = x.shape
        #print(x.shape,y.shape)
        routeing = self.reweight(y.mean(dim=(1, 2))).view(B, self.num_filters,
                                                          -1).softmax(dim=1)
        x = self.pwconv1(x)
        #print(x.shape)
        x = self.act1(x)
        x = x.to(torch.float32)
        #print(x.shape)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1],
                                                    x.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)
        #print(complex_weights.shape,weight.shape)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)
        x = self.pwconv2(x)

        # plt.close(fig)
        return x





def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img
class SAM_nir(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM_nir, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 1, kernel_size, bias=bias)
        self.conv3 = conv(1, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img
##########################################################################
## U-Net
class Encoder_2(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Encoder_2, self).__init__()
        self.dim1=n_feat
        self.dim2=n_feat+scale_unetfeats
        self.dim3=n_feat+(scale_unetfeats*2)
        self.fce1 = [TransformerBlock(n_feat, kernel_size, reduction,act,bias, LayerNorm_type='WithBias',depth=i+1) for i in range(1)]
        self.fce2 = [TransformerBlock(n_feat+scale_unetfeats, kernel_size, reduction,act,bias, LayerNorm_type='WithBias',depth=i+1) for i in range(1)]
        self.fce3 = [TransformerBlock(n_feat+(scale_unetfeats*2), kernel_size, reduction,act,bias, LayerNorm_type='WithBias',depth=i+1) for i in range(1)]
        self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.fce1 = nn.Sequential(*self.fce1)
        self.fce2 = nn.Sequential(*self.fce2)
        self.fce3 = nn.Sequential(*self.fce3)
        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)



    def forward(self, x,y):
        enc1 = self.encoder_level1(x)
        mix_x1 = torch.cat((enc1,y[0]),dim=1)
        enc1 = self.fce1(mix_x1)
        enc1,nir_fea1= torch.split(enc1, split_size_or_sections=self.dim1,dim=1)

        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        mix_x2 = torch.cat((enc2,y[1]),dim=1)
        enc2 = self.fce2(mix_x2)
        enc2,nir_fea2= torch.split(enc2, split_size_or_sections=self.dim2,dim=1)
        
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        mix_x3 = torch.cat((enc3,y[2]),dim=1)
        enc3 = self.fce3(mix_x3)
        enc3,nir_fea3= torch.split(enc3, split_size_or_sections=self.dim3,dim=1)
        
        return [enc1, enc2, enc3]
class Encoder_nir(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Encoder_nir, self).__init__()

        self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(1)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)



    def forward(self, x):
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        
        return [enc1, enc2, enc3]
class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(1)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)



    def forward(self, x):
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        
        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(1)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21  = SkipUpSample(n_feat, scale_unetfeats)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1,dec2,dec3]
class Decoder_2(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder_2, self).__init__()

        self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(1)]
        self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(1)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21  = SkipUpSample(n_feat, scale_unetfeats)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1,dec2,dec3]

##########################################################################
##---------- Resizing Modules ----------    
class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

##########################################################################
##---------- FFTformer -----------------------
class FCENet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=64, scale_unetfeats=32, kernel_size=3, reduction=4, bias=False):
        super(FCENet, self).__init__()
        self.dim=n_feat
        act=nn.PReLU()
        self.embedding = nn.Conv2d(in_c, self.dim, 3, 1, 1, bias=False)
        self.embedding_nir = nn.Conv2d(1, self.dim, 3, 1, 1, bias=False)
        self.shallow_feat_rgb = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        
        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.concat12  = conv(n_feat+n_feat, n_feat, kernel_size, bias=bias)
        self.stage2_encoder_nir=Encoder_nir(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.stage2_encoder_rgb=Encoder_2(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.stage2_decoder = Decoder_2(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.output = nn.Conv2d(self.dim, out_c, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self,rgb,nir):
        #rgb,nir= torch.split(img, split_size_or_sections=3,dim=1)
        rgb_feature = self.embedding(rgb)
        
        nir_feature = self.embedding_nir(nir)
        rgb_shallow = self.shallow_feat_rgb(rgb)
        rgb_shallow = self.stage1_encoder(rgb_shallow)
        rgb_shallow_out = self.stage1_decoder(rgb_shallow)
        ## Apply Supervised Attention Module (SAM)
        stage1_rgbfeature, stage1_rgbout = self.sam12(rgb_shallow_out[0], rgb)
        stage2_rgb_in=self.concat12(torch.cat([rgb_feature, stage1_rgbfeature], 1))

        nir_featureout= self.stage2_encoder_nir(nir_feature)
        rgb_featureout= self.stage2_encoder_rgb(stage2_rgb_in,nir_featureout)
        rgb_out = self.stage2_decoder(rgb_featureout)

        out = self.output(rgb_out[0]) + rgb


        return out,stage1_rgbout



if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    opt=0
    model = FCENet(in_c=3, out_c=3, n_feat=64, scale_unetfeats=32, kernel_size=3, reduction=4, bias=False).cuda()
    #print(model)
    inputs = torch.randn((1, 4, 128, 128)).cuda()
    flops = FlopCountAnalysis(model,inputs)
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
