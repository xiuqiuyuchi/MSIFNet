import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from torchvision.models import resnet50, ResNet50_Weights


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3,dim=2)  # (B, num_heads, head_dim, N)

        attn = (k.transpose(-1, -2) @ q) * self.scale
        attn = attn.softmax(dim=-2)  # (B, h, N, N)
        attn = self.attn_drop(attn)

        x = (v @ attn).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Encoding(nn.Module):
    def __init__(self, in_channels, num_codes):
        super(Encoding, self).__init__()
        self.in_channels = in_channels
        self.num_codes = num_codes
        std = 1. / ((64 * in_channels) ** 0.5)
        self.codewords = nn.Parameter(
            torch.empty(num_codes, in_channels, dtype=torch.float).uniform_(-std, std), requires_grad=True)
        self.scale = nn.Parameter(torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0), requires_grad=True)

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.in_channels
        b, in_channels, w, h = x.size()
        x = x.view(b, in_channels, -1).transpose(1, 2).contiguous()
        num_codes, in_channels = self.codewords.size()
        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, in_channels))
        reshaped_codewords = self.codewords.view((1, 1, num_codes, in_channels))
        reshaped_scale = self.scale.view((1, 1, num_codes))
        scaled_l2_norm = reshaped_scale * (expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        scaled_l2_norm = F.softmax(scaled_l2_norm, dim=2)
        scaled_l2_norm = scaled_l2_norm.unsqueeze(3)
        encoded_feat = (scaled_l2_norm * (expanded_x - reshaped_codewords)).sum(1)
        return encoded_feat


class PosCNN(nn.Module):
    def __init__(self, in_channels, out_channels, sa=False):
        super().__init__()
        expansion = 4
        c = out_channels // expansion
        self.sa = sa
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, c, kernel_size=1, padding=0, bias=False),
                                   nn.BatchNorm2d(c, eps=1e-6),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False),
                                   nn.Conv2d(c, out_channels, 1, padding=0, groups=1, bias=False),
                                   nn.BatchNorm2d(out_channels, eps=1e-6),
                                   nn.ReLU(inplace=True))
        self.act = nn.ReLU(inplace=True)
        if sa is not False:
            self.SA = Attention(1024, attn_drop=0.2, proj_drop=0.2)
        self.apply(init_weights)

    def forward(self, x):
        x_re = x
        if self.sa is not False:
            x = self.SA(x)
            x = self.conv1(x) + x_re
            return self.act(x)
        x = self.conv1(x)
        return self.act(x + x_re)


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class Global_LocalFeatures(nn.Module):
    def __init__(self, in_channels, num_codes):
        super().__init__()

        self.conv1_3_1 = PosCNN(in_channels, in_channels)
        self.LCF = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                 Encoding(in_channels, num_codes),
                                 nn.BatchNorm1d(num_codes),
                                 nn.ReLU(inplace=True),
                                 Mean(dim=1))
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.conv1_3_1(x)
        x_lcf = self.LCF(x)
        x_lcf = self.fc(x_lcf).view(b, c, 1, 1)
        x = F.relu_(x * x_lcf)
        return x


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_small = nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9,dilation=3, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.SA = Attention(512, attn_drop=0.2, proj_drop=0.2)
        self.conv11 = nn.Conv2d(2, 1, 1)
        self.conv22 = nn.Conv2d(2, 1, 1)
        self.apply(init_weights)

    def forward(self, x):
        xs = self.SA(x)
        xs = self.conv0(xs)
        x1 = self.conv_small(xs)
        x2 = self.conv_spatial(xs)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        x1_max, _ = torch.max(x1, dim=1, keepdim=True)
        x1_avg = torch.mean(x1, dim=1, keepdim=True)
        x2_max, _ = torch.max(x2, dim=1, keepdim=True)
        x2_avg = torch.mean(x2, dim=1, keepdim=True)

        x1_c = torch.cat((x1_max, x2_avg), dim=1)
        x1_c = self.conv11(x1_c)
        x2_c = torch.cat((x1_avg, x2_max), dim=1)
        x2_c = self.conv22(x2_c)

        agg_1 = torch.cat((x1_c, x2_c), dim=1)
        agg_1 = self.conv_squeeze(agg_1).sigmoid()
        xt1 = x1 * agg_1[:, 0, :, :].unsqueeze(1) + x2 * agg_1[:, 1, :, :].unsqueeze(1)
        xt1 = self.conv(xt1)

        return xt1 + x


class CrossAttention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(init_weights)

    def forward(self, x, f_kv, f3):
        x0 = f3
        x_b, x_c, x_h, x_w = x.shape

        x = x.flatten(2).transpose(2, 1)
        f_kv = f_kv.flatten(2).transpose(2, 1)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(f_kv).reshape(B, N, 2, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.transpose(2, 1).reshape(x_b, x_c, x_h, x_w)
        out = out + x0

        return out


class Fusionblock(nn.Module):
    def __init__(self, in_channels, num_codes, dropout=0.):
        super().__init__()
        self.LF = Global_LocalFeatures(in_channels=in_channels, num_codes=num_codes)
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.SA = Attention(2048, attn_drop=0.2, proj_drop=0.2)
        self.conv = nn.Conv2d(2*in_channels, in_channels, 1)
        self.apply(init_weights)

    def forward(self, x3):
        x3 = self.Conv1(x3)
        x3_1 = self.SA(x3)
        x3_1 = self.LF(x3_1)

        return x3_1 + x3


def l2_normalization(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x = torch.div(x, norm)
    return x

class ResNet(nn.Module):
    def __init__(self, num_classes=30, dropout=0.2):
        super(ResNet, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.Fub = Fusionblock(2048, 32, dropout=dropout)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.convb = PosCNN(1024, 1024, sa=True)
        self.fc1 = nn.Linear(2048, num_classes)
        self.fc2 = nn.Linear(2048, num_classes)
        self.LS = LSKblock(512)
        self.cr = CrossAttention(2048, attn_drop=dropout, proj_drop=dropout)
        self.conv1 = nn.Sequential(nn.Conv2d(512, 2048, kernel_size=1, stride=1),
                                   nn.AvgPool2d(4, 4))
        self.conv2 = nn.Sequential(nn.Conv2d(1024, 2048, kernel_size=1, stride=1),
                                   nn.AvgPool2d(2, 2))

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x1 = self.LS(x)
        x1 = self.conv1(x1)

        x = self.backbone.layer3(x)
        x2 = self.convb(x)
        x2 = self.conv2(x2)

        x = self.backbone.layer4(x)
        p1 = self.backbone.avgpool(x).flatten(1)
        p1 = self.fc1(p1)

        x3 = self.Fub(x)
        x = self.cr(x1, x2, x3)

        x = self.avgpool(x).flatten(1)
        x = l2_normalization(x)
        p2 = self.fc2(x)
        p = (p1 + p2) / 2.

        return p


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu', mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


if __name__ == '__main__':
    y = torch.randn(1, 3, 256, 256).cuda()
    model = ResNet().cuda()
    summary(model, input_size=(3, 256, 256), batch_size=32)

