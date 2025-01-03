import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class Net(nn.Module):
    def __init__(self, angRes, factor):
        super(Net, self).__init__()
        channels = 64
        self.channels = channels
        self.angRes = angRes
        self.factor = factor
        layer_num = 4

        self.pos_encoding = PositionEncoding(temperature=10000)
        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = 8
        self.MHSA_params['dropout'] = 0.

        ##################### Initial Convolution #####################
        self.conv_init0 = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_init = nn.Sequential(
              nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
              nn.LeakyReLU(0.2, inplace=True),
          )

        ################ Alternate AngTrans & C42Conv ################
        self.altblock = self.make_layer(layer_num=layer_num)

        ####################### UP Sampling ###########################
        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels*self.factor ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.PixelShuffle(self.factor),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def make_layer(self, layer_num):
        layers = []
        for i in range(layer_num):
            layers.append(AngFilter(self.angRes, self.channels, self.MHSA_params))
            layers.append(LWC42_Conv(self.angRes, self.channels))
        layers.append(nn.Conv3d(self.channels, self.channels, kernel_size=(1, 1, 1), padding=(0, 0, 0), dilation=1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, lr):
        # Bicubic
        lr_upscale = interpolate(lr, self.angRes, scale_factor=self.factor, mode='bicubic')

        lr = rearrange(lr, 'b c (a1 h) (a2 w) -> b c (a1 a2) h w', a1=self.angRes, a2=self.angRes)

        # Initial Convolution
        buffer_init = self.conv_init0(lr)
        buffer = self.conv_init(buffer_init)+buffer_init

        # Position Encoding
        ang_position = self.pos_encoding(buffer, dim=[2], token_dim=self.channels)
        for m in self.modules():
            m.ang_position = ang_position

        # Alternate AngTrans & SpaTrans
        buffer = self.altblock(buffer) + buffer

        # Up-Sampling
        buffer = rearrange(buffer, 'b c (a1 a2) h w -> (b a1 a2) c h w', a1=self.angRes, a2=self.angRes)
        buffer = self.upsampling(buffer)
        buffer = rearrange(buffer, '(b a1 a2) c h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        out = buffer + lr_upscale

        return out
    
class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv3d(channel,channel, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                              dilation=(1, 1, 1), bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False)

    def forward(self, x):
        out = self.conv_2(self.conv_1(x))
        return x + out

class PositionEncoding(nn.Module):
    def __init__(self, temperature):
        super(PositionEncoding, self).__init__()
        self.temperature = temperature

    def forward(self, x, dim: list, token_dim):
        self.token_dim = token_dim
        assert len(x.size()) == 5, 'the object of position encoding requires 5-dim tensor! '
        grid_dim = torch.linspace(0, self.token_dim - 1, self.token_dim, dtype=torch.float32)
        
        grid_dim = 2 * (grid_dim // 2) / self.token_dim
        grid_dim = self.temperature ** grid_dim
        position = None
        for index in range(len(dim)):
            pos_size = [1, 1, 1, 1, 1, self.token_dim]
            length = x.size(dim[index])
            pos_size[dim[index]] = length

            pos_dim = (torch.linspace(0, length - 1, length, dtype=torch.float32).view(-1, 1) / grid_dim).to(x.device)
            pos_dim = torch.cat([pos_dim[:, 0::2].sin(), pos_dim[:, 1::2].cos()], dim=1)
            pos_dim = pos_dim.view(pos_size)

            if position is None:
                position = pos_dim
            else:
                position = position + pos_dim
            pass

        position = rearrange(position, 'b 1 a h w dim -> b dim a h w')

        return position / len(dim)

class AngTrans(nn.Module):
    def __init__(self, channels, angRes, MHSA_params):
        super(AngTrans, self).__init__()
        self.angRes = angRes
        self.ang_dim = channels
        self.norm = nn.LayerNorm(self.ang_dim)
        self.attention = nn.MultiheadAttention(self.ang_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.ang_dim),
            nn.Linear(self.ang_dim, self.ang_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(self.ang_dim * 2, self.ang_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )

    def forward(self, buffer):
        b, c, a, h, w = buffer.shape
        ang_token = rearrange(buffer, 'b c a h w -> a (b h w) c')
        ang_PE = rearrange(self.ang_position, 'b c a h w -> a (b h w) c')
        ang_token_norm = self.norm(ang_token + ang_PE)
        ang_token = self.attention(query=ang_token_norm,
                                   key=ang_token_norm,
                                   value=ang_token,
                                   need_weights=False)[0] + ang_token

        ang_token = self.feed_forward(ang_token) + ang_token
        buffer = rearrange(ang_token, '(a) (b h w) (c) -> b c a h w', a=a, h=h, w=w)

        return buffer

class AngFilter(nn.Module):
    def __init__(self, angRes, channels, MHSA_params):
        super(AngFilter, self).__init__()
        self.angRes = angRes
        self.ang_trans = AngTrans(channels, angRes, MHSA_params)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        )
    def forward(self, x):
        shortcut = x
        [_, _, _, h, w] = x.size()
        buffer = self.conv(self.ang_trans(x)) + shortcut
        return buffer

class LWC42_Conv(nn.Module):
    def __init__(self, angRes, ch):
        super(LWC42_Conv, self).__init__()                
        S_ch, A_ch, E_ch, D_ch  = ch//4, ch//4, ch//4, ch//4
        self.angRes = angRes
        self.spaconv = SpatialConv(S_ch)
        self.angconv = AngularConv(A_ch, angRes, A_ch)
        self.epiconv = EPiConv(E_ch, angRes, E_ch)
        self.dpiconv = EPiConv(D_ch, angRes, D_ch)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.fuse = nn.Sequential(
                nn.Conv3d(in_channels = S_ch+A_ch+E_ch*2+D_ch*2+ch//4*3, out_channels = ch, kernel_size = 1, stride = 1, padding = 0, dilation=1),
                nn.LeakyReLU(0.2, inplace=True),
                # nn.Conv3d(ch, ch, kernel_size = (1,3,3), stride = 1, padding = (0,1,1), dilation=1))
                nn.Conv3d(ch, ch, kernel_size = (1,5,5), stride = 1, groups = ch, padding = (0,2,2), dilation=1))
        self.conv_0 = nn.Conv3d(in_channels = ch, out_channels = ch, kernel_size = 1, stride = 1, padding = 0, dilation=1)
        self.conv_1 = nn.Conv3d(in_channels = ch//4*3, out_channels = ch, kernel_size = 1, stride = 1, padding = 0, dilation=1)
        self.conv_2 = nn.Conv3d(in_channels = ch//4*3, out_channels = ch, kernel_size = 1, stride = 1, padding = 0, dilation=1)
        self.conv_3 = nn.Conv3d(in_channels = ch//4*3, out_channels = ch, kernel_size = 1, stride = 1, padding = 0, dilation=1)
        self.conv_4 = nn.Conv3d(in_channels = ch//4*3, out_channels = ch, kernel_size = 1, stride = 1, padding = 0, dilation=1)
        self.conv_5 = nn.Conv3d(in_channels = ch//4*3, out_channels = ch, kernel_size = 1, stride = 1, padding = 0, dilation=1)
        
    
    def forward(self,x):
        b, c, n, h, w = x.shape
        an = self.angRes

        buffer = self.lrelu(self.conv_0(x))
        buffer_1, buffer = ChannelSplit(buffer)
        s_out = self.spaconv(buffer_1)
        buffer = self.lrelu(self.conv_1(buffer))
        buffer_2, buffer = ChannelSplit(buffer)
        a_out = self.angconv(buffer_2)
        buffer = self.lrelu(self.conv_2(buffer))
        buffer_3, buffer = ChannelSplit(buffer)
        epih_in = buffer_3.contiguous().view(b, c//4, an, an, h, w) # b,c,u,v,h,w
        epih_out = self.epiconv(epih_in)
        buffer = self.lrelu(self.conv_3(buffer))
        buffer_4, buffer = ChannelSplit(buffer)
        epiv_in = buffer_4.contiguous().view(b, c//4, an, an, h, w).permute(0,1,3,2,5,4) # b,c,v,u,w,h
        epiv_out = self.epiconv(epiv_in).reshape(b, -1, an, an, w, h).permute(0,1,3,2,5,4).reshape(b, -1, n, h, w)
        buffer = self.lrelu(self.conv_4(buffer))
        buffer_5, buffer = ChannelSplit(buffer)
        dpih_in = buffer_5.contiguous().view(b, c//4, an, an, h, w).permute(0,1,3,2,4,5) # b,c,v,u,h,w
        dpih_out = self.dpiconv(dpih_in).reshape(b, -1, an, an, w, h).permute(0,1,3,2,4,5).reshape(b, -1, n, h, w)
        buffer = self.lrelu(self.conv_5(buffer))
        buffer_6, buffer = ChannelSplit(buffer)
        dpiv_in = buffer_6.contiguous().view(b, c//4, an, an, h, w).permute(0,1,2,3,5,4) # b,c,u,v,w,h
        dpiv_out = self.dpiconv(dpiv_in).reshape(b, -1, an, an, w, h).permute(0,1,2,3,5,4).reshape(b, -1, n, h, w)

        out = torch.cat((s_out, a_out, epih_out, epiv_out, dpih_out, dpiv_out, buffer), 1)
        out = self.fuse(out)

        return out + x
def ChannelSplit(input, factor=4):
    _, C, _, _, _= input.shape
    c = C//factor
    output_1 = input[:, :c,...]
    output_2 = input[:, c:,...]
    return output_1, output_2

class SpatialConv(nn.Module):
    def __init__(self, ch):
        super(SpatialConv, self).__init__()
        self.spaconv_s = nn.Sequential(
                    nn.Conv3d(in_channels = ch, out_channels = ch, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=(1,1,1)),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv3d(in_channels = ch, out_channels = ch, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=(1,1,1)),
                    nn.LeakyReLU(0.2, inplace=True)
                    )

    def forward(self,fm):
        return self.spaconv_s(fm) 



class AngularConv(nn.Module):
    def __init__(self, ch, angRes, AngChannel):
        super(AngularConv, self).__init__()
        self.angconv = nn.Sequential(
            nn.Conv3d(ch*angRes*angRes, AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(AngChannel, AngChannel * angRes * angRes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self,fm):
        b, c, n, h, w = fm.shape
        a_in = fm.contiguous().view(b,c*n,1,h,w)
        out = self.angconv(a_in).view(b,-1,n,h,w) # n == angRes * angRes
        return out

class EPiConv(nn.Module):
    def __init__(self, ch, angRes, EPIChannel):
        super(EPiConv, self).__init__()
        self.epi_ch = EPIChannel
        self.epiconv = nn.Sequential(
                    nn.Conv3d(ch, EPIChannel, kernel_size=(1, angRes, angRes//2*2+1), stride=1, padding=(0, 0, angRes//2), bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv3d(EPIChannel, angRes * EPIChannel, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False), # ksize maybe (1,1,angRes//2*2+1) ?
                    nn.LeakyReLU(0.2, inplace=True),
                    )

    def forward(self,fm):
        b, c, u, v, h, w = fm.shape
        epih_in = fm.permute(0, 1, 2, 4, 3, 5).reshape(b,c,u*h,v,w)
        epih_out = self.epiconv(epih_in) # (b,self.epi_ch*v, u*h, 1, w)
        epih_out = epih_out.reshape(b,self.epi_ch,v,u,h,w).permute(0,1,3,2,4,5).reshape(b, self.epi_ch,u*v,h,w)
        return epih_out

def interpolate(x, angRes, scale_factor, mode):
    [B, _, H, W] = x.size()
    h = H // angRes
    w = W // angRes
    x_upscale = x.view(B, 1, angRes, h, angRes, w)
    x_upscale = x_upscale.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * angRes ** 2, 1, h, w)
    x_upscale = F.interpolate(x_upscale, scale_factor=scale_factor, mode=mode, align_corners=False)
    x_upscale = x_upscale.view(B, angRes, angRes, 1, h * scale_factor, w * scale_factor)
    x_upscale = x_upscale.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, 1, H * scale_factor, W * scale_factor)
    # [B, 1, A*h*S, A*w*S]

    return x_upscale
if __name__ == "__main__":
    net = Net(5, 4)#.cuda()
    # from thop import profile
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    input = (torch.randn(1, 1, 160, 160),)#.cuda()
    # 
    # flops, params = profile(net, inputs=(input,))
    # 
    # print('   Number of FLOPs: %.2fG' % (flops / 1e9))    
    flops = FlopCountAnalysis(net, input)
    print("FLOPs: ", flops.total())
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of parameters: %.2fM' % (total / 1e6))
