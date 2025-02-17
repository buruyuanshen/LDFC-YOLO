import torchvision  
import torch.nn as nn  
import torch 
import math
import torch
import torch.nn as nn
from einops import rearrange

__all__ = ['LDFC']  


class deformable_LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.conv0 = WLConv5(inc=dim, outc=dim, num_param=5, stride=1, bias=None) 
        self.conv_spatial = WLConv7(inc=dim, outc=dim, num_param=7, stride=1,dilation=3, bias=None) 
        self.conv1 = nn.Conv2d(dim, dim, 1) 

    def forward(self, x):
        u = x.clone()  
        attn = self.conv0(x)  
        attn = self.conv_spatial(attn)  
        attn = self.conv1(attn)
        if attn.size(2) > u.size(2) or attn.size(3) > u.size(3):
            attn = attn[:, :, :u.size(2), :u.size(3)]  
        return u * attn  
       

class LDFC(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        
        self.proj_1 = nn.Conv2d(d_model, d_model, 1) 
        self.activation = nn.GELU() 
        self.spatial_gating_unit = deformable_LKA(d_model) 
        self.proj_2 = nn.Conv2d(d_model, d_model, 1) 

    def forward(self, x):
        
        shorcut = x.clone() 
        x = self.proj_1(x) 
        x = self.activation(x) 
        x = self.spatial_gating_unit(x) 
        x = self.proj_2(x) 
        x = x + shorcut 
        return x

def autopad(k, p=None, d=1):  
    """Pad to 'same' shape outputs."""  
    if d > 1: 
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  
    if p is None: 
        p = (k-1)// 2 if isinstance(k, int) else [x // 2 for x in k]  
    return p

class WLConv5(nn.Module):
    def __init__(self, inc, outc, num_param=2, stride=1,p=None,dilation=1,bias=None):
        super(AKConv5, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.padding_h = autopad(num_param, p, dilation) 
        self.padding_w = 0
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1),padding=(self.padding_h, self.padding_w),bias=bias),
                                  nn.BatchNorm2d(outc),
                                  nn.SiLU())  
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)
 
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
 
    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)
 
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
 
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
 
        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)
 
        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
 
        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
 
        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
 
        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)
 
        return out
 
    # generating the inital sampled shapes for the AKConv with different sizes.
    
    def _get_p_n(self, N, dtype):
        p_n_x = torch.tensor([0, 0, 1, 2, 2])
        p_n_y = torch.tensor([0, 2, 1, 0, 2])
        p_n = torch.cat([p_n_x,p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n
       
    def _get_p_n(self, N, dtype):
        p_n_x = torch.tensor([0, 0, 0, 1, 1])
        p_n_y = torch.tensor([0, 1, 2, 0, 2])
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n
    
       
    def _get_p_n(self, N, dtype):
        p_n_x = torch.tensor([0, 0, 0, 0, 1])
        p_n_y = torch.tensor([0, 1, 2, 3, 2])
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n
    
    def _get_p_n(self, N, dtype):
        p_n_x = torch.tensor([0, 1, 2, 3, 4])
        p_n_y = torch.tensor([0, 1, 2, 3, 4])
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    
    def _get_p_n(self, N, dtype):
        p_n_x = torch.tensor([0, 1, 1, 1, 2])
        p_n_y = torch.tensor([0, 1, 2, 3, 0])
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n
    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))
 
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
 
        return p_0
 
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
 
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p
 
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
        x_dim = x.size(-1)
 
        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        index = torch.clamp(index, 0, x_dim - 1)
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
 
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
 
        return x_offset
 
    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)
 
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset

class WLConv7(nn.Module):
    def __init__(self, inc, outc, num_param=2, stride=1,p=None,dilation=1,bias=None):
        super(AKConv7, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.padding_h = autopad(num_param, p, dilation) 
        self.padding_w = 0
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1),padding=(self.padding_h, self.padding_w),bias=bias),
                                  nn.BatchNorm2d(outc),
                                  nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)
 
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
 
    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)
 
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
 
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
 
        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)
 
        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
 
        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
 
        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
 
        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)
 
        return out
 
    # generating the inital sampled shapes for the AKConv with different sizes.
    
    def _get_p_n(self, N, dtype):
        p_n_x = torch.tensor([0, 1, 2, 2, 2, 4, 5])
        p_n_y = torch.tensor([0, 1, 2, 3, 4, 1, 0])
        p_n = torch.cat([p_n_x,p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    
    def _get_p_n(self, N, dtype):
        p_n_x = torch.tensor([0, 1, 1, 2, 3, 3, 4])
        p_n_y = torch.tensor([0, 1, 3, 2, 1, 3, 4])
        p_n = torch.cat([p_n_x,p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_n(self, N, dtype):
        p_n_x = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        p_n_y = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n
        
    def _get_p_n(self, N, dtype):
        p_n_x = torch.tensor([0, 0, 1, 1, 2, 2, 3])
        p_n_y = torch.tensor([0, 1, 0, 1, 0, 1, 1])
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n
        
    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))
 
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
 
        return p_0
 
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
 
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p
 
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
        x_dim = x.size(-1)
 
        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        index = torch.clamp(index, 0, x_dim - 1)
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
 
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
 
        return x_offset
 
    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)
 
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset
