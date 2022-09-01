import torch
import torch.nn as nn
import torch.nn.functional as F
from .aaamixer_sampling_3d import sampling_3d


def dprint(*args, **kwargs):
    import os
    if 'DEBUG' in os.environ:
        print(*args, **kwargs)

_dump_i = 0


class SRShadowForFlops(nn.Module):
    def __init__(self, in_dim, in_points, n_groups, query_dim=None,
                 out_dim=None, out_points=None, **kwargs):
        super(SRShadowForFlops, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

    def forward(self, x, query):
        pass

    @staticmethod
    def __user_flops_handle__(module, input, output):
        B, num_query, num_group, num_point, num_channel = input[0].shape

        eff_in_dim = module.in_dim//num_group
        eff_out_dim = module.out_dim//num_group
        in_points = module.in_points
        out_points = module.out_points

        step1 = B*num_query*num_group*in_points*eff_in_dim*eff_out_dim
        step2 = B*num_query*num_group*eff_out_dim*in_points*out_points
        module.__flops__ += int(step1+step2)
        pass


class AdaptiveMixing(nn.Module):
    def __init__(self, in_dim, in_points, n_groups, query_dim=None,
                 out_dim=None, out_points=None, sampling_rate=None):
        super(AdaptiveMixing, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim
        sampling_rate = sampling_rate if sampling_rate is not None else 1

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points//sampling_rate
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim//n_groups
        self.eff_out_dim = out_dim//n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points

        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Sequential(nn.Linear(self.query_dim, self.n_groups*self.total_parameters))

        self.out_proj = nn.Linear(self.eff_out_dim*self.out_points*self.n_groups, self.query_dim, bias=True)

        self.act = nn.ReLU(inplace=True)

        # virtual modules for FLOPs calculation
        local_dict = locals()
        local_dict.pop('self')
        self.shadow = SRShadowForFlops(**local_dict)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator[-1].weight)

    def forward(self, x, query):

        # Calculate FLOPs
        self.shadow(x, query)
        B, N, g, P, C = x.size()
        # batch, num_query, group, point, channel
        G = self.n_groups
        assert g == G
        # assert C*g == self.in_dim

        # query: B, N, C
        # x: B, N, G, Px, Cx

        global _dump_i

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*N, G, -1)

        out = x.reshape(B*N, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)

        if False:
            out = out.reshape(B*N*G, P, C)

            M = M.reshape(B*N*G, self.eff_in_dim, self.eff_in_dim)
            S = S.reshape(B*N*G, self.out_points, self.in_points)

            out = torch.bmm(out, M)
            out = F.layer_norm(out, [out.size(-2), out.size(-1)])
            out = self.act(out)

            '''adaptive spatial mixing'''
            out = torch.bmm(S, out)  # implicitly transpose and matmul
            out = F.layer_norm(out, [out.size(-2), out.size(-1)])
            out = self.act(out)
        else:
            M = M.reshape(B*N, G, self.eff_in_dim, self.eff_in_dim)
            S = S.reshape(B*N, G, self.out_points, self.in_points)

            '''adaptive channel mixing
            the process also can be done with torch.bmm
            but for clarity, we use torch.matmul
            '''
            out = torch.matmul(out, M)
            out = F.layer_norm(out, [out.size(-2), out.size(-1)])
            out = self.act(out)

            '''adaptive spatial mixing'''
            out = torch.matmul(S, out)  # implicitly transpose and matmul
            out = F.layer_norm(out, [out.size(-2), out.size(-1)])
            out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, N, -1)
        out = self.out_proj(out)
        out = query + out
        return out


class AdaptiveSamplingMixing(nn.Module):
    IND = 0

    def __init__(self,
                 in_points=32,
                 out_points=128,
                 n_groups=4,
                 content_dim=256,
                 feat_channels=None
                 ):
        super(AdaptiveSamplingMixing, self).__init__()
        self.in_points = in_points
        self.out_points = out_points
        self.n_groups = n_groups
        self.content_dim = content_dim
        self.feat_channels = feat_channels if feat_channels is not None else self.content_dim

        self.sampling_offset_generator = nn.Sequential(
            nn.Linear(content_dim, in_points * n_groups * 3))
        self.norm = nn.LayerNorm(content_dim)
        self.adaptive_mixing = AdaptiveMixing(
            self.feat_channels,
            query_dim=self.content_dim,
            in_points=self.in_points,
            out_points=self.out_points,
            n_groups=self.n_groups)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.sampling_offset_generator[-1].weight)
        nn.init.zeros_(self.sampling_offset_generator[-1].bias)

        bias = self.sampling_offset_generator[-1].bias.data.view(
            self.n_groups, self.in_points, 3)

        # if in_points are squared number, then initialize
        # to sampling on grids regularly, not used in most
        # of our experiments.
        if int(self.in_points ** 0.5) ** 2 == self.in_points:
            h = int(self.in_points ** 0.5)
            y = torch.linspace(-0.5, 0.5, h + 1) + 0.5 / h
            yp = y[:-1]
            y = yp.view(-1, 1).repeat(1, h)
            x = yp.view(1, -1).repeat(h, 1)
            y = y.flatten(0, 1)[None, :, None]
            x = x.flatten(0, 1)[None, :, None]
            bias[:, :, 0:2] = torch.cat([y, x], dim=-1)
        else:
            bandwidth = 0.5 * 1.0
            nn.init.uniform_(bias, -bandwidth, bandwidth)

        # initialize sampling delta z
        nn.init.constant_(bias[:, :, 2:3], -1.0)

        self.adaptive_mixing.init_weights()

    def forward(self, x, query_feat, query_xyzr, featmap_strides):
        offset = self.sampling_offset_generator(query_feat)
        sample_points_xyz = make_sample_points(offset, self.n_groups * self.in_points, query_xyzr)
        sampled_feature, _ = sampling_3d(sample_points_xyz, x,
                                         featmap_strides=featmap_strides,
                                         n_points=self.in_points)
        query_feat = self.adaptive_mixing(sampled_feature, query_feat)
        query_feat = self.norm(query_feat)
        AdaptiveSamplingMixing.IND += 1
        return query_feat


def make_sample_points(offset, num_group, xyzr):
    '''
        offset_yx: [B, L, num_group*3], normalized by stride
        return: [B, H, W, num_group, 3]
    '''
    B, L, _ = offset.shape

    offset = offset.view(B, L, 1, num_group, 3)

    # decode from xyzr to ccwh
    # z = (h*w).sqrt().log2()   # r = [(h/w).log2()*-0.5, (h/w).log2()*0.5]
    roi_cc = xyzr[..., :2]
    scale = 2.00 ** xyzr[..., 2:3]   # == z
    ratio = 2.00 ** torch.cat([xyzr[..., 3:4] * -0.5, xyzr[..., 3:4] * 0.5], dim=-1)  # == r
    roi_wh = scale * ratio

    # convert pred-offset to input-img-size
    offset_xy = offset[..., :2] * roi_wh.view(B, L, 1, 1, 2)
    sample_xy = roi_cc.contiguous().view(B, L, 1, 1, 2) + offset_xy
    roi_lvl = xyzr[..., 2:3].view(B, L, 1, 1, 1)
    sample_lvl = roi_lvl + offset[..., 2:3]

    sample_xyz = torch.cat([sample_xy, sample_lvl], dim=-1)

    return sample_xyz