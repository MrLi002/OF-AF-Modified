import torch
from torch import nn
from .blocks import ResBlock, BackwardWarp


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class Query(nn.Module):
    def __init__(self, base_channel, fix_query=False):
        super().__init__()
        self.fix_query = fix_query

    def forward(self, if0, flowti, flowtj, ctxti, ctxtj, t):
        if self.fix_query:
            return flowti * (1 - t) + flowtj * t, ctxti * (1 - t) + ctxtj * t
        return flowti * t + flowtj * (1 - t), ctxti * t + ctxtj * (1 - t)


def conv1x1(ch_in, ch_out):
    return nn.Conv2d(ch_in, ch_out, kernel_size=1)


def conv3x3(ch_in, ch_out):
    return nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)


def conv_resblock_one(ch_in, ch_out):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(ch_out, ch_out, 3, padding=1)
    )


class OffsetEstimator(nn.Module):
    def __init__(self, ch_in_frame, ch_in_event, ch_in_ref):
        super(OffsetEstimator, self).__init__()
        nf = ch_in_frame * 2 + ch_in_event + ch_in_ref
        self.conv0 = nn.Sequential(
            conv1x1(nf, nf),
            nn.ReLU(),
            conv_resblock_one(nf, nf),
            conv_resblock_one(nf, nf),
        )
        num = 2 * 3 * 3
        na = nf
        self.conv2 = nn.Sequential(
            conv3x3(na, nf),
            nn.ReLU()
        )
        self.offset = conv3x3(nf, num)
        self.mask = conv3x3(nf, num // 2)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv2(x)
        offset = self.offset(x)
        mask = self.mask(x)
        return offset, mask


class BidirFlow(nn.Module):
    def __init__(self, base_channel, fix_warp=False):
        super().__init__()
        self.grid = {}
        self.bwarp = BackwardWarp(fix_half_pixel=fix_warp)
        self.res0 = ResBlock(base_channel*12, base_channel*4)
        self.res1 = ResBlock(base_channel*12, base_channel*4)
        self.res2 = ResBlock(base_channel*16 + 6, base_channel*4)
        self.gru = SepConvGRU(base_channel*4, base_channel*8)

        self.conv0 = nn.Conv2d(base_channel*8, 2, 3, 1, 1)
        self.conv1 = nn.Conv2d(base_channel*12, 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(base_channel*4, 2, 3, 1, 1)

        self.flow = nn.Sequential(
            nn.Conv2d(base_channel*4, base_channel*8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(base_channel*8, 2, 3, 1, 1)
        )

    def forward(self, if0, ift0, flowt0, motion0t, ctxt0, voxel):
        ctxt0_cur = self.res0(torch.cat([ctxt0, voxel, ift0], dim=1))
        motion0t_cur = self.res1(torch.cat([ctxt0_cur, motion0t, voxel], dim=1))

        flow0 = self.conv0(torch.cat([ift0, ctxt0_cur], dim=1))
        ift0_cur0 = self.bwarp(ift0, flow0)

        flow1 = flowt0 + flow0
        ift0_cur1 = self.bwarp(if0, flow1)

        flow2 = self.conv1(torch.cat([if0, motion0t_cur, ift0_cur0], dim=1))
        ift0_cur2 = self.bwarp(if0, flow2)

        final = self.res2(torch.cat([if0, ift0_cur0, ift0_cur1, ift0_cur2, flow0, flow1, flow2], dim=1))
        flowt0 = self.conv2(final) + flow2
        ift0_cur = self.bwarp(if0, flowt0)

        flow0t = self.flow(self.gru(if0, torch.cat([ctxt0_cur, motion0t_cur], dim=1)))
        return ift0_cur, motion0t_cur, flow0t, flowt0, ctxt0_cur