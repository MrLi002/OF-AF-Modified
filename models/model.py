import torch
import torch.nn as nn
from .encoder import EventsEncoder, ImageEncoder
from .BiFEB import BidirFlow, Query
from .BiW import BiW

from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.base_channel = 32
		self.voxel_bins = args.voxel_bins
		self.nb_of_flow = args.nb_of_flow
		self.fix_query = getattr(args, "fix_query", False)
		self.fix_warp = getattr(args, "fix_warp", False)
		self.bins_per_flow = self.voxel_bins//self.nb_of_flow

		self.events_encoder = EventsEncoder(self.voxel_bins, self.base_channel, bins_per_flow=self.bins_per_flow)
		self.im_encoder = ImageEncoder(3, self.base_channel)
		self.bidir_flow0 = BidirFlow(self.base_channel, fix_warp=self.fix_warp)
		self.bidir_flow1 = BidirFlow(self.base_channel, fix_warp=self.fix_warp)
		self.query = Query(self.base_channel, fix_query=self.fix_query)
		self.biw = BiW(self.base_channel, fix_warp=self.fix_warp)

	def forward(self, im0, im1, event, interp_times, return_aux=False):
		efs01 = self.events_encoder(im0, im1, event)
		if0_1, if0_4 = self.im_encoder(im0)
		if1_1, if1_4 = self.im_encoder(im1)
		b, t, c, h, w = efs01.shape

		ift0, ift1 = if0_4, if1_4
		ctxt0, ctxt1 = if0_4, if1_4
		ctxst0, ctxst1 = [if0_4], [if1_4]
		motion0t, motion1t = torch.zeros_like(if0_4), torch.zeros_like(if1_4)
		zero_flow = torch.zeros((b, 2, h, w), device=im0.device, dtype=im0.dtype, requires_grad=False)
		flowt0, flowt1 = zero_flow, zero_flow
		flows0t, flows1t = [zero_flow], [zero_flow]
		flowst0, flowst1 = [zero_flow], [zero_flow]
		for t_ in range(self.nb_of_flow):
			ift0, motion0t, flow0t, flowt0, ctxt0 = self.bidir_flow0(if0_4, ift0, flowt0, motion0t, ctxt0, efs01[:, t_])
			ift1, motion1t, flow1t, flowt1, ctxt1 = self.bidir_flow1(if1_4, ift1, flowt1, motion1t, ctxt1, efs01[:, t - t_ - 1])
			ctxst0.append(ctxt0)
			ctxst1.insert(0, ctxt1)
			flowst0.append(flowt0)
			flowst1.insert(0, flowt1)
			flows0t.append(flow0t)
			flows1t.insert(0, flow1t)
			'''
			flows0t =[0→0,0→1/16,0→2/16,...0→15/16,0→1]
			flows1t =[1→0,1→1/16,1→2/16,...1→15/16,1→1]
			flowst0 =[0→0,1/16→0,2/16→0,3/16→0,...15/16→0,1→0]
			flowst1 =[0→1,1/16→1,2/16→1,3/16→1,...15/16→1,1→1]
			'''

		# All lists are ordered by physical time: node k corresponds to k / nb_of_flow.
		# Keep legacy interpolation as the default for published checkpoints.
		imst, fwd0_list, fwd1_list, valid0_list, valid1_list = [], [], [], [], []
		if interp_times.ndim != 2 or interp_times.shape[0] != b:
			raise ValueError("interp_times must be [B, query_count]")
		for i in range(interp_times.shape[1]):
			time_batch = interp_times[:, i].to(dtype=im0.dtype)
			if bool(((time_batch <= 0) | (time_batch >= 1)).any()):
				raise ValueError("interp_times must be strictly within (0, 1)")
			index = time_batch * self.nb_of_flow
			low = index.floor().long().clamp(max=self.nb_of_flow - 1)
			pct = (index - low).view(b, 1, 1, 1)
			t = time_batch.view(b, 1, 1, 1)

			def pick(items, indices):
				# Avoid stacking all context features (especially costly on high-res images).
				return torch.cat([items[int(k)][j:j + 1] for j, k in enumerate(indices.tolist())], dim=0)

			flow0t = pick(flows0t, low) * (1 - pct) + pick(flows0t, low + 1) * pct
			if self.fix_query:
				flow1t = pick(flows1t, low) * (1 - pct) + pick(flows1t, low + 1) * pct
			else:
				flow1t = pick(flows1t, low + 1) * (1 - pct) + pick(flows1t, low) * pct

			flowt0, ctxt0 = self.query(if0_4, pick(flowst0, low), pick(flowst0, low + 1),
										pick(ctxst0, low), pick(ctxst0, low + 1), pct)
			flowt1, ctxt1 = self.query(if1_4, pick(flowst1, low), pick(flowst1, low + 1),
										pick(ctxst1, low), pick(ctxst1, low + 1), pct)

			args = (im0, im1, t, flow0t, flow1t, flowt0, flowt1, ctxt0, ctxt1, if0_1, if1_1)
			if self.training:
				output = checkpoint(self.biw, *args, use_reentrant=False, return_aux=return_aux)
			else:
				output = self.biw(*args, return_aux=return_aux)
			if return_aux:
				imt, fwd0, fwd1, valid0, valid1 = output
				fwd0_list.append(fwd0)
				fwd1_list.append(fwd1)
				valid0_list.append(valid0)
				valid1_list.append(valid1)
			else:
				imt = output
			imst.append(imt)

		preds = torch.stack(imst, dim=1)
		if return_aux:
			return (preds, torch.stack(fwd0_list, dim=1), torch.stack(fwd1_list, dim=1),
					torch.stack(valid0_list, dim=1), torch.stack(valid1_list, dim=1))
		return preds
