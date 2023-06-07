import torch

def sharded_cross_view_inner_product(vid_embds,
                                     text_embds,
                                     vid_weights,
                                     text_weights,
                                     subspaces,
                                     merge_caption_similiarities='avg'):
  """Compute similarities between all captions and videos."""

  b = vid_embds[subspaces[0]].size(0)
  device = vid_embds[subspaces[0]].device
  num_caps = text_embds[subspaces[0]].size(1)
  m = len(subspaces)

  # unroll separate captions onto first dimension and treat them separately
  sims = torch.zeros(b * num_caps, b, device=device)

  text_weights = text_weights.view(b * num_caps, -1)
  vid_weights = vid_weights.view(b, -1)

  moe_weights = vid_weights[None, :, :] * text_weights[:, None, :]

  norm_weights = torch.sum(moe_weights, dim=2)
  norm_weights = norm_weights.unsqueeze(2)
  # If only one modality is used and is missing in some videos, moe_weights will
  # be zero.
  # To avoid division by zero, replace zeros by epsilon
  # (or anything else, in that case moe_weights are zero anyway)
  norm_weights[norm_weights == 0] = 1E-5
  moe_weights = torch.div(moe_weights, norm_weights)

  assert list(moe_weights.size()) == [b * num_caps, b, m]

  for idx, mod in enumerate(subspaces):
    text_embds[mod] = text_embds[mod].view(b * num_caps, -1)
    sims += moe_weights[:, :, idx] * torch.matmul(text_embds[mod],
                                               vid_embds[mod].t())

  if num_caps > 1:
    # aggregate similarities from different captions
    if merge_caption_similiarities == 'avg':
      sims = sims.view(b, num_caps, b)
      sims = torch.mean(sims, dim=1)
      sims = sims.view(b, b)
    elif merge_caption_similiarities == 'indep':
      pass
    else:
      msg = 'unrecognised merge mode: {}'
      raise ValueError(msg.format(merge_caption_similiarities))
  return sims