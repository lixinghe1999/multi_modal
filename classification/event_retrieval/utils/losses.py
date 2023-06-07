"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F
def loss_infoNCE(outputs, text_features):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    # alpha = params.alpha
    # T = params.temperature
    # KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
    #                          F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    # KD_loss = nn.KLDivLoss()(outputs,teacher_outputs)
    # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    # outputs = outputs / outputs.norm(dim=-1, keepdim=True)
    text_features = text_features.to(dtype=outputs.dtype)
    logits_image_text = outputs @ text_features.t()
    logits_text_image = text_features @ outputs.t()
    
    batch_size = outputs.shape[0]
    ground_truth = torch.arange(batch_size, dtype=torch.long, device='cuda')
    
    loss_image_text = F.cross_entropy(logits_image_text, ground_truth)
    loss_text_image = F.cross_entropy(logits_text_image, ground_truth)
    
    return (loss_image_text+loss_text_image)/2
class MaxMarginRankingLoss(torch.nn.Module):
  """Implementation of the Max-margin ranking loss."""

  def __init__(self, margin=0.05, fix_norm=False):
    super().__init__()
    self.fix_norm = fix_norm
    self.loss = torch.nn.MarginRankingLoss(margin)
    self.margin = margin

  def forward(self, x):
    n = x.size()[0]
    x1 = torch.diag(x)
    x1 = x1.unsqueeze(1)
    x1 = x1.expand(n, n)
    x1 = x1.contiguous().view(-1, 1)
    x1 = torch.cat((x1, x1), 0)

    x2 = x.view(-1, 1)
    x3 = x.transpose(0, 1).contiguous().view(-1, 1)

    x2 = torch.cat((x2, x3), 0)
    max_margin = F.relu(self.margin - (x1 - x2))
    # print(x1, x2)
    if self.fix_norm:
      # remove the elements from the diagonal
      keep = torch.ones(x.shape) - torch.eye(x.shape[0])
      keep1 = keep.view(-1, 1)
      keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
      keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
      if x1.is_cuda:
        keep_idx = keep_idx.cuda()
      x1_ = torch.index_select(x1, dim=0, index=keep_idx)
      x2_ = torch.index_select(x2, dim=0, index=keep_idx)
      max_margin = F.relu(self.margin - (x1_ - x2_))

    return max_margin.mean()
  
class ConvNextDistillDiffPruningLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, teacher_model, base_criterion: torch.nn.Module, ratio_weight=10.0, distill_weight=0.5, keep_ratio=[0.9, 0.7, 0.5], clf_weight=0, mse_token=False, print_mode=True, swin_token=False):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.clf_weight = clf_weight
        self.keep_ratio = keep_ratio
        self.count = 0
        self.print_mode = print_mode
        self.cls_loss = 0
        self.ratio_loss = 0
        self.cls_distill_loss = 0
        self.token_distill_loss = 0
        self.mse_token = mse_token
        self.ratio_weight = ratio_weight
        self.distill_weight = distill_weight
        self.swin_token = swin_token

        print('ratio_weight=', ratio_weight, 'distill_weight', distill_weight)


    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        pred, token_pred, out_pred_score = outputs

        pred_loss = 0.0

        ratio = self.keep_ratio
            
        for i, score in enumerate(out_pred_score):
            if not self.swin_token:
                pos_ratio = score.mean(dim=(2,3))
            else:
                pos_ratio = score.mean(dim=1)
            pred_loss = pred_loss + ((pos_ratio - ratio[i]) ** 2).mean()

        cls_loss = self.base_criterion(pred, labels)

        with torch.no_grad():
            cls_t, token_t = self.teacher_model(inputs)

        cls_kl_loss = F.kl_div(
                F.log_softmax(pred, dim=-1),
                F.log_softmax(cls_t, dim=-1),
                reduction='batchmean',
                log_target=True
            )

        token_kl_loss = torch.pow(token_pred - token_t, 2).mean()

        # print(cls_loss, pred_loss)
        loss = self.clf_weight * cls_loss + self.ratio_weight * pred_loss / len(out_pred_score) + self.distill_weight * cls_kl_loss + self.distill_weight * token_kl_loss 
        loss_part = []

        if self.print_mode:
            self.cls_loss += cls_loss.item()
            self.ratio_loss += pred_loss.item()
            self.cls_distill_loss += cls_kl_loss.item()
            self.token_distill_loss += token_kl_loss.item()
            self.count += 1
            loss_part.append(cls_loss)
            loss_part.append(pred_loss)
            loss_part.append(cls_kl_loss)
            loss_part.append(token_kl_loss)
            if self.count == 100:
                print('loss info: cls_loss=%.4f, ratio_loss=%.4f, cls_kl=%.4f, token_kl=%.4f, layer_mse=%.4f, feat_kl=%.4f' % (self.cls_loss / 100, self.ratio_loss / 100, self.cls_distill_loss/ 100, self.token_distill_loss/ 100, self.layer_mse_loss / 100, self.feat_distill_loss / 100))
                self.count = 0
                self.cls_loss = 0
                self.ratio_loss = 0
                self.cls_distill_loss = 0
                self.token_distill_loss = 0
        return loss, loss_part



class DistillDiffPruningLoss_dynamic(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, teacher_model, base_criterion: torch.nn.Module, ratio_weight=2.0, distill_weight=0.5,
                 dynamic=False, pruning_loc=[3,6,9], keep_ratio=[0.75, 0.5, 0.25], clf_weight=0, mse_token=False, print_mode=True, device='cuda'):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.clf_weight = clf_weight
        self.pruning_loc = pruning_loc
        self.keep_ratio = keep_ratio
        self.count = 0
        self.print_mode = print_mode
        self.cls_loss = 0
        self.ratio_loss = 0
        self.cls_distill_loss = 0
        self.token_distill_loss = 0
        self.mse_token = mse_token
        self.dynamic = dynamic
        self.device = device
        self.ratio_weight = ratio_weight
        self.distill_weight = distill_weight

        # print('ratio_weight=', ratio_weight, 'distill_weight', distill_weight)


        if dynamic:
            print('using dynamic loss')

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        pred, token_pred, mask, out_pred_score, early_output = outputs

        ratio_loss = 0.0

        for i, score in enumerate(out_pred_score):
            if self.dynamic:
                pos_ratio = score.mean()
            else:
                pos_ratio = score.mean(1)
            # ratio_loss += ((pos_ratio - self.keep_ratio[i]) ** 2).mean()
            # extra loss: get unbalanced between two modalities
            ratio_loss += (pos_ratio ** 2).mean()
            ratio_loss += (score[:, :256].mean() - score[:, 256:].mean())**2

        cls_loss = 0

        if isinstance(labels, dict):
            for key in labels:
                cls_loss += self.base_criterion(pred[key], labels[key].to(self.device)) * 0.5
        else:
            cls_loss += self.base_criterion(pred, labels.to(self.device))
        early_weight = [i/len(early_output) for i in range(len(early_output))]
        for i in range(len(early_output)):
            if isinstance(labels, dict):
                for key in labels:
                    cls_loss += self.base_criterion(early_output[i][key], labels[key].to(self.device)) * 0.5 * early_weight[i]
            else:
                cls_loss += self.base_criterion(early_output[i], labels.to(self.device)) * early_weight[i]
                    
        with torch.no_grad():
            cls_t, token_t = self.teacher_model(*inputs)
        cls_kl_loss = 0
        if isinstance(labels, dict):
            for key in labels:
                cls_kl_loss += F.kl_div(
                        F.log_softmax(pred[key], dim=-1),
                        F.log_softmax(cls_t[key], dim=-1),
                        reduction='batchmean',
                        log_target=True
                    )
        else:
            cls_kl_loss += F.kl_div(
                        F.log_softmax(pred, dim=-1),
                        F.log_softmax(cls_t, dim=-1),
                        reduction='batchmean',
                        log_target=True
                    )

        B, N, C = token_pred.size()
        assert mask.numel() == B * N

        bool_mask = mask.reshape(B*N) > 0.5

        loss_part = []

        token_pred = token_pred.reshape(B*N, C)
        token_t = token_t.reshape(B*N, C)

        if mask.sum() < 0.1:
            token_kl_loss = token_pred.new(1,).fill_(0.0)
        else:
            token_t = token_t[bool_mask]
            token_pred = token_pred[bool_mask]
            if self.mse_token:
                token_kl_loss = torch.pow(token_pred - token_t, 2).mean()
            else:
                token_kl_loss = F.kl_div(
                        F.log_softmax(token_pred, dim=-1),
                        F.log_softmax(token_t, dim=-1),
                        reduction='batchmean',
                        log_target=True
                    )
        
        # print(cls_loss, pred_loss)
        loss = self.clf_weight * cls_loss + self.ratio_weight * ratio_loss / len(self.pruning_loc) + self.distill_weight * cls_kl_loss + self.distill_weight * token_kl_loss

        if self.print_mode:
            self.cls_loss += cls_loss.item()
            self.ratio_loss += ratio_loss.item()
            self.cls_distill_loss += cls_kl_loss.item()
            self.token_distill_loss += token_kl_loss.item()
            loss_part.append(cls_loss)
            loss_part.append(ratio_loss)
            loss_part.append(cls_kl_loss)
            loss_part.append(token_kl_loss)
            self.count += 1
            if self.count == 100:
                print('loss info: cls_loss=%.4f, ratio_loss=%.4f, cls_kl=%.4f, token_kl=%.4f' % (
                    self.cls_loss / 100, self.ratio_loss / 100, self.cls_distill_loss/ 100, self.token_distill_loss/ 100))
                self.count = 0
                self.cls_loss = 0
                self.ratio_loss = 0
                self.cls_distill_loss = 0
                self.token_distill_loss = 0
        return loss, loss_part

