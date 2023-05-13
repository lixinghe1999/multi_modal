from .detector3d_template import Detector3DTemplate
from ..fusion import late_fusion

class CLOCs(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head', 'late_fusion'
        ]
        self.module_list = self.build_networks()
    def build_late_fusion(self, model_info_dict):
        late_fusion_module = late_fusion.__all__[self.model_cfg.FUSION.NAME](
            nms_config=self.model_cfg.POST_PROCESSING.NMS_CONFIG,
        )
        model_info_dict['module_list'].append(late_fusion_module)
        return late_fusion_module, model_info_dict
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
