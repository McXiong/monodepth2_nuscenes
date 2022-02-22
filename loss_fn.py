
import torch.nn.functional as F
from components import get_smooth_loss
from model.layers import SSIM, Backproject, Project
from model.utils import *


class LossFn:
    def __init__(self, opts):
        self.opt = opts
        self.ssim = SSIM()
        self.backproject = Backproject(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)
        self.project_3d = Project(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)

    def compute_disp_losses(self, outputs, inputs):
        loss_dict = {}
        for scale in self.opt.scales:
            """
            initialization
            """
            disp = outputs[("disp", 0, scale)]
            target = self.get_color_input(inputs, 0, 0)
            reprojection_losses = []

            """
            reconstruction
            """
            outputs = self.generate_images_pred(inputs, outputs, scale)

            """
            automask
            """
            for frame_id in self.opt.frame_ids[1:]:
                pred = self.get_color_input(inputs, frame_id, 0)
                color_diff = self.compute_reprojection_loss(pred, target)
                identity_reprojection_loss = color_diff + torch.randn(color_diff.shape).type_as(color_diff) * 1e-5
                reprojection_losses.append(identity_reprojection_loss)

            """
            minimum reconstruction loss
            """
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_loss = torch.cat(reprojection_losses, 1)

            min_reconstruct_loss, _ = torch.min(reprojection_loss, dim=1)
            loss_dict[('min_reconstruct_loss', scale)] = min_reconstruct_loss.mean() / len(self.opt.scales)

            """
            disp mean normalization
            """
            if self.opt.disp_norm:
                mean_disp = disp.mean(2, True).mean(3, True)
                disp = disp / (mean_disp + 1e-7)

            """
            smooth loss
            """
            smooth_loss = get_smooth_loss(disp, self.get_color_input(inputs, 0, scale))
            loss_dict[('smooth_loss', scale)] = self.opt.disparity_smoothness * smooth_loss / (2 ** scale) / len(
                self.opt.scales)

        return loss_dict

    def compute_reprojection_loss(self, pred, target):
        photometric_loss = robust_l1(pred, target).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        return reprojection_loss

    def get_color_input(self, inputs, frame_id, scale):
        return inputs[("color_equ", frame_id, scale)] if self.opt.use_equ else inputs[("color", frame_id, scale)]

    def generate_images_pred(self, inputs, outputs, scale):
        disp = outputs[("disp", 0, scale)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            T = outputs[("cam_T_cam", 0, frame_id)]
            cam_points = self.backproject(depth, inputs["inv_K", 0])
            pix_coords = self.project_3d(cam_points, inputs["K", 0], T)  # [b,h,w,2]
            src_img = self.get_color_input(inputs, frame_id, 0)
            outputs[("color", frame_id, scale)] = F.grid_sample(src_img, pix_coords, padding_mode="border",
                                                                align_corners=False)
        return outputs

