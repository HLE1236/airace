#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.mcmc_utils import (
    MCMC_RELOCATION_N_MAX,
    compute_mcmc_growth_target,
    compute_relocation_cuda,
    mcmc_noise_gate,
)

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default", mcmc_init_mode="legacy"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.mcmc_init_mode = str(mcmc_init_mode).lower()
        if self.mcmc_init_mode not in ("paper", "legacy"):
            raise ValueError("mcmc_init_mode must be 'paper' or 'legacy'")
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = None
        self.denom = torch.empty(0)
        self.tmp_radii = None
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        base_state = (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
        # Preserve the original 12-field checkpoint contract for baseline
        # 3DGS and older tooling. Improved-GS appends its AbsGrad accumulator
        # only when that state actually exists.
        if self.xyz_gradient_accum_abs is None:
            return base_state
        return base_state + (self.xyz_gradient_accum_abs,)
    
    def restore(self, model_args, training_args):
        # xyz_gradient_accum_abs was added after the October 2024 Graphdeco
        # checkpoint format. Keep accepting existing checkpoints while saving
        # the optional AbsGS accumulator in new ones.
        if len(model_args) == 12:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            xyz_gradient_accum_abs = None
        elif len(model_args) == 13:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            xyz_gradient_accum_abs) = model_args
        else:
            raise ValueError(f"Unsupported GaussianModel checkpoint format ({len(model_args)} entries)")
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if xyz_gradient_accum_abs is not None:
            self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def _initial_scale_and_opacity(self, dist2):
        """Return activated initialization parameters for the selected mode."""
        if self.mcmc_init_mode == "paper":
            scale_multiplier = 0.1
            initial_opacity = 0.5
        else:
            scale_multiplier = 1.0
            initial_opacity = 0.1
        scales = torch.log(torch.sqrt(dist2) * scale_multiplier)[..., None].repeat(1, 3)
        opacities = self.inverse_opacity_activation(
            initial_opacity
            * torch.ones((dist2.shape[0], 1), dtype=dist2.dtype, device=dist2.device)
        )
        return scales, opacities

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales, opacities = self._initial_scale_and_opacity(dist2)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        density_control = getattr(training_args, "density_control", None)
        requested_absgrad = getattr(
            training_args, "use_absgrad", getattr(training_args, "use_abs_grad", False)
        )
        use_absgrad = bool(requested_absgrad) and (
            density_control is None or str(density_control).lower() == "improvedgs"
        )
        self.xyz_gradient_accum_abs = (
            torch.zeros((self.get_xyz.shape[0], 1), device="cuda") if use_absgrad else None
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, max_opacity=0.01):
        if not 0.0 < max_opacity < 1.0:
            raise ValueError("max_opacity must be strictly between 0 and 1")
        opacities_new = self.inverse_opacity_activation(
            torch.clamp_max(self.get_opacity, max_opacity)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                old_parameter = group['params'][0]
                stored_state = self.optimizer.state.get(old_parameter, None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[old_parameter]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        mask = mask.to(device=self.get_xyz.device, dtype=torch.bool).reshape(-1)
        if mask.shape[0] != self.get_xyz.shape[0]:
            raise ValueError(
                f"Prune mask has {mask.shape[0]} entries for {self.get_xyz.shape[0]} Gaussians"
            )

        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.xyz_gradient_accum is not None and self.xyz_gradient_accum.shape[0] == mask.shape[0]:
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        if self.xyz_gradient_accum_abs is not None and self.xyz_gradient_accum_abs.shape[0] == mask.shape[0]:
            self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        if self.denom is not None and self.denom.shape[0] == mask.shape[0]:
            self.denom = self.denom[valid_points_mask]
        if self.max_radii2D is not None and self.max_radii2D.shape[0] == mask.shape[0]:
            self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.tmp_radii is not None and self.tmp_radii.shape[0] == mask.shape[0]:
            self.tmp_radii = self.tmp_radii[valid_points_mask]

    def only_prune(self, value, percent=False):
        """Prune Gaussians by activated opacity.

        When ``percent`` is false, ``value`` is an absolute opacity threshold.
        When it is true, ``value`` is the fraction of the lowest-opacity
        Gaussians to remove (``20`` is also accepted as shorthand for ``0.20``).
        Percentile pruning removes an exact count, including when opacities tie.
        """
        n_points = self.get_xyz.shape[0]
        if n_points == 0:
            return 0

        opacity = self.get_opacity.detach().squeeze(-1)
        if percent:
            fraction = float(value)
            if fraction > 1.0:
                fraction /= 100.0
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("Percentile prune fraction must be in [0, 1] (or [0, 100])")

            # Keep at least one Gaussian so subsequent rendering/optimization
            # remains well-defined even for an accidentally aggressive value.
            prune_count = min(int(n_points * fraction), max(n_points - 1, 0))
            if prune_count == 0:
                return 0
            prune_indices = torch.topk(opacity, prune_count, largest=False, sorted=False).indices
            prune_mask = torch.zeros(n_points, dtype=torch.bool, device=self.get_xyz.device)
            prune_mask[prune_indices] = True
        else:
            threshold = float(value)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("Absolute opacity prune threshold must be in [0, 1]")
            prune_mask = opacity < threshold
            prune_count = int(prune_mask.sum().item())
            if prune_count == 0:
                return 0

        self.prune_points(prune_mask)
        return prune_count

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        n_existing = self.get_xyz.shape[0]
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.tmp_radii is not None:
            if new_tmp_radii is None:
                new_tmp_radii = torch.zeros(
                    new_xyz.shape[0], dtype=self.tmp_radii.dtype, device=self.tmp_radii.device
                )
            self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        elif new_tmp_radii is not None:
            existing_tmp_radii = torch.zeros(
                n_existing, dtype=new_tmp_radii.dtype, device=new_tmp_radii.device
            )
            self.tmp_radii = torch.cat((existing_tmp_radii, new_tmp_radii))

        device = self.get_xyz.device
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        if self.xyz_gradient_accum_abs is not None:
            self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)

    def _reset_mcmc_optimizer_moments(self, indices):
        """Clear stale Adam moments for Gaussians changed by an MCMC move."""
        indices = torch.as_tensor(
            indices, device=self.get_xyz.device, dtype=torch.long
        ).reshape(-1)
        if indices.numel() == 0 or self.optimizer is None:
            return
        indices = torch.unique(indices)
        for group in self.optimizer.param_groups:
            if len(group["params"]) != 1:
                raise RuntimeError("Each Gaussian optimizer group must own one tensor")
            parameter = group["params"][0]
            state = self.optimizer.state.get(parameter)
            if state is None:
                continue
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = state.get(key)
                if (
                    torch.is_tensor(value)
                    and value.ndim > 0
                    and value.shape[0] == parameter.shape[0]
                ):
                    value[indices] = 0

    def _rebind_mcmc_parameters(self, reset_indices):
        """Rebind every Gaussian Parameter after an in-place relocation.

        This mirrors the reference implementation: gradients produced at the
        pre-relocation positions are discarded, optimizer state is transferred,
        and only donor moments are reset.
        """
        tensors = {
            "xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling": self._scaling,
            "rotation": self._rotation,
        }
        reset_indices = torch.as_tensor(
            reset_indices, device=self.get_xyz.device, dtype=torch.long
        ).reshape(-1)
        reset_indices = torch.unique(reset_indices)
        rebound = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) != 1 or group["name"] not in tensors:
                raise RuntimeError("Unexpected Gaussian optimizer parameter group")
            old_parameter = group["params"][0]
            stored_state = self.optimizer.state.get(old_parameter)
            if stored_state is not None:
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    value = stored_state.get(key)
                    if (
                        torch.is_tensor(value)
                        and value.ndim > 0
                        and value.shape[0] == old_parameter.shape[0]
                    ):
                        value[reset_indices] = 0
                del self.optimizer.state[old_parameter]
            new_parameter = nn.Parameter(tensors[group["name"]].detach().requires_grad_(True))
            group["params"][0] = new_parameter
            if stored_state is not None:
                self.optimizer.state[new_parameter] = stored_state
            rebound[group["name"]] = new_parameter

        self._xyz = rebound["xyz"]
        self._features_dc = rebound["f_dc"]
        self._features_rest = rebound["f_rest"]
        self._opacity = rebound["opacity"]
        self._scaling = rebound["scaling"]
        self._rotation = rebound["rotation"]

    def _sample_mcmc_donors(
        self, probabilities, num_samples, candidate_indices=None, generator=None
    ):
        """Sample donor indices and return their global clone multiplicities."""
        if isinstance(num_samples, bool) or not isinstance(num_samples, int):
            raise TypeError("num_samples must be an integer")
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative")
        point_count = int(self.get_xyz.shape[0])
        probabilities = torch.as_tensor(
            probabilities, device=self.get_xyz.device, dtype=self.get_opacity.dtype
        ).detach().reshape(-1)
        if candidate_indices is None:
            candidate_indices = torch.arange(
                point_count, device=self.get_xyz.device, dtype=torch.long
            )
        else:
            candidate_indices = torch.as_tensor(
                candidate_indices, device=self.get_xyz.device, dtype=torch.long
            ).reshape(-1)
        if probabilities.numel() != candidate_indices.numel():
            raise ValueError("probabilities must match candidate_indices")
        if num_samples == 0:
            return candidate_indices.new_empty((0,)), torch.zeros(
                point_count, dtype=torch.long, device=self.get_xyz.device
            )
        if candidate_indices.numel() == 0:
            raise ValueError("cannot sample an empty donor set")
        if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
            raise ValueError("donor probabilities must be finite and non-negative")
        probability_sum = probabilities.sum()
        if float(probability_sum) <= torch.finfo(probabilities.dtype).eps:
            probabilities = torch.ones_like(probabilities)
            probability_sum = probabilities.sum()
        local_indices = torch.multinomial(
            probabilities / probability_sum,
            num_samples,
            replacement=True,
            generator=generator,
        )
        sampled_indices = candidate_indices[local_indices]
        multiplicity = torch.bincount(
            sampled_indices, minlength=point_count
        ).to(dtype=torch.long)
        return sampled_indices, multiplicity

    def _mcmc_relocated_parameters(
        self,
        donor_indices,
        multiplicity,
        min_opacity=0.005,
        relocation_fn=None,
        n_max=MCMC_RELOCATION_N_MAX,
    ):
        donor_indices = torch.as_tensor(
            donor_indices, device=self.get_xyz.device, dtype=torch.long
        ).reshape(-1)
        multiplicity = torch.as_tensor(
            multiplicity, device=self.get_xyz.device, dtype=torch.long
        ).reshape(-1)
        if multiplicity.numel() != self.get_xyz.shape[0]:
            raise ValueError("multiplicity must contain one value per Gaussian")
        if donor_indices.numel() == 0:
            raise ValueError("at least one donor index is required")
        if relocation_fn is None:
            relocation_fn = compute_relocation_cuda
        donor_multiplicity = (multiplicity[donor_indices] + 1).clamp_(
            min=1, max=n_max - 1
        )
        # sigmoid(float32) can round a large finite logit to exactly one. The
        # relocation limit is well-defined, but the float32 alternating sum in
        # the CUDA kernel becomes ill-conditioned at that endpoint for large
        # multiplicities. Use the nearest representable probability below one.
        donor_opacity = self.get_opacity[donor_indices, 0].clamp(
            max=1.0 - torch.finfo(self.get_opacity.dtype).eps
        )
        new_opacity, new_scaling = relocation_fn(
            donor_opacity,
            self.get_scaling[donor_indices],
            donor_multiplicity,
            n_max=n_max,
        )
        new_opacity = torch.as_tensor(
            new_opacity, device=self.get_xyz.device, dtype=self._opacity.dtype
        ).reshape(-1)
        new_scaling = torch.as_tensor(
            new_scaling, device=self.get_xyz.device, dtype=self._scaling.dtype
        ).reshape(-1, 3)
        epsilon = torch.finfo(new_opacity.dtype).eps
        new_opacity = new_opacity.clamp(min=float(min_opacity), max=1.0 - epsilon)
        new_scaling = new_scaling.clamp_min(torch.finfo(new_scaling.dtype).tiny)
        return (
            self._xyz[donor_indices].clone(),
            self._features_dc[donor_indices].clone(),
            self._features_rest[donor_indices].clone(),
            self.inverse_opacity_activation(new_opacity).unsqueeze(-1),
            self.scaling_inverse_activation(new_scaling),
            self._rotation[donor_indices].clone(),
        )

    @torch.no_grad()
    def relocate_mcmc(
        self,
        dead_mask,
        min_opacity=0.005,
        relocation_fn=None,
        generator=None,
        n_max=MCMC_RELOCATION_N_MAX,
    ):
        """Teleport low-opacity Gaussians to opacity-weighted live donors."""
        point_count = int(self.get_xyz.shape[0])
        dead_mask = torch.as_tensor(
            dead_mask, device=self.get_xyz.device, dtype=torch.bool
        ).reshape(-1)
        if dead_mask.numel() != point_count:
            raise ValueError("dead_mask must contain one value per Gaussian")
        dead_indices = dead_mask.nonzero(as_tuple=False).reshape(-1)
        if dead_indices.numel() == 0:
            return 0
        alive_indices = (~dead_mask).nonzero(as_tuple=False).reshape(-1)
        if alive_indices.numel() == 0:
            return 0
        sampled_indices, multiplicity = self._sample_mcmc_donors(
            self.get_opacity[alive_indices, 0],
            int(dead_indices.numel()),
            candidate_indices=alive_indices,
            generator=generator,
        )
        relocated = self._mcmc_relocated_parameters(
            sampled_indices,
            multiplicity,
            min_opacity=min_opacity,
            relocation_fn=relocation_fn,
            n_max=n_max,
        )
        (
            self._xyz[dead_indices],
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices],
        ) = relocated
        # A donor and all copies produced from it use the same analytically
        # adjusted opacity and scale, preserving its rendered contribution.
        self._opacity[sampled_indices] = relocated[3]
        self._scaling[sampled_indices] = relocated[4]
        # Rebinding clears every stale pre-relocation gradient. Following the
        # official code, Adam moments are reset for sampled donors; relocated
        # slots retain their history but cannot receive the old gradient.
        self._rebind_mcmc_parameters(sampled_indices)
        return int(dead_indices.numel())

    @torch.no_grad()
    def grow_mcmc(
        self,
        cap_max,
        growth_rate=1.05,
        min_opacity=0.005,
        relocation_fn=None,
        generator=None,
        n_max=MCMC_RELOCATION_N_MAX,
    ):
        """Add at most one MCMC growth step without exceeding ``cap_max``."""
        current_count = int(self.get_xyz.shape[0])
        target_count = compute_mcmc_growth_target(
            current_count, int(cap_max), growth_rate
        )
        add_count = max(target_count - current_count, 0)
        if add_count == 0:
            return 0
        sampled_indices, multiplicity = self._sample_mcmc_donors(
            self.get_opacity[:, 0], add_count, generator=generator
        )
        new_parameters = self._mcmc_relocated_parameters(
            sampled_indices,
            multiplicity,
            min_opacity=min_opacity,
            relocation_fn=relocation_fn,
            n_max=n_max,
        )
        self._opacity[sampled_indices] = new_parameters[3]
        self._scaling[sampled_indices] = new_parameters[4]
        self.densification_postfix(*new_parameters, new_tmp_radii=None)
        self._reset_mcmc_optimizer_moments(sampled_indices)
        return int(add_count)

    @torch.no_grad()
    def mcmc_relocate_and_grow(
        self,
        cap_max,
        min_opacity=0.005,
        growth_rate=1.05,
        relocation_fn=None,
        generator=None,
        n_max=MCMC_RELOCATION_N_MAX,
    ):
        """Run one coupled dead-sample relocation and capped growth event."""
        before = int(self.get_xyz.shape[0])
        dead_mask = (self.get_opacity <= float(min_opacity)).squeeze(-1)
        relocated = self.relocate_mcmc(
            dead_mask,
            min_opacity=min_opacity,
            relocation_fn=relocation_fn,
            generator=generator,
            n_max=n_max,
        )
        added = self.grow_mcmc(
            cap_max,
            growth_rate=growth_rate,
            min_opacity=min_opacity,
            relocation_fn=relocation_fn,
            generator=generator,
            n_max=n_max,
        )
        return {
            "before": before,
            "relocated": int(relocated),
            "added": int(added),
            "after": int(self.get_xyz.shape[0]),
            "cap_max": int(cap_max),
        }

    @staticmethod
    def _mcmc_covariance_from_scale_rotation(scaling, rotation):
        """Build covariance on the input device without a full-scene tensor."""
        w, x, y, z = rotation.unbind(dim=1)
        matrix = torch.empty(
            (rotation.shape[0], 3, 3),
            dtype=rotation.dtype,
            device=rotation.device,
        )
        matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
        matrix[:, 0, 1] = 2 * (x * y - w * z)
        matrix[:, 0, 2] = 2 * (x * z + w * y)
        matrix[:, 1, 0] = 2 * (x * y + w * z)
        matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
        matrix[:, 1, 2] = 2 * (y * z - w * x)
        matrix[:, 2, 0] = 2 * (x * z - w * y)
        matrix[:, 2, 1] = 2 * (y * z + w * x)
        matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
        factor = matrix * scaling.unsqueeze(1)
        return factor @ factor.transpose(1, 2)

    @torch.no_grad()
    def add_mcmc_position_noise(
        self,
        noise_lr,
        xyz_lr,
        min_opacity=0.005,
        chunk_size=250_000,
        generator=None,
    ):
        """Apply covariance-shaped SGLD noise to positions in bounded chunks."""
        noise_lr = float(noise_lr)
        xyz_lr = float(xyz_lr)
        if not np.isfinite(noise_lr) or not np.isfinite(xyz_lr):
            raise ValueError("noise_lr and xyz_lr must be finite")
        if noise_lr < 0.0 or xyz_lr < 0.0:
            raise ValueError("noise_lr and xyz_lr must be non-negative")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        point_count = int(self.get_xyz.shape[0])
        if point_count == 0 or noise_lr == 0.0 or xyz_lr == 0.0:
            return point_count
        noise_scale = noise_lr * xyz_lr
        for start in range(0, point_count, chunk_size):
            end = min(start + chunk_size, point_count)
            covariance = self._mcmc_covariance_from_scale_rotation(
                self.get_scaling[start:end], self.get_rotation[start:end]
            )
            gate = mcmc_noise_gate(
                self.get_opacity[start:end], min_opacity=float(min_opacity)
            )
            standard_noise = torch.randn(
                (end - start, 3),
                dtype=self._xyz.dtype,
                device=self._xyz.device,
                generator=generator,
            )
            shaped_noise = torch.bmm(
                covariance, (standard_noise * gate * noise_scale).unsqueeze(-1)
            ).squeeze(-1)
            self._xyz[start:end].add_(shaped_noise)
        return point_count

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = (
            self.tmp_radii[selected_pts_mask].repeat(N) if self.tmp_radii is not None else None
        )

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask] if self.tmp_radii is not None else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def _prepare_densification_scores(self, scores):
        """Return one finite, non-negative score per current Gaussian."""
        n_points = self.get_xyz.shape[0]
        if scores is None:
            accumulator = (
                self.xyz_gradient_accum_abs
                if self.xyz_gradient_accum_abs is not None
                else self.xyz_gradient_accum
            )
            scores = accumulator / torch.clamp_min(self.denom, 1.0)

        scores = torch.as_tensor(scores, device=self.get_xyz.device).detach()
        if scores.ndim == 0:
            scores = scores.reshape(1)
        elif scores.ndim > 1:
            if scores.shape[-1] == 1:
                scores = scores.squeeze(-1)
            else:
                scores = torch.norm(scores, dim=-1)
        scores = scores.reshape(-1)

        padded_scores = torch.zeros(n_points, dtype=scores.dtype, device=self.get_xyz.device)
        copy_count = min(n_points, scores.shape[0])
        padded_scores[:copy_count] = scores[:copy_count]
        return torch.nan_to_num(padded_scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)

    def _weighted_candidate_mask(self, eligible_mask, weights, max_candidates):
        """Sample at most ``max_candidates`` eligible points without replacement."""
        weights = torch.nan_to_num(
            weights.float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min_(0.0)
        # EAS is a selector, not a quota: candidates with no positive edge
        # contribution are not split merely to fill every available slot.
        # When EAS is disabled, callers pass positive gradient magnitudes here.
        eligible_mask = torch.logical_and(eligible_mask, weights > 0)
        candidate_indices = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)
        candidate_count = candidate_indices.numel()
        sample_count = min(max(int(max_candidates), 0), candidate_count)
        selected_mask = torch.zeros_like(eligible_mask, dtype=torch.bool)

        if sample_count == 0:
            return selected_mask
        if sample_count == candidate_count:
            selected_mask[candidate_indices] = True
            return selected_mask

        candidate_weights = weights[candidate_indices]
        sampled_local = torch.multinomial(
            candidate_weights, sample_count, replacement=False
        )
        selected_mask[candidate_indices[sampled_local]] = True
        return selected_mask

    def _long_axis_split(self, selected_pts_mask, split_distance=0.45, opacity_reduction=0.6):
        """Replace each selected Gaussian by the two Improved-GS LAS children."""
        rho = float(split_distance)
        if not 0.0 < rho < 1.0:
            raise ValueError("split_distance must be strictly between 0 and 1")
        opacity_reduction = float(opacity_reduction)
        if not 0.0 < opacity_reduction <= 1.0:
            raise ValueError("opacity_reduction must be in (0, 1]")

        selected_pts_mask = selected_pts_mask.to(
            device=self.get_xyz.device, dtype=torch.bool
        ).reshape(-1)
        n_selected = int(selected_pts_mask.sum().item())
        if n_selected == 0:
            return 0

        parent_xyz = self.get_xyz[selected_pts_mask]
        parent_scaling = self.get_scaling[selected_pts_mask]
        parent_rotation = self._rotation[selected_pts_mask]
        long_axis = torch.argmax(parent_scaling, dim=1, keepdim=True)
        long_scale = torch.gather(parent_scaling, 1, long_axis)

        # The paper parameterizes the child displacement in normalized
        # Gaussian coordinates, hence the factor of 3 before rho * sigma_max.
        local_offset = torch.zeros_like(parent_xyz)
        local_offset.scatter_(1, long_axis, 3.0 * rho * long_scale)
        rotation_matrices = build_rotation(parent_rotation)
        world_offset = torch.bmm(rotation_matrices, local_offset.unsqueeze(-1)).squeeze(-1)
        new_xyz = torch.cat((parent_xyz + world_offset, parent_xyz - world_offset), dim=0)

        short_axis_factor = float(np.sqrt(1.0 - rho * rho))
        child_scaling = parent_scaling * short_axis_factor
        child_scaling.scatter_(1, long_axis, (1.0 - rho) * long_scale)
        child_scaling = child_scaling.repeat(2, 1)
        new_scaling = self.scaling_inverse_activation(child_scaling)

        new_rotation = parent_rotation.repeat(2, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)
        child_opacity = self.get_opacity[selected_pts_mask] * opacity_reduction
        opacity_epsilon = torch.finfo(child_opacity.dtype).eps
        child_opacity = child_opacity.clamp(opacity_epsilon, 1.0 - opacity_epsilon)
        new_opacity = self.inverse_opacity_activation(child_opacity).repeat(2, 1)
        new_tmp_radii = (
            self.tmp_radii[selected_pts_mask].repeat(2) if self.tmp_radii is not None else None
        )

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )

        # Two children replace one parent: each selected point consumes exactly
        # one slot from the active Gaussian budget.
        child_mask = torch.zeros(
            2 * n_selected, dtype=torch.bool, device=self.get_xyz.device
        )
        self.prune_points(torch.cat((selected_pts_mask, child_mask)))
        return n_selected

    def _random_split(self, selected_pts_mask, n_children=2):
        """Apply the original 3DGS stochastic split to a preselected mask."""
        selected_pts_mask = selected_pts_mask.to(
            device=self.get_xyz.device, dtype=torch.bool
        ).reshape(-1)
        n_selected = int(selected_pts_mask.sum().item())
        if n_selected == 0:
            return 0
        if n_children < 2:
            raise ValueError("A split requires at least two children")

        stds = self.get_scaling[selected_pts_mask].repeat(n_children, 1)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rotations = build_rotation(self._rotation[selected_pts_mask]).repeat(
            n_children, 1, 1
        )
        new_xyz = (
            torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[selected_pts_mask].repeat(n_children, 1)
        )
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(n_children, 1)
            / (0.8 * n_children)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(n_children, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(n_children, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(n_children, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(n_children, 1)
        new_tmp_radii = (
            self.tmp_radii[selected_pts_mask].repeat(n_children)
            if self.tmp_radii is not None
            else None
        )

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )
        child_mask = torch.zeros(
            n_children * n_selected, dtype=torch.bool, device=self.get_xyz.device
        )
        self.prune_points(torch.cat((selected_pts_mask, child_mask)))
        return n_selected

    def densify_and_split_improved(
        self,
        grad_values,
        grad_threshold,
        budget,
        split_distance=0.45,
        opacity_reduction=0.6,
        sampling_weights=None,
        eligible_mask=None,
        use_las=True,
    ):
        """Budgeted, weighted Long-Axis Split used by Improved-GS."""
        n_points = self.get_xyz.shape[0]
        budget = int(budget)
        available_slots = max(budget - n_points, 0)
        if available_slots == 0 or n_points == 0:
            return 0

        grad_values = self._prepare_densification_scores(grad_values)
        if eligible_mask is None:
            eligible_mask = grad_values >= float(grad_threshold)
        else:
            eligible_mask = torch.as_tensor(
                eligible_mask, device=self.get_xyz.device, dtype=torch.bool
            ).reshape(-1)
            if eligible_mask.shape[0] != n_points:
                raise ValueError(
                    f"Eligibility mask has {eligible_mask.shape[0]} entries for {n_points} Gaussians"
                )
            eligible_mask = torch.logical_and(
                eligible_mask, grad_values >= float(grad_threshold)
            )

        weights = (
            grad_values
            if sampling_weights is None
            else self._prepare_densification_scores(sampling_weights)
        )
        selected_pts_mask = self._weighted_candidate_mask(
            eligible_mask, weights, available_slots
        )
        if use_las:
            return self._long_axis_split(
                selected_pts_mask,
                split_distance=split_distance,
                opacity_reduction=opacity_reduction,
            )
        return self._random_split(selected_pts_mask, n_children=2)

    def densify_and_prune_improved(self, scores, min_opacity, budget, opt, iteration, extent):
        """Run Improved-GS LAS under a hard budget, then opacity pruning.

        ``iteration`` and ``extent`` intentionally remain in this interface so
        the training loop can share its density-control call site with 3DGS;
        the new LAS rule itself does not use scene extent.
        """
        del extent
        before = self.get_xyz.shape[0]
        use_absgrad = getattr(opt, "use_absgrad", getattr(opt, "use_abs_grad", False))
        if use_absgrad:
            if self.xyz_gradient_accum_abs is None:
                raise RuntimeError(
                    "use_absgrad is enabled but no absolute-gradient statistics were accumulated"
                )
            gradient_accumulator = self.xyz_gradient_accum_abs
        else:
            gradient_accumulator = self.xyz_gradient_accum
        grad_values = gradient_accumulator / torch.clamp_min(self.denom, 1.0)

        grad_threshold = getattr(opt, "improvedgs_grad_threshold", 0.0003)
        split_distance = getattr(opt, "split_distance", 0.45)
        opacity_reduction = getattr(opt, "opacity_reduction", 0.6)
        use_las = getattr(opt, "use_las", True)

        split_count = self.densify_and_split_improved(
            grad_values=grad_values,
            grad_threshold=grad_threshold,
            budget=budget,
            split_distance=split_distance,
            opacity_reduction=opacity_reduction,
            sampling_weights=scores,
            use_las=use_las,
        )
        pruned_count = self.only_prune(min_opacity, percent=False)
        self.tmp_radii = None
        torch.cuda.empty_cache()

        return {
            "iteration": int(iteration),
            "before": int(before),
            "split": int(split_count),
            "pruned": int(pruned_count),
            "after": int(self.get_xyz.shape[0]),
            "budget": int(budget),
        }

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_densification_stats_pixelgs(
        self,
        viewspace_point_tensor,
        update_filter,
        pixel_counts,
        depth_scale,
    ):
        """Accumulate Pixel-GS' pixel-aware, depth-scaled gradient score.

        ``pixel_counts`` is the number of image pixels to which each Gaussian
        contributed in the current view.  It weights both the numerator and
        denominator, while ``depth_scale`` only attenuates the gradient in the
        numerator for Gaussians close to the camera.
        """
        if viewspace_point_tensor.grad is None:
            raise RuntimeError("View-space point gradients are unavailable; call after backward()")

        num_gaussians = int(self.xyz_gradient_accum.shape[0])
        device = self.xyz_gradient_accum.device
        dtype = self.xyz_gradient_accum.dtype

        counts = torch.as_tensor(pixel_counts, device=device, dtype=dtype).detach().reshape(-1)
        scales = torch.as_tensor(depth_scale, device=device, dtype=dtype).detach().reshape(-1)
        if counts.numel() != num_gaussians:
            raise ValueError(
                "pixel_counts must contain one value per Gaussian ({} != {})".format(
                    counts.numel(), num_gaussians
                )
            )
        if scales.numel() != num_gaussians:
            raise ValueError(
                "depth_scale must contain one value per Gaussian ({} != {})".format(
                    scales.numel(), num_gaussians
                )
            )
        if update_filter.dtype == torch.bool:
            visible_mask = update_filter.reshape(-1).to(device=device)
            if visible_mask.numel() != num_gaussians:
                raise ValueError(
                    "Boolean visibility mask must contain one value per Gaussian"
                )
            visible_indices = visible_mask.nonzero(as_tuple=False).reshape(-1)
        else:
            visible_indices = update_filter.reshape(-1).to(device=device, dtype=torch.long)
        if visible_indices.numel() == 0:
            return

        # The rasterizer guarantees finite, non-negative counts and valid,
        # unique visibility indices.  Avoid whole-N validation scans here: at
        # HCM0204 scale they would synchronize the GPU on every training view.
        visible_indices = visible_indices[counts[visible_indices] > 0]
        gradients = viewspace_point_tensor.grad.to(device=device)
        signed_gradient_norm = torch.norm(
            gradients[visible_indices, :2], dim=-1, keepdim=True
        ).to(dtype=dtype)
        count_weights = counts[visible_indices, None]
        self.xyz_gradient_accum[visible_indices] += (
            signed_gradient_norm * count_weights * scales[visible_indices, None]
        )
        self.denom[visible_indices] += count_weights

    def add_densification_stats_abs(self, viewspace_point_tensor, update_filter):
        """Accumulate both signed and per-pixel absolute screen-space gradients.

        The modified Improved-GS rasterizer stores the conventional signed
        gradient in channels 0:2 and AbsGS' accumulated absolute gradient in
        channels 2:4. Call this instead of ``add_densification_stats`` (not in
        addition to it), because this method increments the shared denominator.
        """
        if viewspace_point_tensor.grad is None:
            raise RuntimeError("View-space point gradients are unavailable; call after backward()")
        if viewspace_point_tensor.grad.shape[-1] < 4:
            raise RuntimeError(
                "AbsGrad requires an Improved-GS rasterizer that returns four "
                "view-space gradient channels"
            )
        if (
            self.xyz_gradient_accum_abs is None
            or self.xyz_gradient_accum_abs.shape[0] != self.get_xyz.shape[0]
        ):
            self.xyz_gradient_accum_abs = torch.zeros(
                (self.get_xyz.shape[0], 1), device=self.get_xyz.device
            )

        gradients = viewspace_point_tensor.grad
        self.xyz_gradient_accum[update_filter] += torch.norm(
            gradients[update_filter, :2], dim=-1, keepdim=True
        )
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(
            gradients[update_filter, 2:4], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
