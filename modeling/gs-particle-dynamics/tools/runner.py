import os
import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from pytorch3d.structures import Pointclouds
from gsplat.rendering import rasterization


class Model_Runner(object):

    def __init__(self, models, criterion, optimizer, cfg):

        self.pointconv_internaction_network = models[0]
        self.predictor = models[1]
        assert(len(models) == 2)
        self.criterion = criterion
        self.optimizer = optimizer
        self.cfg = cfg
        self.all_params = []
        for model in models:
            self.all_params += list(model.parameters())

    def put_models_in_eval_mode(self):
        self.pointconv_internaction_network.eval()
        self.predictor.eval()

    def put_models_in_train_mode(self):
        self.pointconv_internaction_network.train()
        self.predictor.train()

    def store_checkpoint(self, cur_epoch):

        exp_folder = os.path.join(self.cfg['exp_dir'], self.cfg['exp_name'])
        out_folder = os.path.join(exp_folder, 'epoch_%02d' % cur_epoch)
        out_file_path = os.path.join(out_folder, 'model.pt')
        if not os.path.isdir(exp_folder):
            os.mkdir(exp_folder)
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)

        torch.save(
            {'epoch': cur_epoch,
             'pointconv_internaction_network': self.pointconv_internaction_network.state_dict(),                
             'predictor': self.predictor.state_dict(),
             'optim_state_dict': self.optimizer.state_dict()},
            out_file_path
        )

    def optimize(self, loss):
                        
        loss.backward()   
        self.optimizer.step()
        self.optimizer.zero_grad()        

    def compute_loss(self, batch_all, phase):

        pred_log = {}
        pred = None
        seq_len = self._get_seq_len(phase)
        single_frame_input = {}
        idx_info = self._get_idx_info(batch_all) # doesn't change over time
        gt_info = batch_all['gt']
        gt_lookahead_info = batch_all['gt_lookahead']
        lookahead_frame_num = len(gt_lookahead_info[0])
        lookahead_frame_ct = 0

        # Keep all loss accumulators as tensors even when one loss type is
        # disabled. This lets position-only training coexist with optional hard
        # example mining, which expects per-point and per-scene tensors.
        reference_tensor = batch_all['velocity'][0]
        loss = reference_tensor.new_zeros(())
        point_loss = reference_tensor.new_zeros(sum(idx_info['batch_idx'][0]))
        frame_loss = reference_tensor.new_zeros(len(idx_info['batch_idx'][0]))

        for curr_frame in range(0, seq_len - 1):
            assert(curr_frame not in single_frame_input.keys())
            curr_batch = self._get_batch(pred, batch_all, curr_frame)
            single_frame_input[curr_frame] = curr_batch
            # skip if the number of input frames is fewer than a required number
            if curr_frame < self.cfg['model']['input_frame_num'] - 1: continue

            # get multi-frame frame prediction   
            multi_frame_input = []
            for j in range(self.cfg['model']['input_frame_num']):
                multi_frame_input.append(single_frame_input[curr_frame - j])            
            # get prediction
            multi_frame_feats = self._run_pointconv_internaction_networks(multi_frame_input, idx_info)
            out = self._run_prediction_heads(multi_frame_feats)
            pred = self._collect_pointwise_pred_and_gt(out, curr_batch, curr_frame, phase)

            pos_loss_weight = 1.0            
            rend_loss_weight = 3.0
            gt_pos_idx = curr_frame - (self.cfg['model']['input_frame_num'] - 1)
            if phase == "train" or phase == "train_hem":
                if self._is_gt_frame(curr_frame):
                    assert gt_pos_idx == 0
                    if self.cfg['train']['loss_type'] == 'all' or self.cfg['train']['loss_type'] == 'position':
                        point_loss_avg_curr, point_loss_curr, _ = self._compute_position_loss(pred, gt_info ,gt_pos_idx)
                        loss = loss + pos_loss_weight * point_loss_avg_curr                    
                        point_loss = point_loss + pos_loss_weight * point_loss_curr
                    if self.cfg['train']['loss_type'] == 'all' or self.cfg['train']['loss_type'] == 'rendering':
                        frame_loss_avg_curr, frame_loss_curr = self._compute_rendering_loss(pred, idx_info, gt_info, -1)
                        loss = loss + rend_loss_weight * frame_loss_avg_curr
                        frame_loss = frame_loss + rend_loss_weight * frame_loss_curr
                else:
                    if self.cfg['train']['loss_type'] == 'all' or self.cfg['train']['loss_type'] == 'position':
                        point_loss_avg_curr, point_loss_curr, _ = self._compute_position_loss(pred, gt_info ,gt_pos_idx)
                        loss = loss + pos_loss_weight * point_loss_avg_curr
                        point_loss = point_loss + pos_loss_weight * point_loss_curr
                    if self.cfg['train']['loss_type'] == 'all' or self.cfg['train']['loss_type'] == 'rendering':
                        frame_loss_avg_curr, frame_loss_curr = self._compute_rendering_loss(pred, idx_info, gt_lookahead_info, lookahead_frame_ct)
                        loss = loss + rend_loss_weight * frame_loss_avg_curr
                        frame_loss = frame_loss + rend_loss_weight * frame_loss_curr
                    lookahead_frame_ct += 1
            elif phase == "test":          
                # loss_curr, loss_all, topk_inds = self._compute_position_loss_test(pred, gt_info)   
                # _ = self._compute_rendering_loss_test(pred, vars, idx_info, cam_info, gt_info, topk_inds, loss_all)   
                self._update_pred_log_using_gs_format(pred, pred_log, gt_info, gt_pos_idx)                
                if gt_pos_idx == len(gt_info[0]['position']) - 1: break # stop inference when the rollout frame reaches the end of GT frames.
            else:
                raise NotImplementedError("phase should be one of train, train_hem, test")
        
        # average over predicted frames
        actual_pred_frame_num = seq_len - 1 - (self.cfg['model']['input_frame_num'] - 1)
        loss = loss / actual_pred_frame_num
        point_loss = point_loss / actual_pred_frame_num
        frame_loss = frame_loss / actual_pred_frame_num

        assert phase == 'test' or gt_pos_idx == len(gt_info[0]['position']) - 1
        assert phase == 'test' or lookahead_frame_num == lookahead_frame_ct
        assert phase == 'test' or actual_pred_frame_num == (1 + lookahead_frame_ct)

        if phase == "test":
            self._store_pred_log(pred_log, batch_all['seq_name'])

        return loss, point_loss, frame_loss

    def _compute_position_loss(self, pred, gt_info, gt_pos_idx):

        all_gt = []
        for i in range(len(gt_info)):
            all_gt.append(gt_info[i]['position'][gt_pos_idx])
        gt_position = torch.cat(all_gt, dim=0)        
        pred_position = pred['pc'][0].points_packed()

        loss_points = self.criterion(pred_position, gt_position)
        # top_k = int(loss_points.shape[0]*1.0)
        # top_k_out = torch.topk(torch.sum(loss_points, dim=-1), k=top_k)
        # loss = torch.mean(top_k_out.values)
        point_loss = torch.sum(loss_points, dim=-1)
        loss = torch.mean(point_loss)

        # return loss, torch.sum(loss_points, dim=-1), top_k_out.indices
        return loss, point_loss, None

    def _compute_position_loss_test(self, pred, gt_info):

        all_gt = []
        for i in range(len(gt_info)):
            all_gt.append(gt_info[i]['position'])
        gt_position = torch.cat(all_gt, dim=0)        
        pred_position = pred['pc'][0].points_packed()

        error = (pred_position - gt_position).abs().detach()
        weight = (error + 1e-8).pow(1.0)
        weight = weight/ (weight.mean() + 1e-8)

        loss_points = self.criterion(pred_position, gt_position)
        top_k = int(pred_position.shape[0] * 0.025)        
        top_k_out = torch.topk(torch.sum(loss_points, dim=-1), k=top_k)
        loss = torch.mean(top_k_out.values)

        return loss, torch.sum(loss_points, dim=-1), top_k_out.indices

    def _compute_rendering_loss(self, pred, idx_info, gt_info, lookahead_frame_ct):
        
        num_pts_per_batch = idx_info['batch_idx'][0] # densest level
        batch_num = len(num_pts_per_batch)
        assert sum(num_pts_per_batch) == pred['pc'][0].points_packed().shape[0]        
        assert len(gt_info) == batch_num
        # Rendering uses only the first 59 placeholder Gaussian attributes.
        # Additional channels are allowed, but object-assignment channels are
        # no longer required by either the renderer or the dynamics backbone.
        assert pred['pc'][0].features_packed().shape[1] >= 59
            
        st = 0
        loss = 0.0
        loss_list = []
        for i in range(batch_num):
            ed = st + num_pts_per_batch[i]

            params = {
                'means3D': pred['pc'][0].points_packed()[st:ed],
                'shs': pred['pc'][0].features_packed()[st:ed, 11:59].reshape(-1, 16, 3),
                'seg_colors': pred['pc'][0].features_packed()[st:ed, 8:11],
                'unnorm_rotations': pred['pc'][0].features_packed()[st:ed, :4],
                'logit_opacities': torch.unsqueeze(pred['pc'][0].features_packed()[st:ed, 4], dim=1),
                'log_scales': pred['pc'][0].features_packed()[st:ed, 5:8],
            }

            # TODO: batch rendering -- now, just using a for loop
            if lookahead_frame_ct != -1:
                loss_curr = self._get_loss(params, gt_info[i][lookahead_frame_ct])
            else:
                loss_curr = self._get_loss(params, gt_info[i])
            
            loss_list.append(loss_curr)
            loss += loss_curr
            st = ed        

        return loss / batch_num, torch.stack(loss_list, dim=0)

    def _update_pred_log_using_gs_format(self, pred, pred_log, gt_info, gt_pos_idx):

        # # for pseudo-GT visualization
        # all_gt_pos = []
        # all_gt_rot = []
        # for i in range(len(gt_info)):
        #     all_gt_pos.append(gt_info[i]['position'][gt_pos_idx])
        #     all_gt_rot.append(gt_info[i]['rotation'][gt_pos_idx])
        # gt_position = torch.cat(all_gt_pos, dim=0) 
        # gt_rotation = torch.cat(all_gt_rot, dim=0)

        pc = pred['pc'][0]
        # go back to the original scale of the pointcloud
        pos = pc.points_packed().clone().detach().cpu().numpy() * (1 / self.cfg['pointcloud']['scaling'])
        # pos = gt_position.clone().detach().cpu().numpy() * (1 / self.cfg['pointcloud']['scaling'])
        feats = pc.features_packed().clone().detach().cpu().numpy()

        if len(pred_log.keys()) == 0:
            pred_log['means3D'] = []
            pred_log['shs'] = []
            pred_log['unnorm_rotations'] = []
            pred_log['seg_colors'] = feats[:, 8:11]
            pred_log['logit_opacities'] = feats[:, [4]]
            pred_log['log_scales'] = feats[:, 5:8]

        pred_log['means3D'].append(pos)
        pred_log['shs'].append(feats[:, 11:59].reshape(-1, 16, 3))
        pred_log['unnorm_rotations'].append(feats[:, :4])
        # pred_log['unnorm_rotations'].append(gt_rotation.clone().detach().cpu().numpy())


    def _store_pred_log(self, pred_log, seq_name):

        pred_log['means3D'] = np.stack(pred_log['means3D'])
        pred_log['shs'] = np.stack(pred_log['shs'])
        pred_log['unnorm_rotations'] = np.stack(pred_log['unnorm_rotations'])
        
        output_dir = os.path.join(self.cfg['output_dir'], self.cfg['exp_name_epoch'])
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        assert len(seq_name) == 1
        out_scene_dir = os.path.join(output_dir, seq_name[0])
        if not os.path.isdir(out_scene_dir):
            os.mkdir(out_scene_dir)
        np.savez(f"{out_scene_dir}/params", **pred_log)


    def _get_loss(self, params, curr_data):

        scales = torch.exp(params['log_scales'])
        opacities = torch.squeeze(torch.sigmoid(params['logit_opacities']), axis=-1)

        render_colors, render_alphas, info = rasterization(
            means=params['means3D'] * (1.0 /self.cfg['pointcloud']['scaling']), # back to the original scale for rendering
            quats=params['unnorm_rotations'],
            scales=scales,
            opacities=opacities,
            colors=params['shs'],
            viewmats=curr_data['w2c'],  # [C, 4, 4]
            Ks=curr_data['k'],  # [C, 3, 3]
            width=curr_data['w'],
            height=curr_data['h'],
            sh_degree=3,
            render_mode='RGB+ED'
        )

        # render_depth = torch.squeeze(render_colors[..., 3:4], dim=-1)
        render_colors = render_colors[..., 0:3]

        # first channel encodes the foreground
        valid_mask = curr_data['seg'][:, [0]].float()

        render_colors = render_colors.permute(0, 3, 1, 2)
        predicted_colors_masked = render_colors
        gt_colors_masked = torch.mul(curr_data['im'], valid_mask)
        l1loss = F.l1_loss(predicted_colors_masked, gt_colors_masked, reduction='mean')

        return l1loss

    def _get_seq_len(self, phase):

        return self.cfg[phase]['seq_len']

    def _get_batch(self, pred, batch, curr_frame):

        if self._is_gt_frame(curr_frame):
            curr_batch = self._get_input_data_from_gt(batch, curr_frame)        
        else:
            curr_batch = self._get_input_data_from_prediction(pred, batch, curr_frame)

        return curr_batch

    def _is_gt_frame(self, curr_frame):

        input_frame_num = self.cfg['model']['input_frame_num']

        return curr_frame < input_frame_num 

    def _get_input_data_from_gt(self, batch, curr_frame):

        curr_pc = batch['pc'][curr_frame]
        curr_pc_grid_idx = batch['grid_idx'][curr_frame]
        curr_velocity = batch['velocity'][curr_frame] # curr_pc - prev_pc
        
        data = {       
            'curr_pc': curr_pc, # input
            'curr_pc_grid_idx': curr_pc_grid_idx,
            'curr_velocity': curr_velocity, # input
        }

        return data

    def _get_input_data_from_prediction(self, prediction, batch, curr_frame):

        curr_pc = prediction['pc']
        grid_frame = self.cfg['model']['input_frame_num'] - 1
        curr_pc_grid_idx = batch['grid_idx'][grid_frame] # grid_idx doesn't have to come from prediction
        curr_velocity = prediction['velocity'] # curr_pc - prev_pc

        data = {       
            'curr_pc': curr_pc, # input
            'curr_pc_grid_idx': curr_pc_grid_idx,
            'curr_velocity': curr_velocity, # input
        }

        return data

    def _run_pointconv_internaction_networks(self, multi_frame_input, idx_info):
        
        recent_frame = 0
        velocity = []        
        for frame in range(self.cfg['model']['input_frame_num']):
            velocity.append(multi_frame_input[frame]['curr_velocity'])
        # Concatenate D-dimensional velocities from all input frames.
        velocity = torch.cat(velocity, dim=1)
        pc = []

        for l in range(len(multi_frame_input[recent_frame]['curr_pc'])):
            pc.append(multi_frame_input[recent_frame]['curr_pc'][l].points_packed())

        # Placeholder Gaussian attributes remain attached to Pointclouds for
        # rendering/logging, but the assignment-free dynamics backbone consumes
        # only motion features, point positions, and geometric neighborhoods.
        pc_feat = self.pointconv_internaction_network(velocity, pc, idx_info)

        return pc_feat

    def _run_prediction_heads(self, feats):

        return self.predictor(feats)

    def _update_gaussians(self, curr_pos, curr_feat, motion_pred, curr_vel):
        """Advance every Gaussian independently without fitting a rigid pose."""

        # Step 1: interpret the network output. By default it is the next
        # planar displacement. In acceleration mode, promote the planar
        # acceleration to XYZ before adding the previous XYZ displacement.
        if self.cfg["model"]["acceleration_prediction"]:
            zero_z_acceleration = torch.zeros_like(curr_vel[:, -1:])
            acceleration = torch.cat(
                (motion_pred, zero_z_acceleration),
                dim=1,
            )
            displacement = acceleration + curr_vel
            next_pos = curr_pos + displacement
        else:
            displacement = motion_pred
            next_pos = torch.cat(
                (
                    curr_pos[:, :2] + displacement,
                    curr_pos[:, 2:3],  # preserve z exactly
                ),
                dim=1,
            )

        # Step 2: apply the displacement point by point. No object-level
        # transform is estimated, so Gaussians in one object may move differently.

        # Step 3: this version predicts translation only. Preserve every
        # placeholder Gaussian attribute exactly as it was supplied.
        next_feat = curr_feat

        # Step 4: feed the realized displacement into the next rollout step.
        next_pos_vel = next_pos - curr_pos

        return next_pos, next_feat, next_pos_vel

    def _collect_pointwise_pred_and_gt(self, network_output, batch, frame, phase):

        curr_pc_pred = batch['curr_pc']
        curr_velocity = batch['curr_velocity']
        curr_pc_grid_idx = batch['curr_pc_grid_idx']

        spatial_dim = self.cfg['pointcloud']['input_dim']
        motion_dim = self.cfg["model"]["output_dim"]
        assert network_output.shape[1] == motion_dim
        assert curr_velocity.shape[1] == spatial_dim

        curr_motion_vel = curr_velocity[:, :motion_dim]

        if phase == "train" or phase == "train_hem":
            assert self._get_seq_len(phase) >= self.cfg['model']['input_frame_num'] + 1
        elif phase == "test":
            # Testing currently supports one scene at a time.
            assert len(curr_pc_pred[0].points_list()) == 1, (
                "Testing with more than one point cloud is not supported."
            )
        else:
            raise ValueError("phase should be one of train, train_hem, test")

        next_pc_all_layers = []
        for layer_idx, curr_layer_pc in enumerate(curr_pc_pred):
            idx_sel = curr_pc_grid_idx[layer_idx]

            if layer_idx == 0:
                # The prediction head operates at the densest level and emits
                # a separate translation (or acceleration) for every Gaussian.
                curr_pos = curr_layer_pc.points_packed()
                curr_feat = curr_layer_pc.features_packed()
                motion_pred = network_output[idx_sel]
                curr_pos_vel = curr_velocity[idx_sel, :spatial_dim]

                assert curr_pos.shape[1] == spatial_dim
                assert motion_pred.shape[0] == curr_pos.shape[0]
                assert motion_pred.shape[1] == motion_dim
                assert curr_pos_vel.shape == curr_pos.shape
                assert torch.isfinite(motion_pred).all(), (
                    "Per-Gaussian motion predictions must be finite."
                )

                next_pos, next_feat, next_pos_vel = self._update_gaussians(
                    curr_pos, curr_feat, motion_pred, curr_pos_vel
                )
            else:
                # Coarser pyramid levels reuse the independently updated dense
                # positions and features at their precomputed grid indices.
                next_pos = next_pc_all_layers[0].points_packed()[idx_sel]
                next_feat = next_pc_all_layers[0].features_packed()[idx_sel]

            # Restore the packed tensors to the per-scene layout expected by
            # PyTorch3D and by the following rollout step.
            point_counts = curr_layer_pc.num_points_per_cloud().tolist()
            next_pos_list = list(torch.split(next_pos, point_counts, dim=0))
            next_feat_list = list(torch.split(next_feat, point_counts, dim=0))
            next_layer_pc = Pointclouds(
                points=next_pos_list,
                features=next_feat_list,
            )
            next_pc_all_layers.append(next_layer_pc)

        pred = {}
        # The next rollout consumes positional velocity directly; no height or
        # gravity-axis feature is appended.
        pred['velocity'] = next_pos_vel
        pred['pc'] = next_pc_all_layers

        return pred

    def _get_idx_info(self, batch):

        idx = {}
        idx['nn_idx_forward'] = []
        idx['nn_idx_propagate'] = []
        idx['batch_idx'] = []
        for l in sorted(batch['nn_idx_self'].keys()):
            idx['batch_idx'].append(
                    batch['batch_idx'][l].tolist()
            )
            if l != max(batch['nn_idx_self'].keys()):
                idx['nn_idx_forward'].append(
                    batch['nn_idx_forward'][l].features_packed().to(dtype=torch.long)
                )
                idx['nn_idx_propagate'].append(
                    batch['nn_idx_propagate'][l].features_packed().to(dtype=torch.long)
                )
        
        return idx

    def _get_cam_info(self, batch):

        return {'cam_m': batch['cam_m'], 'cam_c': batch['cam_c']}

