import torch
import numpy as np
import os
from pytorch3d.structures import Pointclouds
from sklearn.neighbors import KDTree
from externals.pointconvformer.util.voxelize import voxelize


def find_knns(pc, nsamples):

    tree = KDTree(pc)
    if pc.shape[0] >= nsamples:
        nn_idx = tree.query(pc, k=nsamples, return_distance=False)
    else:
        knn_reduced_size = pc.shape[0]
        nn_idx = tree.query(pc, k=knn_reduced_size, return_distance=False)
        dummy_num = (nsamples - knn_reduced_size)*knn_reduced_size
        dummy_nn_idx = np.random.choice(knn_reduced_size, dummy_num)
        dummy_nn_idx = dummy_nn_idx.reshape(pc.shape[0], -1)
        nn_idx= np.concatenate([nn_idx, dummy_nn_idx], axis=1)
    
    return nn_idx


def find_knns_across_levels(pc, feat, nsamples_min, knn, knn_k_decay_factor, grid_size, scaling):

    # grid downsample
    num_points = pc.shape[0]
    idx = np.arange(num_points)
    idx_sel = torch.tensor(idx, dtype=torch.long)
    pc_sel = torch.tensor(pc, dtype=torch.float)
    feat_sel = torch.tensor(feat, dtype=torch.float)
    pc_all = [pc_sel]
    idx_all = [idx_sel]
    feat_all = [feat_sel]

    for i in range(len(grid_size)):
        assert nsamples_min >= 1, (
            "Too many downsampling layers for the current knn number or decay factor. "
            "Either increase the initial knn number, decrease the decay factor, "
            "or reduce the number of downsampling layers."
        )
        idx = voxelize(pc, scaling * grid_size[i])
        if idx.shape[0] < nsamples_min:
            raise NotImplementedError("Too few points after downsampling.")
        nsamples_min = int(nsamples_min // knn_k_decay_factor)
        pc_sel = pc[idx]
        feat_sel = feat[idx]
        idx_sel = torch.tensor(idx, dtype=torch.long)
        pc_sel = torch.tensor(pc_sel, dtype=torch.float)
        feat_sel = torch.tensor(feat_sel, dtype=torch.float)
        idx_all.append(idx_sel)
        pc_all.append(pc_sel)
        feat_all.append(feat_sel)

    # compute knns between downsampled points clouds
    assert len(grid_size) == (len(pc_all) - 1)
    k_curr = knn
    k_all = []
    nn_idx_self = []
    for i in range(len(pc_all)):
        k_curr = k_curr if i == 0 else int(k_curr // knn_k_decay_factor)
        assert k_curr >= 1
        tree = KDTree(pc_all[i])
        nn_idx = tree.query(pc_all[i], k=k_curr, return_distance=False)
        nn_idx_self.append(torch.tensor(nn_idx, dtype=torch.int32))
        k_all.append(k_curr)
    
    # edge forward
    assert len(k_all) == len(pc_all)
    nn_idx_foward = []
    for i in range(1, len(pc_all)):
        tree = KDTree(pc_all[i - 1])
        nn_idx = tree.query(pc_all[i], k=k_all[i], return_distance=False)
        nn_idx_foward.append(torch.tensor(nn_idx, dtype=torch.int32))
    
    # edge propagate
    nn_idx_propagate = []
    for i in range(len(pc_all) - 1):
        tree = KDTree(pc_all[i + 1])
        nn_idx = tree.query(pc_all[i], k=k_all[i], return_distance=False)
        nn_idx_propagate.append(torch.tensor(nn_idx, dtype=torch.int32))

    nn_idx_all = {}
    nn_idx_all['self'] = nn_idx_self
    nn_idx_all['forward'] = nn_idx_foward
    nn_idx_all['propagate'] = nn_idx_propagate

    return pc_all, feat_all, idx_all, nn_idx_all


class Dataset(torch.utils.data.Dataset):

    def __init__(self, phase, cfg, hard_examples=None, hard_examples_frames=None):

        self.cfg = cfg
        self.phase = phase
        self.start_timestep = 0
        self.all_trials = []
        self.n_rollout = 0
        self.lookahead_frames = 0 if phase == "test" else cfg[phase]['seq_len'] - (cfg['model']['input_frame_num'] + 1)
        self.hard_examples = hard_examples
        self.hard_examples_frames = hard_examples_frames
        assert self.lookahead_frames >= 0, "Lookahead frames should be non-negative"

        # Only supports position loss now
        
        if phase == "train_hem":
            self.data_dir = os.path.join(cfg['dataset']['data_dir'], cfg['dataset']['scenario'], "train")
        else:
            self.data_dir = os.path.join(cfg['dataset']['data_dir'], cfg['dataset']['scenario'], phase)

        # Preserve exclusions from the published datasets, while allowing new
        # scenario and scene names which do not encode a numeric release ID.
        skip_lists = {
            'bowling': {13, 48, 50, 127, 128, 130, 141},
            'cube_stacks': {
                36, 72, 74, 75, 78, 81, 82, 83, 105, 106, 124, 130,
                131, 133, 161, 162, 163, 174, 191, 193, 208, 234,
            },
        }
        skip_list = skip_lists.get(cfg['dataset']['scenario'], set())

        trial_names_all = sorted(os.listdir(self.data_dir))
        scene_folders = []
        for trial_name in trial_names_all:
            scene_folder = os.path.join(self.data_dir, trial_name, 'gs')
            if not os.path.isdir(scene_folder):
                continue
            try:
                scene_number = int(trial_name.rsplit('_', 1)[1])
            except (IndexError, ValueError):
                scene_number = None
            if scene_number not in skip_list:
                scene_folders.append(scene_folder)

        trial_names = []
        for scene_folder in scene_folders:
            frame_numbers = self._get_frame_numbers(scene_folder)
            required_frames = self._required_position_frame_count(phase)
            if len(frame_numbers) < required_frames:
                print(
                    os.path.basename(os.path.dirname(scene_folder))
                    + " doesn't have enough point-cloud frames, skipping..."
                )
                continue
            trial_names.append(os.path.basename(os.path.dirname(scene_folder)))

        if phase == "train" or phase == "test":
            n_trials = len(trial_names)
            self.all_trials += [os.path.join(self.data_dir, trial_name, "gs") for trial_name in trial_names]
            self.n_rollout += n_trials
            self.mean_time_step = self.cfg[phase]['frames_per_scene'] if phase == "train" else 1
        elif phase == "train_hem":
            # HEM enumerates every start with a complete training window.
            scene_folders = [os.path.join(self.data_dir, trial_name, "gs") for trial_name in trial_names]
            for scene_folder in scene_folders:
                frame_numbers = self._get_frame_numbers(scene_folder)
                frame_paths = self._get_frame_paths(scene_folder)
                valid_start_count = len(frame_numbers) - self._required_position_frame_count(phase) + 1
                self.all_trials += [
                    os.path.join(scene_folder, frame_paths[frame_number])
                    for frame_number in frame_numbers[:valid_start_count]
                ]
        else:
            raise NotImplementedError("phase should be one of train, train_hem, test")

        if len(self.all_trials) == 0:
            raise ValueError(f"No usable {phase} sequences were found in {self.data_dir}.")

    def __len__(self):

        length = len(self.all_trials) if self.phase == "train_hem" else self.n_rollout * self.mean_time_step

        return length

    def __getitem__(self, idx):
        if self.phase in ['train', 'test']:
            idx = idx % self.n_rollout
            trial_dir = self.all_trials[idx]
            trial_fullname = os.path.basename(os.path.dirname(trial_dir))
            frame_list = self._get_frame_numbers(trial_dir)

            if self.phase == 'train' and self.hard_examples:
                losses = self.hard_examples[trial_fullname]
                start_frames = self.hard_examples_frames[trial_fullname]
                loss_sum = sum(losses)
                if loss_sum <= 0:
                    sampling_weights = None
                else:
                    sampling_weights = [loss / loss_sum for loss in losses]
                timestep_sel = int(np.random.choice(start_frames, p=sampling_weights))
                if timestep_sel not in frame_list:
                    raise ValueError(
                        f"Hard-example frame {timestep_sel} is missing from {trial_fullname}."
                    )
                start_idx = frame_list.index(timestep_sel)
            elif self.phase == 'train':
                window_size = self._required_position_frame_count(self.phase)
                start_idx = int(np.random.randint(0, len(frame_list) - window_size + 1))
                timestep_sel = frame_list[start_idx]
            else:
                start_idx = 0
                timestep_sel = frame_list[0]
        else:
            data_path = self.all_trials[idx]
            trial_dir = os.path.dirname(data_path)
            trial_fullname = os.path.basename(os.path.dirname(trial_dir))
            frame_list = self._get_frame_numbers(trial_dir)
            timestep_sel = int(os.path.basename(data_path))
            start_idx = frame_list.index(timestep_sel)

        if self.phase == 'test':
            selected_frames = frame_list[start_idx:]
        else:
            window_size = self._required_position_frame_count(self.phase)
            selected_frames = frame_list[start_idx:start_idx + window_size]
            if len(selected_frames) != window_size:
                raise ValueError(
                    f"Incomplete point-cloud window at {trial_fullname}/{timestep_sel}."
                )

        params_unscaled = self._load_params_sequence(trial_dir, selected_frames)
        params = {key: value.copy() for key, value in params_unscaled.items()}
        params['means3D'] *= self.cfg['pointcloud']['scaling']

        input_frame_num = self.cfg['model']['input_frame_num']
        target_start = input_frame_num + 1
        if len(params['means3D']) <= target_start:
            raise ValueError(
                f"{trial_fullname} needs at least {target_start + 1} frames "
                "for two velocity inputs and one prediction target."
            )

        target_positions = torch.from_numpy(params['means3D'][target_start:])
        target_rotations = torch.from_numpy(params['unnorm_rotations'][target_start:])
        target_count = target_positions.shape[0]
        expected_target_count = self.cfg[self.phase]['seq_len'] - input_frame_num
        if self.phase != 'test' and target_count != expected_target_count:
            raise AssertionError(
                f"Expected {expected_target_count} targets, got {target_count}."
            )
        if self.phase == 'test' and target_count > expected_target_count:
            raise ValueError(
                f"test.seq_len supports {expected_target_count} predictions, but "
                f"{trial_fullname} requires {target_count}. Increase test.seq_len."
            )

        example = {
            'seq_name': trial_fullname,
            'start_frame': timestep_sel,
            'inv_rot': torch.eye(3, dtype=torch.float32),
            'seq_info': {},
            'gt': {
                'position': target_positions,
                'rotation': target_rotations,
            },
            # Position loss reads every target from gt. These entries preserve
            # the existing runner's lookahead-count bookkeeping.
            'gt_lookahead': [
                {
                    'position': target_positions[target_idx],
                    'rotation': target_rotations[target_idx],
                }
                for target_idx in range(1, target_count)
            ],
        }

        reference_frame_idx = 0
        for input_idx, t in enumerate(range(1, input_frame_num + 1)):
            pos = params['means3D'][t]
            velocity = params['means3D'][t] - params['means3D'][t - 1]
            rotation = params['unnorm_rotations'][t]
            opacity = params['logit_opacities'][reference_frame_idx]
            log_scale = params['log_scales'][reference_frame_idx]
            dummy_segmentation = np.zeros((pos.shape[0], 3), dtype=np.float32)
            sh = params['shs'][reference_frame_idx].reshape(-1, 16 * 3)
            features = np.concatenate(
                (rotation, opacity, log_scale, dummy_segmentation, sh), axis=1
            )
            if features.shape[1] != 59:
                raise AssertionError(
                    f"Expected 59 Gaussian placeholder channels, got {features.shape[1]}."
                )

            seq_info, nn_idx = self._prepare_model_input(
                [pos, velocity], [features]
            )
            example['seq_info'][input_idx] = seq_info
            if input_idx == input_frame_num - 1:
                example['nn_idx'] = nn_idx

        if self.phase == 'test':
            output_scene_dir = os.path.join(
                self.cfg['output_dir'], self.cfg['exp_name_epoch'], trial_fullname
            )
            os.makedirs(output_scene_dir, exist_ok=True)
            np.savez(
                os.path.join(output_scene_dir, 'params_gt.npz'),
                **params_unscaled,
            )

        return example

    def _required_position_frame_count(self, phase):

        if phase == 'test':
            # p0, p1 and p2 form two velocity inputs; p3 is the first target.
            return self.cfg['model']['input_frame_num'] + 2

        # A training sequence with S logical frames needs S + 1 positions
        # because the oldest position is used only to form the first velocity.
        return self.cfg[phase]['seq_len'] + 1

    @staticmethod
    def _get_frame_paths(scene_folder):

        frame_paths = {}
        for entry in os.listdir(scene_folder):
            entry_path = os.path.join(scene_folder, entry)
            if not os.path.isdir(entry_path):
                continue
            try:
                frame_number = int(entry)
            except ValueError:
                continue
            if frame_number in frame_paths:
                raise ValueError(
                    f"Duplicate numeric frame {frame_number} in {scene_folder}."
                )
            param_path = os.path.join(entry_path, 'params_coarse.npz')
            if not os.path.isfile(param_path):
                raise FileNotFoundError(
                    f"Missing params_coarse.npz for frame {entry_path}."
                )
            frame_paths[frame_number] = entry

        if not frame_paths:
            raise ValueError(f"No numeric point-cloud frames found in {scene_folder}.")

        return frame_paths

    @classmethod
    def _get_frame_numbers(cls, scene_folder):

        frame_numbers = sorted(cls._get_frame_paths(scene_folder))
        if len(frame_numbers) > 2:
            frame_steps = np.diff(frame_numbers)
            if not np.all(frame_steps == frame_steps[0]):
                raise ValueError(
                    f"Frame spacing must be constant in {scene_folder}; got "
                    f"steps {sorted(set(frame_steps.tolist()))}."
                )
        return frame_numbers

    def _load_params_sequence(self, scene_folder, frame_numbers):

        required_shapes = {
            'means3D': (3,),
            'unnorm_rotations': (4,),
            'logit_opacities': (1,),
            'log_scales': (3,),
            'shs': (16, 3),
        }
        frame_paths = self._get_frame_paths(scene_folder)
        sequence = {key: [] for key in required_shapes}
        expected_point_count = None

        for frame_number in frame_numbers:
            if frame_number not in frame_paths:
                raise FileNotFoundError(
                    f"Frame {frame_number} is missing from {scene_folder}."
                )
            param_path = os.path.join(
                scene_folder, frame_paths[frame_number], 'params_coarse.npz'
            )
            with np.load(param_path, allow_pickle=False) as data:
                missing = [key for key in required_shapes if key not in data.files]
                if missing:
                    raise KeyError(
                        f"{param_path} is missing required arrays: {missing}."
                    )

                for key, trailing_shape in required_shapes.items():
                    value = np.asarray(data[key])
                    expected_ndim = 2 + len(trailing_shape)
                    if (
                        value.ndim != expected_ndim
                        or value.shape[0] != 1
                        or value.shape[2:] != trailing_shape
                    ):
                        raise ValueError(
                            f"{param_path}:{key} has shape {value.shape}; expected "
                            f"(1, N, {', '.join(map(str, trailing_shape))})."
                        )
                    if not np.issubdtype(value.dtype, np.number):
                        raise TypeError(f"{param_path}:{key} must be numeric.")
                    if not np.isfinite(value).all():
                        raise ValueError(f"{param_path}:{key} contains non-finite values.")

                    point_count = value.shape[1]
                    if point_count == 0:
                        raise ValueError(f"{param_path}:{key} contains no points.")
                    if expected_point_count is None:
                        expected_point_count = point_count
                    elif point_count != expected_point_count:
                        raise ValueError(
                            f"Point count changed in {param_path}: expected "
                            f"{expected_point_count}, got {point_count}. Stable point "
                            "correspondence is required across a rollout."
                        )

                    sequence[key].append(value.astype(np.float32, copy=False))

        params = {
            key: np.concatenate(values, axis=0)
            for key, values in sequence.items()
        }
        quaternion_norms = np.linalg.norm(params['unnorm_rotations'], axis=-1)
        if np.any(quaternion_norms <= 1e-12):
            raise ValueError(
                "unnorm_rotations contains a zero quaternion; use the identity "
                "placeholder [1, 0, 0, 0]."
            )

        return params

    def _prepare_model_input(self, data_curr, data_curr_aux):

        positions_curr, velocities_curr = data_curr
        assert len(data_curr_aux) == 1
        feats_curr = data_curr_aux[0]
        assert positions_curr.shape[0] == feats_curr.shape[0]
        assert positions_curr.shape[0] == velocities_curr.shape[0]

        levels = self.cfg['pointcloud']['downsampling_layer_num'] + 1
        idx_curr_list = {}
        pos_curr_list = {}
        feats_curr_list = {}
        nn_idx_self_list = {}
        nn_idx_forward_list = {}
        nn_idx_propagate_list = {}
        for i in range(levels - 1):
            idx_curr_list[i] = []
            pos_curr_list[i] = []
            feats_curr_list[i] = []
            nn_idx_self_list[i] = []
            nn_idx_forward_list[i] = []
            nn_idx_propagate_list[i] = []
        idx_curr_list[levels - 1] = []
        pos_curr_list[levels - 1] = []
        feats_curr_list[levels - 1] = []
        nn_idx_self_list[levels - 1] = []
        instance_idx = [0, positions_curr.shape[0]]

        for i in range(len(instance_idx) - 1):
            st, ed = instance_idx[i], instance_idx[i + 1]
            pc_all, feat_all, grid_idx_all, nn_idx_all = find_knns_across_levels(
                positions_curr[st:ed],
                feats_curr[st:ed],
                self.cfg['pointcloud']['minimum_point_num'],
                self.cfg['pointcloud']['knn'],
                self.cfg['pointcloud']['knn_k_decay_factor'],
                self.cfg['pointcloud']['grid_size'],
                self.cfg['pointcloud']['scaling']
            )
            grid_idx_offset = st
            assert len(grid_idx_all) == levels
            assert len(pc_all) == levels
            assert len(feat_all) == levels
            assert len(nn_idx_all['self']) == levels
            assert len(nn_idx_all['forward']) == levels - 1
            assert len(nn_idx_all['propagate']) == levels - 1
            idx_curr_list[levels - 1].append(grid_idx_all[levels - 1] + grid_idx_offset)
            pos_curr_list[levels - 1].append(pc_all[levels - 1])
            feats_curr_list[levels - 1].append(feat_all[levels - 1])
            nn_idx_self_list[levels - 1].append(nn_idx_all['self'][levels - 1])
            for j in range(levels - 1):
                idx_curr_list[j].append(grid_idx_all[j] + grid_idx_offset)
                pos_curr_list[j].append(pc_all[j])
                feats_curr_list[j].append(feat_all[j])
                nn_idx_self_list[j].append(nn_idx_all['self'][j])
                nn_idx_forward_list[j].append(nn_idx_all['forward'][j])
                nn_idx_propagate_list[j].append(nn_idx_all['propagate'][j])
        assert grid_idx_offset == 0, "A scene is prepared as one point cloud."
        assert ed == positions_curr.shape[0]

        grid_idx_curr = []
        pos_curr = []
        nn_idx_self = []
        nn_idx_forward = []
        nn_idx_propagate = []
        for i in range(levels - 1):
            pos_curr.append(Pointclouds(points=pos_curr_list[i], features=feats_curr_list[i]))
            nn_idx_self.append(Pointclouds(points=pos_curr_list[i], features=nn_idx_self_list[i]))
            nn_idx_forward.append(Pointclouds(points=pos_curr_list[i + 1], features=nn_idx_forward_list[i]))
            nn_idx_propagate.append(Pointclouds(points=pos_curr_list[i], features=nn_idx_propagate_list[i]))
            grid_idx_curr.append(torch.cat(idx_curr_list[i], dim=0))
        pos_curr.append(Pointclouds(points=pos_curr_list[levels - 1], features=feats_curr_list[levels - 1]))
        nn_idx_self.append(Pointclouds(points=pos_curr_list[levels - 1], features=nn_idx_self_list[levels - 1]))
        grid_idx_curr.append(torch.cat(idx_curr_list[levels - 1], dim=0))

        seq_info = {}
        vel_packed = torch.tensor(velocities_curr)
        assert vel_packed.shape[1] == 3, "velocities should only contain position velocities"
        seq_info['pc'] = pos_curr
        seq_info['grid_idx'] = grid_idx_curr
        # The assignment-free model consumes positional velocity directly.
        # No height, gravity-axis, rotation, or scale channels are appended.
        seq_info['velocity'] = vel_packed

        nn_idx = {}
        nn_idx['self'] = nn_idx_self
        nn_idx['forward'] = nn_idx_forward
        nn_idx['propagate'] = nn_idx_propagate

        return seq_info, nn_idx

    def update_hard_examples(self, hard_examples, hard_examples_frames):

        self.hard_examples = hard_examples
        self.hard_examples_frames = hard_examples_frames


def collate_fn(data):

    batch = {}    
    # get scene info here
    all_seq_name = []
    all_start_frame = []    
    all_inv_rot = []
    all_gt = []
    all_gt_lookahead = []
    for i in range(len(data)):
        all_seq_name.append(data[i]['seq_name'])
        all_start_frame.append(data[i]['start_frame'])
        all_inv_rot.append(data[i]['inv_rot'])
        all_gt.append(data[i]['gt'])
        all_gt_lookahead.append(data[i]['gt_lookahead'])
    batch['seq_name'] = all_seq_name
    batch['start_frame'] = all_start_frame
    batch['inv_rot'] = all_inv_rot
    batch['gt'] = all_gt
    batch['gt_lookahead'] = all_gt_lookahead

    # get mullti-frame info here
    batch['pc'] = {}
    batch['grid_idx'] = {}
    batch['velocity'] = {}
    for frame_num in range(len(data[0]['seq_info'].keys())):
        offset_grid_idx = 0
        all_grid_idx_tmp = {}
        all_points_tmp = {}
        all_feats_tmp = {}
        all_velocities = []        
        for i in range(len(data)):
            pointclouds = data[i]['seq_info'][frame_num]['pc']            
            grid_idx = data[i]['seq_info'][frame_num]['grid_idx']
            velocities = data[i]['seq_info'][frame_num]['velocity']
            assert len(pointclouds) == len(grid_idx)
            if i == 0:
                for j in range(len(pointclouds)):
                    all_grid_idx_tmp[j] = []
                    all_points_tmp[j] = []
                    all_feats_tmp[j] = []
            for j in range(len(pointclouds)):
                all_grid_idx_tmp[j].append(grid_idx[j] + offset_grid_idx)
                all_points_tmp[j] += pointclouds[j].points_list()
                all_feats_tmp[j] += pointclouds[j].features_list()
            offset_grid_idx += pointclouds[0].points_packed().shape[0]
            all_velocities.append(velocities)        
        
        pc = []
        grid_idx = []
        for j in sorted(all_points_tmp.keys()):
            pc.append(Pointclouds(points=all_points_tmp[j], features=all_feats_tmp[j]))
            grid_idx.append(torch.cat(all_grid_idx_tmp[j], dim=0))
            _ = pc[-1].points_packed()

        batch['pc'][frame_num] = pc
        batch['grid_idx'][frame_num] = grid_idx
        batch['velocity'][frame_num] = torch.cat(all_velocities, dim=0)

    # set up variables
    batch_idx = {}
    offset_self = {}
    all_nn_self = {}
    all_points_self = {}
    pointclouds_self_tmp = data[0]['nn_idx']['self']
    for j in range(len(pointclouds_self_tmp)):
        batch_idx[j] = torch.zeros(len(data), dtype=torch.int)  
        offset_self[j] = 0
        all_nn_self[j] = []
        all_points_self[j] = []

    # merge knn indices here (at the same level)
    nn_self = {}
    for i in range(len(data)):
        pointclouds_self = data[i]['nn_idx']['self']  
        for j in range(len(pointclouds_self)):
            nn_self[j] = []
            for ii, features in enumerate(pointclouds_self[j].features_list()):
                nn_self[j].append(features + offset_self[j])
                offset_self[j] += features.shape[0]
            all_nn_self[j] += nn_self[j]
            all_points_self[j] += pointclouds_self[j].points_list() # dummy points
            batch_idx[j][i] = torch.sum(pointclouds_self[j].num_points_per_cloud())       

    # set up variables
    offset_forward = {}
    offset_propagate = {}
    all_nn_forward = {}
    all_nn_propagate = {}
    all_points_forward = {}
    all_points_propagate = {}
    levels = len(pointclouds_self_tmp)
    for j in range(len(pointclouds_self_tmp)):
        offset_forward[j] = 0
        offset_propagate[j] = 0
        all_nn_forward[j] = []
        all_nn_propagate[j] = []
        all_points_forward[j] = []
        all_points_propagate[j] = []

    # merge knn indices here (across levels)
    nn_forward = {}
    nn_propagate = {}
    for i in range(len(data)):
        pointclouds_self = data[i]['nn_idx']['self']
        pointclouds_forward = data[i]['nn_idx']['forward']
        pointclouds_propagate = data[i]['nn_idx']['propagate']
        assert len(pointclouds_forward) == len(pointclouds_propagate)
        for j in range(len(pointclouds_forward)):
            pc_self_dense = pointclouds_self[j].features_list()
            pc_self_sparse = pointclouds_self[j+1].features_list()
            pc_forward_list = pointclouds_forward[j].features_list()
            pc_propagate_list = pointclouds_propagate[j].features_list()
            nn_forward[j] = []
            nn_propagate[j] = []
            assert len(pc_forward_list) == len(pc_propagate_list)
            for k in range(len(pc_forward_list)):
                dense_pc = pc_self_dense[k]
                sparse_pc = pc_self_sparse[k]
                features_forward = pc_forward_list[k]
                features_propagate = pc_propagate_list[k]
                nn_forward[j].append(features_forward + offset_forward[j])
                nn_propagate[j].append(features_propagate + offset_propagate[j])
                offset_forward[j] += dense_pc.shape[0]
                offset_propagate[j] += sparse_pc.shape[0]
            all_nn_forward[j] += nn_forward[j]
            all_nn_propagate[j] += nn_propagate[j]
            all_points_forward[j] += pointclouds_forward[j].points_list() # dummy points            
            all_points_propagate[j] += pointclouds_propagate[j].points_list() # dummy points
            assert len(all_points_forward[j]) ==  len(all_points_propagate[j])

    pc_self =  {}
    pc_forward = {}
    pc_propagagte = {}    
    for i in range(levels):
        pc_self[i] = Pointclouds(points=all_points_self[i], features=all_nn_self[i])
        if i != levels - 1:
            pc_forward[i] = Pointclouds(points=all_points_forward[i], features=all_nn_forward[i])
            pc_propagagte[i] = Pointclouds(points=all_points_propagate[i], features=all_nn_propagate[i])

    batch['batch_idx'] = batch_idx
    batch['nn_idx_self'] = pc_self
    batch['nn_idx_forward'] = pc_forward
    batch['nn_idx_propagate'] = pc_propagagte

    return batch


def parse_target_info(target_dict, device):

    tmp = {}
    for key, value in target_dict.items():
        if key in ['position', 'rotation']:
            tmp[key] = value.to(device)
        else:
            raise ValueError(f"Unsupported target field: {key}")
    
    return tmp


def move_to_gpu(tensor_dict, device):

    tensors_out = {}
    for key1 in tensor_dict.keys():
        if key1 in ['seq_name', 'start_frame']:
            tensors_out[key1] = tensor_dict[key1]
        elif key1 == 'inv_rot':
            tensors_out[key1] = []
            for tensor in tensor_dict[key1]:
                tensors_out[key1].append(tensor.to(device))
        elif key1 == 'gt':
            tensors_out[key1] = []
            for dict_tmp in tensor_dict[key1]:
                tensors_out[key1].append(parse_target_info(dict_tmp, device))
        elif key1 == 'gt_lookahead':
            tensors_out[key1] = []
            for dict_list in tensor_dict[key1]:
                dict_out_list = []
                for dict_tmp in dict_list:
                    dict_out_list.append(parse_target_info(dict_tmp, device))
                tensors_out[key1].append(dict_out_list)
        else:
            tensors_out[key1] = {}
            for key2 in sorted(tensor_dict[key1].keys()):
                if key1 in ['pc', 'grid_idx']:
                    tensors_out[key1][key2] = []
                    for tensor in tensor_dict[key1][key2]:
                        tensors_out[key1][key2].append(tensor.to(device))
                else:                
                    tensors_out[key1][key2] = tensor_dict[key1][key2].to(device)
    
    return tensors_out
