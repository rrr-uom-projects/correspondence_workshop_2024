# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
from neuromorph_adapted.data.data import *
from neuromorph_adapted.model.layers import *
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from neuromorph_adapted.utils.utils import RunningAverage
import pyvista as pv
pv.global_theme.background = 'white'
pv.start_xvfb()

class InterpolationModBase(torch.nn.Module):
    def __init__(self, interp_energy):
        super().__init__()
        self.interp_energy = interp_energy

    def get_pred(self, shape_x, shape_y):
        raise NotImplementedError()

    def compute_loss(self, shape_x, shape_y, point_pred_arr):
        raise NotImplementedError()

    def forward(self, shape_x, shape_y):
        point_pred_arr = self.get_pred(shape_x, shape_y)
        return self.compute_loss(shape_x, shape_y, point_pred_arr)


class InterpolationModGeoEC(InterpolationModBase):
    def __init__(self, interp_energy, param):
        super().__init__(interp_energy)
        self.param = param
        param.print_self()
        self.rn_ec = ResnetECPos(c_dim=3, dim=7, hidden_dim=param.hidden_dim)
        self.feat_module = ResnetECPos(c_dim=param.hidden_dim, dim=6, hidden_dim=param.hidden_dim)
        print("Uses module 'InterpolationModGeoEC'")
        self.Pi = None
        self.Pi_inv = None

    def get_pred(self, shape_x, shape_y, update_corr=True):
        if update_corr:
            self.match(shape_x, shape_y)

        step_size = 1 / (self.param.num_timesteps + 1)
        timesteps = step_size + torch.arange(0, 1, step_size, device=device).unsqueeze(1).unsqueeze(2)  # [T, 1, 1]
        timesteps_up = timesteps * (torch.as_tensor([0, 0, 0, 0, 0, 0, 1], device=device, dtype=torch.float).unsqueeze(0).unsqueeze(1))  # [T, 1, 7]
        
        points_in = torch.cat(
            (
                shape_x.verts,
                torch.mm(self.Pi, shape_y.verts) - shape_x.verts,
                my_zeros((shape_x.verts.shape[0], 1)),
            ),
            dim=1).unsqueeze(0)  # [1, n, 7]
        points_in = points_in + timesteps_up
        edge_index = shape_x.get_edge_index()

        displacement = my_zeros([points_in.shape[0], points_in.shape[1], 3])
        for i in range(points_in.shape[0]):
            displacement[i, :, :] = self.rn_ec(points_in[i, :, :], edge_index)
        # the previous three lines used to support batchwise processing in torch-geometric but are now deprecated:
        # displacement = self.rn_ec(points_in, edge_index)  # [T, n, 3]

        point_pred_arr = shape_x.verts.unsqueeze(0) + displacement * timesteps
        point_pred_arr = point_pred_arr.permute([1, 2, 0])
        return point_pred_arr

    def compute_loss(self, shape_x, shape_y, point_pred_arr, n_normalize=201.0):

        E_x_0 = self.param.lambd_arap  * (
            self.interp_energy.forward_single(shape_x.verts, point_pred_arr[:, :, 0], shape_x) + 
            self.interp_energy.forward_single(point_pred_arr[:, :, 0], shape_x.verts, shape_x)
            )

        lambda_align = n_normalize / shape_x.verts.shape[0]
        E_align = (
            lambda_align
            * self.param.lambd
            * (
                (torch.mm(self.Pi, shape_y.verts) - point_pred_arr[:, :, -1]).norm() ** 2
                + (
                    shape_y.verts - torch.mm(self.Pi_inv, point_pred_arr[:, :, -1])
                ).norm()
                ** 2
            )
        )

        if shape_x.D is None:
            E_geo = my_tensor(0)
        elif self.param.lambd_geo == 0:
            E_geo = my_tensor(0)
        else:
            E_geo = (
                self.param.lambd_geo
                * (
                    (
                        torch.mm(torch.mm(self.Pi, shape_y.D), self.Pi.transpose(0, 1))
                        - shape_x.D
                    )
                    ** 2
                ).mean()
            )

        E = E_x_0 + E_align + E_geo

        for i in range(self.param.num_timesteps):
            E_x = self.param.lambd_arap * self.interp_energy.forward_single(point_pred_arr[:, :, i], point_pred_arr[:, :, i + 1], shape_x)
            E_y = self.param.lambd_arap * self.interp_energy.forward_single(point_pred_arr[:, :, i + 1], point_pred_arr[:, :, i], shape_x)
            E = E + E_x + E_y

        return E, [E - E_align - E_geo, E_align, E_geo]

    def match(self, shape_x, shape_y):
        feat_x = torch.cat((shape_x.verts, shape_x.get_normal()), dim=1)
        feat_y = torch.cat((shape_y.verts, shape_y.get_normal()), dim=1)

        feat_x = self.feat_module(feat_x, shape_x.get_edge_index())
        feat_y = self.feat_module(feat_y, shape_y.get_edge_index())

        feat_x = feat_x / feat_x.norm(dim=1, keepdim=True)
        feat_y = feat_y / feat_y.norm(dim=1, keepdim=True)

        D = torch.mm(feat_x, feat_y.transpose(0, 1))

        sigma = 1e2
        self.Pi = F.softmax(D * sigma, dim=1)
        self.Pi_inv = F.softmax(D * sigma, dim=0).transpose(0, 1)

        return self.Pi

    def load_self(self, folder_path, num_epoch=None):
        if num_epoch is None:
            ckpt_name = "ckpt_last.pth"
            ckpt_path = os.path.join(folder_path, ckpt_name)
        else:
            ckpt_name = f"ckpt_ep{num_epoch}.pth"
            ckpt_path = os.path.join(folder_path, ckpt_name)

        self.load_chkpt(ckpt_path)

        if num_epoch is None:
            print("Loaded model from ", folder_path, " with the latest weights")
        else:
            print("Loaded model from ", folder_path, " with the weights from epoch ", num_epoch)

    def load_chkpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)

        self.load_state_dict(ckpt["interp_module"])

        if "par" in ckpt:
            self.param.from_dict(ckpt["par"])
            self.param.print_self()


################################################################################################


class InterpolNet:
    def __init__(self, interp_module: InterpolationModBase, train_loader, val_loader=None, time_stamp=None, description="", preproc_mods=[], settings_module=None):
        super().__init__()
        self.time_stamp = time_stamp
        self.interp_module = interp_module
        self.settings_module = settings_module
        self.preproc_mods = preproc_mods
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.i_epoch = 0
        self.optimizer = torch.optim.Adam(self.interp_module.parameters(), lr=self.interp_module.param.lr)
        if description == "":
            self.checkpoint_dir = save_path(self.time_stamp)
        else:
            self.checkpoint_dir = save_path(self.time_stamp + "_" + description)
        self.writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_dir, 'logs'))

    def train(self):
        print("start training ...")
        self.interp_module.train()
        while self.i_epoch < self.interp_module.param.num_it:
            
            loss_total = RunningAverage()
            loss_arap = RunningAverage()
            loss_reg = RunningAverage()
            loss_geo = RunningAverage()
            self.update_settings()

            for i, data in enumerate(tqdm(self.train_loader)):
                shape_x = batch_to_shape(data["X"])
                shape_y = batch_to_shape(data["Y"])

                shape_x, shape_y = self.preprocess(shape_x, shape_y)

                loss, loss_comp = self.interp_module(shape_x, shape_y)
                loss_total.update(loss.item())
                loss_arap.update(loss_comp[0].item())
                loss_reg.update(loss_comp[1].item())
                loss_geo.update(loss_comp[2].item())
                loss.backward()

                if (i + 1) % self.interp_module.param.batch_size == 0 and i < len(self.train_loader) - 1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self.optimizer.step()
            self.optimizer.zero_grad()

            print(f"epoch {self.i_epoch:04d}, loss = {loss_total.avg:.5f} (arap: {loss_arap.avg:.5f}, reg: {loss_reg.avg:.5f}, geo: {loss_geo.avg:.5f}), reserved memory={torch.cuda.memory_reserved(0) // (1024 ** 2)}MB")
            self._log_stats('train', "total", loss_total.avg)
            self._log_stats('train', "arap", loss_arap.avg)
            self._log_stats('train', "reg", loss_reg.avg)
            self._log_stats('train', "geo", loss_geo.avg)

            if (self.i_epoch + 1) % self.interp_module.param.log_freq == 0:
                self.save_self()
            if (self.i_epoch + 1) % self.interp_module.param.val_freq == 0 and self.val_loader is not None:
                self.validate()

            self.i_epoch += 1

    def validate(self, data_loader=None, log=True, plot_example=True):
        if data_loader is None:
            data_loader = self.val_loader

        loss_total = RunningAverage()
        loss_arap = RunningAverage()
        loss_reg = RunningAverage()
        loss_geo = RunningAverage()

        with torch.no_grad():
            for i, data in enumerate(tqdm(data_loader)):
                shape_x = batch_to_shape(data["X"])
                shape_y = batch_to_shape(data["Y"])

                if plot_example and i == 1:
                    pi = self.interp_module.match(shape_x, shape_y).detach().cpu().numpy()
                    self.plot_example(pi, shape_x.verts.detach().cpu().numpy(), shape_y.verts.detach().cpu().numpy())
                point_pred = self.interp_module.get_pred(shape_x, shape_y)

                loss, loss_comp = self.interp_module.compute_loss(shape_x, shape_y, point_pred)
                loss_total.update(loss.item())
                loss_arap.update(loss_comp[0].item())
                loss_reg.update(loss_comp[1].item())
                loss_geo.update(loss_comp[2].item())

        if log:
            print(f"Validation loss = {loss_total.avg:.5f} (arap: {loss_arap.avg:.5f}, reg: {loss_reg.avg:.5f}, geo: {loss_geo.avg:.5f})")
            self._log_stats('val', "total", loss_total.avg)
            self._log_stats('val', "arap", loss_arap.avg)
            self._log_stats('val', "reg", loss_reg.avg)
            self._log_stats('val', "geo", loss_geo.avg)


    def test(self, data_loader=None):
        if data_loader is None:
            data_loader = self.val_loader
        shape_x_out = []
        shape_y_out = []
        points_out = []
        fname_x_out = []
        fname_y_out = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(data_loader)):
                shape_x = batch_to_shape(data["X"])
                shape_y = batch_to_shape(data["Y"])

                point_pred = self.interp_module.get_pred(shape_x, shape_y)

                shape_x.detach_cpu()
                shape_y.detach_cpu()
                point_pred = point_pred.detach().cpu()

                points_out.append(point_pred)
                shape_x_out.append(shape_x)
                shape_y_out.append(shape_y)
                fname_x_out.append(data["X"]["fname"])
                fname_y_out.append(data["Y"]["fname"])           
        
        return shape_x_out, shape_y_out, points_out, fname_x_out, fname_y_out

    def preprocess(self, shape_x, shape_y):
        for pre in self.preproc_mods:
            shape_x, shape_y = pre.preprocess(shape_x, shape_y)
        return shape_x, shape_y

    def update_settings(self):
        if self.settings_module is not None:
            self.settings_module.update(self.interp_module, self.i_epoch)

    def save_self(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        ckpt_last_name = "ckpt_last.pth"
        ckpt_last_path = os.path.join(self.checkpoint_dir, ckpt_last_name)

        ckpt_name = "ckpt_ep{}.pth".format(self.i_epoch)
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)

        self.save_chkpt(ckpt_path)
        self.save_chkpt(ckpt_last_path)

    def save_chkpt(self, ckpt_path):
        ckpt = {
            "i_epoch": self.i_epoch,
            "interp_module": self.interp_module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "par": self.interp_module.param.__dict__,
        }

        torch.save(ckpt, ckpt_path)

    def load_self(self, folder_path, num_epoch=None):
        if num_epoch is None:
            ckpt_name = "ckpt_last.pth"
            ckpt_path = os.path.join(folder_path, ckpt_name)
        else:
            ckpt_name = f"ckpt_ep{num_epoch}.pth"
            ckpt_path = os.path.join(folder_path, ckpt_name)

        self.load_chkpt(ckpt_path)

        if num_epoch is None:
            print("Loaded model from ", folder_path, " with the latest weights")
        else:
            print("Loaded model from ", folder_path, " with the weights from epoch ", num_epoch)

    def load_chkpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)

        self.i_epoch = ckpt["i_epoch"]
        self.interp_module.load_state_dict(ckpt["interp_module"])

        if "par" in ckpt:
            self.interp_module.param.from_dict(ckpt["par"])
            self.interp_module.param.print_self()

        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        self.interp_module.train()
    
    def _log_stats(self, phase, loss_comp, loss_avg):
        tag_value = {f'{phase}_loss_{loss_comp}_avg': loss_avg}
        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.i_epoch)

    def plot_example(self, pi, verts_x, verts_y):
        assignment = np.argmax(pi, axis=1)
        # create 3D plot
        plotter_gs = pv.Plotter(off_screen=True)
        plotter_pred = pv.Plotter(off_screen=True)
        # reorder axes for plotting
        verts_x = verts_x[:, [0,2,1]]
        verts_y = verts_y[:, [0,2,1]]
        # Forward assignment
        # set colors on target mesh corresponding with xyz position
        colors_y = verts_y - np.min(verts_y, axis=0)
        colors_y = colors_y / np.max(colors_y, axis=0)
        colors_y = np.concatenate([colors_y, np.ones((len(colors_y), 1))], axis=1)
        # set colors on source mesh with the hard corrspondence assignment
        colors_x = colors_y[assignment]
        # add points to plotter
        plotter_gs.add_points(verts_y, opacity=1., point_size=10, render_points_as_spheres=True, scalars=colors_y, rgb=True)
        plotter_pred.add_points(verts_x, opacity=1., point_size=10, render_points_as_spheres=True, scalars=colors_x, rgb=True)
        plotter_gs.store_image, plotter_pred.store_image = True, True
        plotter_gs.show(window_size=[500, 500], auto_close=True)
        plotter_pred.show(window_size=[500, 500], auto_close=True)
        # jump to matplotlib
        fig, (ax_gs, ax_pred) = plt.subplots(1, 2, figsize=(12, 6.2), tight_layout=True)
        ax_gs.imshow(plotter_gs.image)
        ax_pred.imshow(plotter_pred.image)
        ax_gs.axis('off')
        ax_pred.axis('off')
        ax_gs.set_title(f"Target")
        ax_pred.set_title(f"Prediction")
        self.writer.add_figure(tag="Correspondences", figure=fig, global_step=self.i_epoch)


class timestep_settings:
    def __init__(self, increase_thresh):
        super().__init__()
        self.increase_thresh = increase_thresh

    def update(self, interp_module, i_epoch):
        num_t_before = interp_module.param.num_timesteps
        if i_epoch < self.increase_thresh:
            return
        elif i_epoch < self.increase_thresh * 1.5:
            num_t = 1
        else:
            num_t = 3

        if num_t_before != num_t:
            interp_module.param.num_timesteps = num_t
            print("Set the # of timesteps to ", num_t)
            interp_module.param.lambd_geo = 0
            print("Deactivated the geodesic loss")
        
class PreprocessRotateBase:
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def _create_rot_matrix(self, alpha):
        return create_rotation_matrix(alpha, self.axis)

    def _rand_rot(self):
        alpha = torch.rand(1) * 360
        return self._create_rot_matrix(alpha)

    def rot_sub(self, shape, r):
        if shape.sub is not None:
            for i_p in range(len(shape.sub[0])):
                shape.sub[0][i_p][0, :, :] = torch.mm(shape.sub[0][i_p][0, :, :], r)

        if shape.verts_full is not None:
            shape.verts_full = torch.mm(shape.verts_full, r)

        return shape

    def preprocess(self, shape_x, shape_y):
        raise NotImplementedError()


class PreprocessRotate(PreprocessRotateBase):
    def __init__(self, axis=1):
        super().__init__(axis)
        print("Uses preprocessing module 'PreprocessRotate'")

    def preprocess(self, shape_x, shape_y):
        r_x = self._rand_rot()
        r_y = self._rand_rot()
        shape_x.verts = torch.mm(shape_x.verts, r_x)
        shape_y.verts = torch.mm(shape_y.verts, r_y)
        shape_x = self.rot_sub(shape_x, r_x)
        shape_y = self.rot_sub(shape_y, r_y)
        return shape_x, shape_y


class PreprocessRotateSame(PreprocessRotateBase):
    def __init__(self, axis=1):
        super().__init__(axis)
        print("Uses preprocessing module 'PreprocessRotateSame'")

    def preprocess(self, shape_x, shape_y):
        r = self._rand_rot()
        shape_x.verts = torch.mm(shape_x.verts, r)
        shape_y.verts = torch.mm(shape_y.verts, r)

        shape_x = self.rot_sub(shape_x, r)
        shape_y = self.rot_sub(shape_y, r)
        return shape_x, shape_y


class PreprocessRotateAugment(PreprocessRotateBase):
    def __init__(self, axis=1, sigma=0.3):
        super().__init__(axis)
        self.sigma = sigma
        print(
            "Uses preprocessing module 'PreprocessRotateAugment' with sigma =",
            self.sigma,
        )

    def preprocess(self, shape_x, shape_y):
        r_x = self._rand_rot_augment()
        r_y = self._rand_rot_augment()
        shape_x.verts = torch.mm(shape_x.verts, r_x)
        shape_y.verts = torch.mm(shape_y.verts, r_y)

        shape_x = self.rot_sub(shape_x, r_x)
        shape_y = self.rot_sub(shape_y, r_y)
        return shape_x, shape_y

    # computes a pair of approximately similar rotation matrices
    def _rand_rot_augment(self):
        rot = torch.randn(
            [3, 3], dtype=torch.float, device=device
        ) * self.sigma + my_eye(3)

        U, _, V = torch.svd(rot, compute_uv=True)

        rot = torch.mm(U, V.transpose(0, 1))

        return rot


if __name__ == "__main__":
    print("main of interpolation_net.py")
