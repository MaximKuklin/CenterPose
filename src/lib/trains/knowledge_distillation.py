from copy import deepcopy

import torch
import time
import numpy as np

from progress.bar import Bar
from .base_trainer import BaseTrainer
from src.lib.models.model import create_model, load_model, save_model
from lib.utils.utils import AverageMeter
from lib.models.data_parallel import DataParallel
from lib.models.decode import object_pose_decode
from lib.utils.debugger import Debugger


MODELS = {
    "bike": "models/bike_v1_140.pth",
    "book": "models/book_v1_140.pth",
    "bottle": "models/bottle_v1_sym_12_140.pth",
    "camera": "models/camera_v1_140.pth",
    "cereal_box": "models/cereal_box_v1_140.pth",
    "chair": "models/chair_v1_140.pth",
    "cup_cup": "models/cup_cup_v1_sym_12_140.pth",
    "cup_mug": "models/cup_mug_v1_140.pth",
    "laptop": "models/laptop_v1_140.pth",
    "shoe": "models/shoe_v1_140.pth"
}

MODELS_TRAIN = ["laptop"]

TEACHER_TO_MODEL = {
    "motobike": "bike",
    "bike": "bike",
    "camera": "camera",
    "laptop": "laptop"
}


class ModelWithLossKD(torch.nn.Module):
    def __init__(self, teachers, model, loss):
        super(ModelWithLossKD, self).__init__()
        self.teachers = teachers
        self.model = model
        self.loss = loss

    def forward(self, cls, batch, phase):
        pre_img = batch['pre_img'] if 'pre_img' in batch else None
        pre_hm = batch['pre_hm'] if 'pre_hm' in batch else None
        pre_hm_hp = batch['pre_hm_hp'] if 'pre_hm_hp' in batch else None

        #TODO: batches
        teacher_outputs = self.teachers[TEACHER_TO_MODEL[cls[0]]](batch['input'], pre_img, pre_hm, pre_hm_hp)
        outputs = self.model(batch['input'], pre_img, pre_hm, pre_hm_hp)

        loss, loss_stats, choice_list = self.loss(teacher_outputs, outputs, batch, phase)
        return outputs[-1], loss, loss_stats, choice_list


class KDLoss(torch.nn.Module):
    def __init__(self, opt):
        super(KDLoss, self).__init__()
        self.crit = torch.nn.MSELoss()
        self.opt = opt

        self.loss_names = [
            'hm_loss', 'wh_loss', 'hp_loss', 'off_loss', 'hm_hp_loss', 'hp_offset_loss', 'obj_scale_loss',
            'tracking_loss', 'tracking_hp_loss'
        ]

    def pad_teacher_hm(self, hm, batch):
        size = list(hm.size())
        size[1] = len(MODELS_TRAIN)
        ## TODO: batch                                     here
        cls_idx = MODELS_TRAIN.index(batch['meta']['class'][0])
        zeros = torch.zeros(size, dtype=hm.dtype, layout=hm.layout, device=hm.device)
        zeros[:, cls_idx] = hm[:, 0]
        return zeros

    def forward(self, teacher_outputs, outputs, batch, phase):

        ret = {n: 0 for n in self.loss_names}

        for idx in range(self.opt.num_stacks):
            model_output = outputs[idx]
            teacher_output = teacher_outputs[idx]

            teacher_hm = self.pad_teacher_hm(teacher_output['hm'], batch)
            ret['hm_loss'] = self.crit(model_output['hm'], teacher_hm)

            ret['wh_loss'] = self.crit(model_output['wh'], teacher_output['wh'])
            ret['hp_loss'] = self.crit(model_output['hps'], teacher_output['hps'])
            ret['off_loss'] = self.crit(model_output['reg'], teacher_output['reg'])
            ret['hm_hp_loss'] = self.crit(model_output['hm_hp'], teacher_output['hm_hp'])
            ret['hp_offset_loss'] = self.crit(model_output['hp_offset'], teacher_output['hp_offset'])
            ret['obj_scale_loss'] = self.crit(model_output['scale'], teacher_output['scale'])

        loss = 0

        for loss_name, l in ret.items():
            ret[loss_name] /= self.opt.num_stacks
            loss += l

        ret["loss"] = loss

        return loss, ret, [0]



class KDTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(KDTrainer, self).__init__(opt, model, optimizer=optimizer)

        teacher_heads = deepcopy(opt.heads)
        teacher_heads['hm'] = 1
        self.teachers = {m: create_model(opt.arch, teacher_heads, opt.head_conv, opt=opt) for m in MODELS_TRAIN}
        for key in self.teachers.keys():
            self.teachers[key] = load_model(self.teachers[key], MODELS[key])

        self.model_with_loss = ModelWithLossKD(self.teachers, model, self.loss)

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for t in self.teachers.keys():
            if len(gpus) > 1:
                self.teachers[t] = DataParallel(
                    self.teachers[t], device_ids=gpus,
                    chunk_sizes=chunk_sizes).to(device)
            else:
                self.teachers[t] = self.teachers[t].to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)


    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()

        writer_imgs = []  # Clear before each epoch # For tensorboard
        for iter_id, batch in enumerate(data_loader):

            # Skip the bad example
            if batch is None:
                continue

            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            cls = batch['meta']['class']
            output, loss, loss_stats, choice_list = model_with_loss(cls, batch, phase)
            loss = loss.mean()  # No effect for our case

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()

                if isinstance(self.model_with_loss, torch.nn.DataParallel):
                    torch.nn.utils.clip_grad_norm_(self.model_with_loss.module.model.parameters(), 100.)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model_with_loss.model.parameters(), 100.)

                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                # Sometimes, some heads are not enabled
                if torch.is_tensor(loss_stats[l]) == True:
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            # Save everything for debug, including gt_hm/gt_hmhp/out_gt/out_pred/pred_hm/pred_hmhp/out_pred_gt_blend
            if phase == 'train':
                # Only save the first sample to save space

                # Debug only
                # if opt.debug > 0 :
                if opt.debug > 0 and iter_id == 0:
                    writer_imgs.append(self.debug(batch, output, iter_id, choice_list))

            elif opt.debug > 0:
                if opt.debug == 5:
                    # Todo: since validation dataset is not shuffled, we only care about 10+ images
                    if iter_id % (500 / opt.batch_size) == 0:
                        writer_imgs.append(self.debug(batch, output, iter_id, choice_list))
                else:
                    writer_imgs.append(self.debug(batch, output, iter_id, choice_list))

            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results, writer_imgs

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'hp_loss', 'hm_hp_loss',
                       'hp_offset_loss', 'wh_loss', 'off_loss', 'obj_scale_loss', 'tracking_loss', 'tracking_hp_loss',
        ]
        loss = KDLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id, choice_list):
        opt = self.opt

        hps_uncertainty = output['hps_uncertainty'] if opt.hps_uncertainty else None
        reg = output['reg'] if opt.reg_offset else None
        hm_hp = output['hm_hp'] if opt.hm_hp else None
        hp_offset = output['hp_offset'] if opt.reg_hp_offset else None
        obj_scale = output['scale'] if opt.obj_scale else None
        obj_scale_uncertainty = output['scale_uncertainty'] if opt.obj_scale_uncertainty else None
        wh = output['wh'] if opt.reg_bbox else None
        tracking = output['tracking'] if 'tracking' in opt.heads else None
        tracking_hp = output['tracking_hp'] if 'tracking_hp' in opt.heads else None

        dets = object_pose_decode(
            output['hm'], output['hps'], wh=wh, kps_displacement_std=hps_uncertainty, obj_scale=obj_scale,
            obj_scale_uncertainty=obj_scale_uncertainty,
            reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, tracking=tracking, tracking_hp=tracking_hp, opt=self.opt)

        for k in dets:
            dets[k] = dets[k].detach().cpu().numpy()

        dets['bboxes'] *= opt.input_res / opt.output_res
        dets['kps'] *= opt.input_res / opt.output_res

        if 'tracking' in opt.heads:
            dets['tracking'] *= opt.input_res / opt.output_res

        if 'tracking_hp' in opt.heads:
            dets['tracking_hp'] *= opt.input_res / opt.output_res

        # Todo: Right now, only keep the best matched gt
        dets_gt = batch['meta']['gt_det']
        dets_gt = torch.stack([dets_gt[idx][choice] for idx, choice in enumerate(choice_list)])
        dets_gt = dets_gt.numpy()

        dets_gt[:, :, :4] *= opt.input_res / opt.output_res  # bbox
        dets_gt[:, :, 5:21] *= opt.input_res / opt.output_res  # kps
        dets_gt[:, :, 25:27] *= opt.input_res / opt.output_res  # tracking
        dets_gt[:, :, 28:44] *= opt.input_res / opt.output_res  # tracking_hp

        for i in range(1):  # We only care about the first sample in the batch
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                                   img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)

            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i][choice_list[i]].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'out_hm_pred')
            debugger.add_blend_img(img, gt, 'out_hm_gt')

            if 'pre_img' in batch:
                pre_img = batch['pre_img'][i].detach().cpu().numpy().transpose(1, 2, 0)
                pre_img = np.clip(((pre_img * opt.std + opt.mean) * 255), 0, 255).astype(np.uint8)

                if 'pre_hm' in batch:
                    pre_hm = debugger.gen_colormap(
                        batch['pre_hm'][i].detach().cpu().numpy())
                    debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')

                if 'pre_hm_hp' in batch:
                    pre_hmhp = debugger.gen_colormap_hp(
                        batch['pre_hm_hp'][i].detach().cpu().numpy())
                    debugger.add_blend_img(pre_img, pre_hmhp, 'pre_hmhp')

            # Predictions
            debugger.add_img(img, img_id='out_img_pred')
            for k in range(len(dets['scores'][i])):
                if dets['scores'][i][k][0] > opt.center_thresh:

                    if self.opt.reg_bbox:
                        debugger.add_coco_bbox(dets['bboxes'][i][k], dets['clses'][i][k],
                                               dets['scores'][i][k][0], img_id='out_img_pred')
                    debugger.add_coco_hp(dets['kps'][i][k], img_id='out_img_pred')

                    if self.opt.obj_scale == True:
                        if self.opt.reg_bbox:
                            debugger.add_obj_scale(dets['bboxes'][i][k], dets['obj_scale'][i][k], img_id='out_img_pred')
                        else:
                            # Todo: A temporary location, need updates
                            debugger.add_obj_scale([20, 20, 0, 0], dets['obj_scale'][i][k], img_id='out_img_pred')

                    if 'tracking' in opt.heads:
                        debugger.add_arrow(
                            [(dets['bboxes'][i][k][0] + dets['bboxes'][i][k][2]) / 2,
                             (dets['bboxes'][i][k][1] + dets['bboxes'][i][k][3]) / 2, ],
                            dets['tracking'][i][k],
                            img_id='out_img_pred', c=(0, 255, 255))  # yellow
                        debugger.add_arrow(
                            [(dets['bboxes'][i][k][0] + dets['bboxes'][i][k][2]) / 2,
                             (dets['bboxes'][i][k][1] + dets['bboxes'][i][k][3]) / 2, ],
                            dets['tracking'][i][k],
                            img_id='pre_hm', c=(0, 255, 255))  # yellow

                    if 'tracking_hp' in opt.heads:

                        for idx in range(8):

                            if dets['kps'][i][k][idx * 2] == 0 and dets['kps'][i][k][idx * 2 + 1] == 0:
                                continue
                            debugger.add_arrow(
                                dets['kps'][i][k][idx * 2:idx * 2 + 2],
                                dets['tracking_hp'][i][k][idx * 2:idx * 2 + 2],
                                img_id='out_img_pred', c=(0, 0, 255))  # red
                            debugger.add_arrow(
                                dets['kps'][i][k][idx * 2:idx * 2 + 2],
                                dets['tracking_hp'][i][k][idx * 2:idx * 2 + 2],
                                img_id='pre_hmhp', c=(0, 0, 255))  # red

            if opt.hm_hp:
                pred = debugger.gen_colormap_hp(output['hm_hp'][i].detach().cpu().numpy())
                debugger.add_blend_img(img, pred, 'out_hmhp_pred')

            # Ground truth
            debugger.add_img(img, img_id='out_img_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    if self.opt.reg_bbox:
                        debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, 21],
                                               dets_gt[i, k, 4], img_id='out_img_gt')
                    debugger.add_coco_hp(dets_gt[i, k, 5:21], img_id='out_img_gt', pred_flag='gt')

                    if self.opt.obj_scale == True:
                        if self.opt.reg_bbox:
                            debugger.add_obj_scale(dets_gt[i, k, :4], dets_gt[i, k, 22:25], img_id='out_img_gt',
                                                   pred_flag='gt')
                        else:
                            # Todo: A temporary location, need updates
                            debugger.add_obj_scale([20, 20, 0, 0], dets_gt[i, k, 22:25], img_id='out_img_gt',
                                                   pred_flag='gt')

                    if 'tracking' in opt.heads:
                        # first param: current
                        # second param: previous - current
                        if dets_gt[i][k][27] == 1:
                            debugger.add_arrow(
                                [(dets_gt[i][k][0] + dets_gt[i][k][2]) / 2,
                                 (dets_gt[i][k][1] + dets_gt[i][k][3]) / 2, ],
                                [dets_gt[i][k][25], dets_gt[i][k][26]],
                                img_id='out_img_gt')  # cyan-blue
                            debugger.add_arrow(
                                [(dets_gt[i][k][0] + dets_gt[i][k][2]) / 2,
                                 (dets_gt[i][k][1] + dets_gt[i][k][3]) / 2, ],
                                [dets_gt[i][k][25], dets_gt[i][k][26]],
                                img_id='pre_hm')  # cyan-blue

                    if 'tracking_hp' in opt.heads:

                        for idx in range(8):

                            # tracking_hp_mask == 0 then continue
                            if dets_gt[i][k][44 + idx * 2] == 0 or dets_gt[i][k][44 + idx * 2 + 1] == 0:
                                continue

                            debugger.add_arrow(
                                dets_gt[i][k][5 + idx * 2:5 + idx * 2 + 2],
                                dets_gt[i][k][28 + idx * 2:28 + idx * 2 + 2],
                                img_id='out_img_gt', c=(0, 255, 0))  # green
                            debugger.add_arrow(
                                dets_gt[i][k][5 + idx * 2:5 + idx * 2 + 2],
                                dets_gt[i][k][28 + idx * 2:28 + idx * 2 + 2],
                                img_id='pre_hmhp', c=(0, 255, 0))  # green
            # Blended
            debugger.add_img(img, img_id='out_pred_gt_blend')
            for k in range(len(dets['scores'][i])):
                if dets['scores'][i][k][0] > opt.center_thresh:
                    debugger.add_coco_hp(dets['kps'][i][k], img_id='out_pred_gt_blend')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_hp(dets_gt[i, k, 5:21], img_id='out_pred_gt_blend', pred_flag='gt')

            if opt.hm_hp:
                gt = debugger.gen_colormap_hp(batch['hm_hp'][i][choice_list[i]].detach().cpu().numpy())
                debugger.add_blend_img(img, gt, 'out_hmhp_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            elif opt.debug == 5:  # return result, wait for further processing
                pass
            else:
                debugger.show_all_imgs(pause=True)

        return debugger.imgs
