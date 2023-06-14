from copy import deepcopy
import os
import numpy as np
import pandas as pd
import torch
import time
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch import nn
from nmrpred.utils.torch_util import set_model_parameters
from itertools import chain
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    Parameters
    ----------
    """
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 requires_dr,
                 device,
                 yml_path,
                 output_path,
                 script_name,
                 lr_scheduler,
                 initial_lr=None,
                 normalizer=None,
                 checkpoint_log=1,
                 checkpoint_val=1,
                 checkpoint_test=20,
                 checkpoint_model=1,
                 verbose=False,
                 training=True,
                 hooks=None,
                 save_validation_data=False,
                 save_batch_data=False,
                 preempt=False,
                 test_names=None,
                 use_resample_algorithm=False,
                 nni_module=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.requires_dr = requires_dr
        self.device = device
        self.preempt = preempt
        self.use_resample_algorithm = use_resample_algorithm
        self.current_lr = initial_lr

        if type(device) is list and len(device) > 1:
            self.multi_gpu = True
        else:
            self.multi_gpu = False
        self.verbose = verbose
        self.normalizer = normalizer

        # outputs
        last_checkpoint = self._subdirs(yml_path, output_path, script_name)
        print("Output path: %s"%self.output_path)
        
        if training:

            # hooks
            if hooks:
                self.hooks = None
                self._hooks(hooks)

            # learning rate scheduler
            self._handle_scheduler(lr_scheduler, optimizer)
            self.lr_scheduler = lr_scheduler

        # checkpoints
        self.check_log = checkpoint_log
        self.check_val = checkpoint_val
        self.check_test = checkpoint_test
        self.check_model = checkpoint_model

        # checkpoints
        self.epoch = 0  # number of epochs of any steps that model has gone through so far
        self.log_loss = {
            'epoch': [],
            'loss(MSE)': [],
            'lr': [],
            'time': []
        }
        if test_names is None:
            test_names = ['test']
        self.test_names = test_names
        for category in ['tr', 'val'] + test_names:
            self.log_loss[category + '_err(RMSE)'] = []
        for category in test_names:
            self.log_loss[category + '_exclusion_rate'] = []
        self.best_val_loss = float("inf")
        self.save_validation_data = save_validation_data
        self.save_batch_data = save_batch_data

        if preempt and last_checkpoint is not None:
            self.resume_model(last_checkpoint)

        if nni_module is not None:
            self.nni_trainer = True
            self.nni_module = nni_module
            tb_log_path = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard')
        else:
            self.nni_trainer = False
            tb_log_path = self.output_path
        self.summary_writer = SummaryWriter(tb_log_path)
        
            

    def _handle_scheduler(self, lr_scheduler, optimizer):
        if lr_scheduler is None:
            self.scheduler = None


        elif lr_scheduler[0] == 'plateau':
            self.scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                               mode='min',
                                               patience=lr_scheduler[2],
                                               factor=lr_scheduler[3],
                                               min_lr=lr_scheduler[4])
        elif lr_scheduler[0] == 'decay':
            lambda1 = lambda epoch: np.exp(-epoch * lr_scheduler[1])
            self.scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda1, verbose=self.verbose)

        else:
            raise NotImplemented('scheduler "%s" is not implemented yet.'%lr_scheduler[0])

    def _subdirs(self, yml_path, output_path, script_name):
        last_checkpoint = None
        # create output directory and subdirectories
        path_iter = output_path[1]
        out_path = os.path.join(output_path[0], 'training_%i'%path_iter)
        while os.path.exists(out_path):
            path_iter+=1
            out_path = os.path.join(output_path[0],'training_%i'%path_iter)
        if self.preempt:
            # check whether can continue from last path_iter
            path_iter-=1
            parent_folder = os.path.join(output_path[0],'training_%i'%path_iter)
            if not os.path.exists(parent_folder) or 'preempt_lock' in os.listdir(parent_folder) \
                 or '1' not in os.listdir(parent_folder):
                # continue from last job not allowed or not possible
                path_iter+=1
                out_path = os.path.join(output_path[0],'training_%i'%path_iter, '1')
            else:
                # can continue last preempted job
                last_preempted = max([int(n) for n in os.listdir(parent_folder)])
                last_checkpoint = os.path.join(parent_folder, str(last_preempted), 'models/model_state.tar')
                out_path = os.path.join(output_path[0],'training_%i'%path_iter, str(last_preempted + 1))
                # make sure resume from a run that has saved model state
                while not os.path.exists(last_checkpoint):
                    last_preempted -= 1
                    if last_preempted == 0:
                        raise RuntimeError("Cannot find a checkpoint to resume from!")
                    last_checkpoint = os.path.join(parent_folder, str(last_preempted), 'models/model_state.tar')

        os.makedirs(out_path)
        self.output_path = out_path

        self.val_out_path = os.path.join(self.output_path, 'validation')
        os.makedirs(self.val_out_path)

        # subdir for computation graph
        self.graph_path = os.path.join(self.output_path, 'graph')
        if not os.path.exists(self.graph_path):
            os.makedirs(self.graph_path)

        # saved models
        self.model_path = os.path.join(self.output_path, 'models')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        script_out = os.path.join(self.output_path, 'run_scripts')
        os.makedirs(script_out)
        shutil.copyfile(yml_path, os.path.join(script_out,os.path.basename(yml_path)))
        shutil.copyfile(script_name, os.path.join(script_out,script_name))
        return last_checkpoint

    def _hooks(self, hooks):
        hooks_list = []
        if 'vismolvector3d' in hooks and hooks['vismolvector3d']:
            from nmrpred.train.hooks import VizMolVectors3D

            vis = VizMolVectors3D()
            vis.set_output(True, None)
            hooks_list.append(vis)

        if len(hooks_list) > 0:
            self.hooks = hooks_list


    def print_layers(self):
        total_n_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
                if len(param.shape) > 1:
                    total_n_params += param.shape[0] * param.shape[1]
                else:
                    total_n_params += param.shape[0]
        print('\n total trainable parameters: %i\n' % total_n_params)

    def plot_grad_flow(self):
        ave_grads = []
        layers = []
        for n, p in self.model.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                # shorten names
                layer_name = n.split('.')
                layer_name = [l[:3] for l in layer_name]
                layer_name = '.'.join(layer_name[:-1])
                layers.append(layer_name)
                # print(layer_name, p.grad)
                if p.grad is not None:
                    ave_grads.append(p.grad.abs().mean().detach().cpu())
                else:
                    ave_grads.append(0)

        fig, ax = plt.subplots(1, 1)
        ax.plot(ave_grads, alpha=0.3, color="b")
        ax.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow: epoch#%i" %self.epoch)
        plt.grid(True)
        ax.set_axisbelow(True)

        file_name= os.path.join(self.graph_path,"avg_grad.png")
        plt.savefig(file_name, dpi=300,bbox_inches='tight')
        plt.close(fig)

    def store_checkpoint(self, input, steps):
        # csv logging
        self.log_loss['epoch'].append(self.epoch)
        for k in input:
            self.log_loss[k].append(input[k])


        df = pd.DataFrame(self.log_loss)
        df.applymap('{:.5f}'.format).to_csv(os.path.join(
            self.output_path, 'log.csv'),
                                           index=False)

        print("[%d, %3d]" % (self.epoch, steps), end="")
        for k in input:
            print("%s: %.5f; " % (k, input[k]), end="")
        print("\n")

        # tensorboard logging
        for k in input:
            self.summary_writer.add_scalar(k, input[k], self.epoch)
        #  loss_mse: %.5f; "
        #     "tr_E(MAE): %.5f; tr_F(MAE): %.5f; "
        #     "val_E(MAE): %.5f; val_F(MAE): %.5f; "
        #     "irc_E(MAE): %.5f; irc_F(MAE): %.5f; "
        #     "test_E(MAE): %.5f; test_F(MAE): %.5f; "
        #     "lr: %.9f; epoch_time: %.3f\n"
        #     % (self.epoch, steps, np.sqrt(input[0]),
        #        input[1], input[2], input[3], input[4], input[5], input[6],
        #        input[7], input[8], input[9], input[10]))

    def _optimizer_to_device(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device[0])

    

    def metric_rmse(self, preds, data, mask=None, normalizer=None):
        """Root mean square error"""
        if normalizer is not None:
            mean, std = normalizer
            preds = preds * std + mean
        preds = preds.squeeze()
        data = data.squeeze()
        if preds.shape != data.shape:
            assert data[mask].shape == preds.shape
            se = np.square(preds - data[mask])
            return np.sqrt(np.mean(se))
        else:
            se = np.square(preds - data)
            if mask is not None:
                se *= mask
                return np.sqrt(np.sum(se) / (np.sum(mask) + 1e-7))
            else:
                return np.sqrt(np.mean(se))

    def masked_average(self, y, atom_mask):
        """

        Parameters
        ----------
        y: numpy array
        atom_mask: numpy array

        Returns
        -------

        """
        # handle rotation-wise loader batch size mismatch
        if atom_mask.shape[0] > y.shape[0]:
            # assert atom_mask.shape[1] == y.shape[1]
            atom_mask = atom_mask.reshape(y.shape[0], -1, y.shape[1])  # B, n_rot, A
            atom_mask = atom_mask.mean(axis=1)

        # size = np.sum(atom_mask, axis=1, keepdims=True)
        # size = np.maximum(size, np.ones_like(size))
        # y = np.sum(y, axis=1)
        # y = y / size

        y = y[atom_mask!=0]

        return y

    def resume_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        if checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Resumed training from checkpoint [%s]" % path)
        return loss


    def validation_atomic_properties(self, name, generator, steps, error_include_fn=None):
        self.model.eval()
        self.model.requires_dr = False

        val_rmse = []
        pred = []
        targets = []
        masks = []
        labels = []
        batch_data = []
        # e = []
        # f = []
        # ei = []
        # fi = []
        AM = []
        RM = []  # rotation angles/matrix

        for val_step in range(steps):
            val_batch = next(generator)
            with torch.no_grad():
                val_preds = self.model(val_batch)
            batch_target = val_batch["targets"]
            if 'M' in val_batch:
                mask = val_batch['M'].detach().cpu().numpy()
                masks.append(mask)
            else:
                mask = None

            val_pred_np = val_preds.detach().cpu().numpy()
            batch_target_np = batch_target.detach().cpu().numpy()
            
            pred.append(val_pred_np)
            targets.append(batch_target_np)
            if "labels" in val_batch:
                labels.extend(val_batch["labels"])


            if self.verbose:

                print(
                    "%s: %i/%i - %s_RMSE: %.5f"
                    % (name, val_step, steps,
                    "target", np.mean(np.array(val_rmse))
                       ))

            if self.save_batch_data:
                batch_data.append(val_batch)

        pred = np.concatenate(pred, axis=0)
        targets = np.concatenate(targets, axis=0)
        if error_include_fn is not None:
            inclusion = error_include_fn(np.abs(pred - targets))
        else:
            inclusion = np.ones_like(pred, dtype=bool)
        if len(masks) > 0:
            mask = np.concatenate(masks, axis=0)
        else:
            mask = np.ones_like(pred, dtype=bool)
        exclude_rate = (mask & (~inclusion)).sum() / mask.sum()
        final_mask = mask & inclusion
        val_rmse = self.metric_rmse(
                pred,
                targets,
                mask=final_mask,
                normalizer=self.normalizer
            ) # not sure if this is correct
        outputs = dict()
        outputs['pred'] = pred
        outputs['targets'] = targets
        outputs['RMSE'] = val_rmse
        outputs['exclude_rate'] = exclude_rate
        outputs["labels"] = labels
        outputs["batch_data"] = batch_data
        return outputs

    def get_resample_weights(self, train_batch, val_batch):
        pred_model = deepcopy(self.model)
        dual_model = deepcopy(self.model)
        preds_train = pred_model(train_batch)
        sample_weights = torch.zeros_like(preds_train, requires_grad=True)
        train_loss = self.loss_fn(preds_train, train_batch, sample_weights=sample_weights)

        params = {k:v for k,v in pred_model.named_parameters()}
        param_grads = torch.autograd.grad(train_loss, params.values(), create_graph=True)
        param_grads = dict(zip(params.keys(), param_grads))
        updated_state_dict = {k: params[k].detach() - self.current_lr * param_grads[k] for k in params}
        set_model_parameters(dual_model, updated_state_dict)

        preds_val = dual_model(val_batch)
        val_loss = self.loss_fn(preds_val, val_batch)
        sample_weight_grads = torch.autograd.grad(val_loss, sample_weights)[0].detach()

        # calculate adjusted sample weights from gradients
        clamped_grads = torch.maximum(-sample_weight_grads, torch.tensor(0., device=sample_weight_grads.device))
        grads_sum = torch.sum(clamped_grads) + 1e-8
        new_sample_weights = clamped_grads / grads_sum

        for p in pred_model.parameters():
            p.grad = None   
        torch.cuda.empty_cache()
        return new_sample_weights


    def train(self,
              train_generator,
              epochs,
              steps,
              val_generator=None,
              val_steps=None,
              test_generators=None,
              test_steps=None,
              clip_grad=0,
              resample_val_generator=None,
              err_inclusion_fn=None,):
        """
        The main function to train model for the given number of epochs (and steps per epochs).
        The implementation allows for resuming the training with different data and number of epochs.

        Parameters
        ----------
        epochs: int
            number of training epochs

        steps: int
            number of steps to call it an epoch (designed for nonstop data generators)


        """
        self.model.to(self.device[0])
        if self.multi_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.device)
        self._optimizer_to_device()

        running_val_loss = []
        last_test_epoch = -100

        # prepare test generators so that it is a list
        if test_generators is not None:
            if type(test_generators) is not list:
                test_generators = [test_generators]
                test_steps = [test_steps]

        while 1:
            t0 = time.time()

            # record total number of epochs so far
            self.epoch += 1
            if self.epoch > epochs:
                break


            # training
            running_loss = 0.0
            rmse_ai = []
            n_data = 0
            n_atoms = 0
            self.model.train()
            self.optimizer.zero_grad()
            step_iterator = range(steps)
            if not self.verbose:
                step_iterator = tqdm(step_iterator)

            for s in step_iterator:
                self.optimizer.zero_grad()

                train_batch = next(train_generator)
                # self.model.module(train_batch)
                # preds = self.model.forward(train_batch)
                if self.use_resample_algorithm:
                    val_batch = next(resample_val_generator)
                    sample_weights = self.get_resample_weights(train_batch, val_batch)
                else:
                    sample_weights = None
                preds = self.model(train_batch)
                if sample_weights is not None:
                    loss = self.loss_fn(preds, train_batch, sample_weights=sample_weights)
                else:
                    loss = self.loss_fn(preds, train_batch)
                loss.backward()
                if clip_grad>0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                self.optimizer.step()
                # if (s+1)%4==0 or s==steps-1:
                #     self.optimizer.step()          # if in, comment out the one in the loop
                #     self.optimizer.zero_grad()     # if in, comment out the one in the loop

                current_loss = loss.detach().item()
                running_loss += current_loss

                # atom_mask = train_batch["AM"].detach().cpu().numpy()
                # n_atoms += np.sum(atom_mask)

                # decide whether to apply mask
                if 'M' in train_batch:
                    mask = train_batch['M'].detach().cpu().numpy()
                else:
                    mask = None
                # calc rmse
                rmse_ai.append(np.mean(self.metric_rmse(
                    preds.detach().cpu().numpy(),
                    train_batch["targets"].detach().cpu().numpy(),
                    mask=mask,
                    normalizer=self.normalizer
                )))

                n_data += train_batch["targets"].size()[0]

                if self.verbose:
                    print(datetime.now(),
                        "Train: Epoch %i/%i - %i/%i - loss: %.5f - running_loss(RMSE): %.5f - RMSE: %.5f"
                        % (self.epoch, epochs, s, steps, current_loss,
                        np.sqrt(running_loss / (s + 1)),
                        (np.mean(rmse_ai[-100:]))
                        ))
                del train_batch

            running_loss /= steps

            rmse_ai = np.mean(rmse_ai[-100:])

            # plots
            self.plot_grad_flow()

            # validation
            val_error = float("inf")
            if val_generator is not None and \
                self.epoch % self.check_val == 0:

                outputs = self.validation_atomic_properties('valid', val_generator, val_steps, err_inclusion_fn)
                if self.save_validation_data:
                    torch.save(outputs, os.path.join(self.val_out_path, "validation" + "_" + str(self.epoch) + '.sav'))
                val_error = outputs["RMSE"]
                print("val_error", val_error)

            if self.multi_gpu:
                save_model = self.model.module
            else:
                save_model = self.model

            # checkpoint every epoch when preempt is allowed
            if self.preempt:
                state_dict = {
                            'epoch': self.epoch,
                            'model_state_dict': save_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss
                            }
                if self.lr_scheduler is not None:
                    state_dict['scheduler_state_dict'] = self.scheduler.state_dict()
                else:
                    state_dict['scheduler_state_dict'] = None
                torch.save(state_dict,
                    os.path.join(self.model_path, 'model_state.tar'))

            # checkpoint model when validation error gets lower
            test_error = 0
            if self.best_val_loss > val_error:
                self.best_val_loss = val_error

                # torch.save({"model": save_model, "normalizer": self.normalizer},#.state_dict(),
                torch.save(save_model,#.state_dict(),
                           os.path.join(self.model_path, 'best_model.pt'))
                state_dict = {
                            'epoch': self.epoch,
                            'model_state_dict': save_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss
                            }
                if self.lr_scheduler is not None:
                    state_dict['scheduler_state_dict'] = self.scheduler.state_dict()
                else:
                    state_dict['scheduler_state_dict'] = None
                torch.save(state_dict,
                    os.path.join(self.model_path, 'best_model_state.tar')
                )

                

                # save test predictions
                if test_generators is not None and self.epoch - last_test_epoch >= self.check_test:
                    test_errors = []
                    test_exclusion_rates = []
                    for test_generator, step, test_name in zip(test_generators, test_steps, self.test_names):
                        outputs = self.validation_atomic_properties(test_name, test_generator, step, err_inclusion_fn)
                        if self.save_validation_data:
                            torch.save(outputs, os.path.join(self.val_out_path, test_name + "_" + str(self.epoch) + '_results.sav'))
                        test_errors.append(outputs["RMSE"])
                        test_exclusion_rates.append(outputs["exclude_rate"])
                    last_test_epoch = self.epoch
                    # np.save(os.path.join(self.val_out_path, 'test_Ei_epoch%i'%self.epoch), outputs['Ei'])

            # learning rate decay
            if self.lr_scheduler is not None:
                if self.lr_scheduler[0] == 'plateau':
                    running_val_loss.append(val_error)
                    if len(running_val_loss) > self.lr_scheduler[1]:
                        running_val_loss.pop(0)
                    accum_val_loss = np.mean(running_val_loss)
                    self.scheduler.step(accum_val_loss)
                elif self.lr_scheduler[0] == 'decay':
                    self.scheduler.step()
                    accum_val_loss = 0.0

            # logging
            if self.epoch % self.check_log == 0:

                for i, param_group in enumerate(
                        self.optimizer.param_groups):
                        # self.scheduler.optimizer.param_groups):
                    old_lr = float(param_group["lr"])

                self.current_lr = old_lr
                checkpoint_results = {
                    "loss(MSE)": running_loss,
                    "tr_err(RMSE)": rmse_ai,
                    "val_err(RMSE)": val_error,
                    "lr": old_lr,
                    "time": time.time() - t0
                }

                metric = {"default": val_error}
                for test_name, test_error, test_exclusion_rate in zip(self.test_names, test_errors, test_exclusion_rates):
                    checkpoint_results[test_name + "_err(RMSE)"] = test_error
                    checkpoint_results[test_name + "_exclusion_rate"] = test_exclusion_rate
                    metric[test_name] = test_error
                self.store_checkpoint(checkpoint_results, steps)
                if self.nni_trainer:
                    self.nni_module.report_intermediate_result(metric)
        
        return metric

    def log_statistics(self, splits, normalizer, target_hash=None):
        with open(os.path.join(self.output_path, "stats.txt"), "w") as f:
            for item in splits:
                f.write(f"{item} data: %d\n" % len(splits[item])) 

            f.write("Normalizer: %s\n" % str(normalizer))
            if target_hash is not None:
                f.write("Test target hash: %s\n" % target_hash)
