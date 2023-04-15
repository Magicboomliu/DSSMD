import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.AverageMeter import AverageMeter
from utils.common import logger, check_path, write_pfm,count_parameters

from dataloader.ComplexSceneflowLoader import ComplexStereoDataset
from dataloader import complex_transforms
from dataloader.preprocess import scale_disp
# metric
from utils.metric import P1_metric,P1_Value,D1_metric,Disparity_EPE_Loss
from utils.visual import save_images,disp_error_img
# Loading the Nework Architecture
from models.DSSMD import DSSMMD
from DeBug.inference import convert_disp_to_depth,convert_depth_to_disp,recover_depth,recover_clear_images,recover_haze_images,depth2trans,trans2depth
from losses.DSSMDLoss import Disparity_Loss,TransmissionMap_Loss,Airlight_Loss,RecoveredCleanImagesLoss
import os
import time

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

# Keys 
complete_data=['clear_left_image','clear_right_image','left_disp','right_disp',
                    'focal_length','baseline','beta','airlight']


class DisparityTrainer(object):
    def __init__(self, lr, devices, dataset, trainlist, vallist, datapath, 
                 batch_size, maxdisp,use_deform=False, pretrain=None, 
                        model='DSSMD', test_batch=4,initial_pretrain=None):
        super(DisparityTrainer, self).__init__()
        
        self.lr = lr
        self.initial_pretrain = initial_pretrain
        self.current_lr = lr
        self.devices = devices
        self.devices = [int(item) for item in devices.split(',')]
        ngpu = len(devices)
        self.ngpu = ngpu
        self.trainlist = trainlist
        self.vallist = vallist
        self.dataset = dataset
        self.datapath = datapath
        self.batch_size = batch_size
        self.test_batch = test_batch
        self.pretrain = pretrain 
        self.maxdisp = maxdisp
        self.use_deform= use_deform
        self.criterion = None
        self.epe = Disparity_EPE_Loss
        self.p1_error = P1_metric
        self.model = model
        self.initialize()
    # Get Dataset Here
    def _prepare_dataset(self):
        if self.dataset == 'sceneflow':
            train_transform_list = [complex_transforms.RandomCrop(320, 640),
                            complex_transforms.ToTensor()
                            ]
            train_transform =complex_transforms.Compose(train_transform_list)

            val_transform_list = [complex_transforms.ToTensor()
                         ]
            val_transform = complex_transforms.Compose(val_transform_list)
            
            train_dataset = ComplexStereoDataset(data_dir=self.datapath,train_datalist=self.trainlist,test_datalist=self.vallist,
                                    dataset_name='SceneFlow',mode='train',transform=train_transform)
            test_dataset = ComplexStereoDataset(data_dir=self.datapath,train_datalist=self.trainlist,test_datalist=self.vallist,
                                    dataset_name='SceneFlow',mode='val',transform=val_transform)

        self.img_height, self.img_width = train_dataset.get_img_size()

        self.scale_height, self.scale_width = test_dataset.get_scale_size()

        datathread=4
        if os.environ.get('datathread') is not None:
            datathread = int(os.environ.get('datathread'))
        logger.info("Use %d processes to load data..." % datathread)

        self.train_loader = DataLoader(train_dataset, batch_size = self.batch_size, \
                                shuffle = True, num_workers = datathread, \
                                pin_memory = True)

        self.test_loader = DataLoader(test_dataset, batch_size = self.test_batch, \
                                shuffle = False, num_workers = datathread, \
                                pin_memory = True)
        self.num_batches_per_epoch = len(self.train_loader)

    def _build_net(self):
        # Build the Network architecture according to the model name
        if self.model == 'DSSMD':
            self.net = DSSMMD(dehaze_switch=True,in_channels=3)
        else:
            raise NotImplementedError
        
        self.is_pretrain = False
        if self.ngpu > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=self.devices).cuda()
        else:
            # self.net.cuda()
            self.net = torch.nn.DataParallel(self.net, device_ids=self.devices).cuda()
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in self.net.parameters()])))

        if self.pretrain == 'none':
            logger.info('Initial a new model...')
            if self.initial_pretrain !='none':
                pretrain_ckpt = self.initial_pretrain
                print("Loading the Model with Some initial Weights........")
                ckpt = torch.load(pretrain_ckpt)
                current_model_dict = self.net.state_dict()
                useful_dict ={k:v for k,v in ckpt['state_dict'].items() if k in current_model_dict.keys()}
                print("{}/{} has been re-used in this training".format(len(useful_dict) ,len(ckpt['state_dict'])))
                current_model_dict.update(useful_dict)
                self.net.load_state_dict(current_model_dict)
        else:
            if os.path.isfile(self.pretrain):
                model_data = torch.load(self.pretrain)
                logger.info('Load pretrain model: %s', self.pretrain)
                if 'state_dict' in model_data.keys():
                    self.net.load_state_dict(model_data['state_dict'])
                else:
                    self.net.load_state_dict(model_data)
                self.is_pretrain = True
            else:
                logger.warning('Can not find the specific model %s, initial a new model...', self.pretrain)

    def _build_optimizer(self):
        beta = 0.999
        momentum = 0.9
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), self.lr,
                                        betas=(momentum, beta), amsgrad=True)

    def initialize(self):
        self._prepare_dataset()
        self._build_net()
        self._build_optimizer()

    def adjust_learning_rate(self, epoch):
        if epoch>=0 and epoch<=10:
            cur_lr = 3e-4
        elif epoch > 10 and epoch<45:
            cur_lr = 1e-4
        elif epoch>=40 and epoch<50:
            cur_lr = 5e-5
        elif epoch>=50 and epoch<60:
            cur_lr = 3e-5
        elif epoch>=60:
            cur_lr =1.5e-5
        else:
            cur_lr = self.lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr
        self.current_lr = cur_lr
        return cur_lr


    # Train One Epoch
    def train_one_epoch(self, epoch, round,iterations,summary_writer):
        
        # Data Summary
        batch_time = AverageMeter()
        data_time = AverageMeter()    
        
        losses = AverageMeter()
        disp_loss = AverageMeter()
        transmision_loss = AverageMeter()
        airlight_loss = AverageMeter()
        recovered_rgb_loss = AverageMeter()

        # PSNR
        nums_samples = len(self.train_loader)
        train_count = 0
        
        # non-detection
        torch.autograd.set_detect_anomaly(True)
        
        # switch to train mode
        self.net.train()
        end = time.time()
        cur_lr = self.adjust_learning_rate(epoch)
        logger.info("learning rate of epoch %d: %f." % (epoch, cur_lr))
        summary_writer.add_scalar("Learning_Rate",cur_lr,epoch+1)
        
        
        for i_batch, sample_batched in enumerate(self.train_loader):
            # clear left 
            # clear right
            # target disp
            # beta
            # airlight
            # baseline
            # focal length 

            clear_left = torch.autograd.Variable(sample_batched['clear_left_image'].cuda(), requires_grad=False)
            clear_right = torch.autograd.Variable(sample_batched['clear_right_image'].cuda(), requires_grad=False)
            target_disp_left = torch.autograd.Variable(sample_batched['left_disp'].cuda(), requires_grad=False).unsqueeze(1)
            target_disp_right = torch.autograd.Variable(sample_batched['right_disp'].cuda(), requires_grad=False).unsqueeze(1)
            focal_length = torch.autograd.Variable(sample_batched['focal_length'].cuda(), requires_grad=False)
            baseline = torch.autograd.Variable(sample_batched['baseline'].cuda(), requires_grad=False)
            beta = torch.autograd.Variable(sample_batched['beta'].cuda(), requires_grad=False)
            airlight = torch.autograd.Variable(sample_batched['airlight'].cuda(), requires_grad=False)
            
            left_depth = convert_disp_to_depth(baseline=baseline,focal_length=focal_length,disp=target_disp_left)
            right_depth = convert_disp_to_depth(baseline=baseline,focal_length=focal_length,disp=target_disp_right)
            left_trans = depth2trans(left_depth,beta=beta)
            right_trans = depth2trans(right_depth,beta=beta)
            
            haze_left = recover_haze_images(clean_images=clear_left,beta=beta,A=airlight,depth=left_depth)
            haze_right = recover_haze_images(clean_images=clear_right,beta=beta,A=airlight,depth=right_depth)
            # get left trans and right trans
            # generate left haze image and right haze image.
            

            data_time.update(time.time() - end)
            self.optimizer.zero_grad()

            # Inference Here 
            if self.model =="DSSMD":
                
                pass
            
            
            
            # Loss Meterization
            if type(loss) is list or type(loss) is tuple:
                loss = np.sum(loss)
            if type(output) is list or type(output) is tuple: 
                flow2_EPE = self.epe(output[-1], target_disp)    
            else:
                if output.size(-1)!= target_disp.size(-1):
                    output = F.interpolate(output,scale_factor=8.0,mode='bilinear',align_corners=False) * 8.0
                assert (output.size(-1) == target_disp.size(-1))
                flow2_EPE = self.epe(output, target_disp)

            

            # Record loss and EPE in the tfboard
            losses.update(loss.data.item(), left_input.size(0))
            flow2_EPEs.update(flow2_EPE.data.item(), left_input.size(0))
            losses_unsupervised.update(loss_P.data.item(),left_input.size(0))
            losses_smoothness.update(loss_S.data.item(),left_input.size(0))
            losses_pam.update(loss_PAM.data.item(),left_input.size(0))
            
            summary_writer.add_scalar("Total_loss",losses.val,iterations+1)
            summary_writer.add_scalar("Unsupervised_Disp_Loss",losses_unsupervised.val,iterations+1)
            summary_writer.add_scalar("Smooth_Loss",losses_smoothness.val,iterations+1)
            summary_writer.add_scalar("PAM_Loss",losses_pam.val,iterations+1)
            summary_writer.add_scalar("disp_EPE_on_train",flow2_EPEs.val,iterations+1)


            # Save Some Images
            if i_batch in [0,nums_samples//2,nums_samples//4*3]:
                pred_disp = output.squeeze(1).detach()
                gt_disp = target_disp.squeeze(1).detach()
                valid_mask_vis = valid_mask[-1][0].squeeze(1).float().detach()

                img_summary = dict()
                img_summary['disp_error'] = disp_error_img(pred_disp, gt_disp)
                img_summary['left'] = left_input.detach()
                img_summary['right'] = right_input.detach()
                img_summary['gt_disp'] = gt_disp.detach()
                img_summary['pred_disp'] = pred_disp.detach()
                img_summary['pred_occlusion'] = valid_mask_vis.detach()

                save_images(summary_writer, 'train' + str(train_count), img_summary, epoch)
                train_count = train_count +1
            
            # compute gradient and do SGD step
            with torch.autograd.detect_anomaly():
                loss.backward()
            self.optimizer.step()
            iterations = iterations+1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            

            if i_batch % 10 == 0:
                logger.info('this is round %d', round)
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'Disp_EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})\t'.format(
                epoch, i_batch, self.num_batches_per_epoch, 
                batch_time=batch_time,
                data_time=data_time, loss=losses,flow2_EPE=flow2_EPEs))


        return losses.avg, flow2_EPEs.avg,iterations
    
    
    # Validation One Epoch
    def validate(self,summary_writer,epoch,vis=False):
        
        
        batch_time = AverageMeter()
        flow2_EPEs = AverageMeter()
        P1_errors = AverageMeter()    
        losses = AverageMeter()
        # switch to evaluate mode
        self.net.eval()
        end = time.time()
        inference_time = 0
        img_nums = 0
        nums_samples = len(self.test_loader)
        test_count = 0
        
        for i, sample_batched in enumerate(self.test_loader):
            left_input = torch.autograd.Variable(sample_batched['img_left'].cuda(), requires_grad=False)
            right_input = torch.autograd.Variable(sample_batched['img_right'].cuda(), requires_grad=False)
            input = torch.cat((left_input, right_input), 1)
            input_var = torch.autograd.Variable(input, requires_grad=False)
            target_disp = sample_batched['gt_disp'].unsqueeze(1)
            target_disp = target_disp.cuda()
            target_disp = torch.autograd.Variable(target_disp, requires_grad=False)
 
            
            with torch.no_grad():
                start_time = time.perf_counter()
                # Get the predicted disparity
                if self.model=="PAMStereo":
                    output = self.net(left_input,right_input)
                    output = scale_disp(output, (output.size()[0], self.img_height, self.img_width))  

                inference_time += time.perf_counter() - start_time
                img_nums += left_input.shape[0]
                loss = self.epe(output, target_disp)
                flow2_EPE = self.epe(output, target_disp)
                P1_error = self.p1_error(output, target_disp)
                
            if loss.data.item() == loss.data.item():
                losses.update(loss.data.item(), input_var.size(0))
            if flow2_EPE.data.item() == flow2_EPE.data.item():
                flow2_EPEs.update(flow2_EPE.data.item(), input_var.size(0))
            if P1_error.data.item() == P1_error.data.item():
                P1_errors.update(P1_error.data.item(), input_var.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % 10 == 0:
                logger.info('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                      .format(i, len(self.test_loader), batch_time.val, flow2_EPEs.val))
            
            
            if i in [0,nums_samples//2,nums_samples//4*3]:
                pred_disp = output.squeeze(1).detach()
                gt_disp = target_disp.squeeze(1).detach()
    
                img_summary = dict()
                img_summary['disp_error'] = disp_error_img(pred_disp, gt_disp)
                img_summary['left'] = left_input.detach()
                img_summary['right'] = right_input.detach()
                img_summary['gt_disp'] = gt_disp.detach()
                img_summary['pred_disp'] = pred_disp.detach()
                save_images(summary_writer, 'test' + str(test_count), img_summary, epoch)
                test_count = test_count +1


        logger.info(' * DISP EPE {:.3f}'.format(flow2_EPEs.avg))
        logger.info(' * P1_error {:.3f}'.format(P1_errors.avg))
        logger.info(' * avg inference time {:.3f}'.format(inference_time / img_nums))
        return flow2_EPEs.avg


    def get_model(self):
        return self.net.state_dict()