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
from models.FFA import FFA

from DeBug.inference import convert_disp_to_depth,convert_depth_to_disp,recover_depth,recover_clear_images,recover_haze_images,depth2trans,trans2depth
from losses.DSSMDLoss import Disparity_Loss,TransmissionMap_Loss,Airlight_Loss,RecoveredCleanImagesLoss
import os
import time

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).type_as(x))

# Keys 
complete_data=['clear_left_image','left_disp',
                    'focal_length','baseline','beta','airlight']


class DisparityTrainer(object):
    def __init__(self, lr, devices, dataset, trainlist, vallist, datapath, 
                 batch_size, maxdisp,use_deform=False, pretrain=None, 
                        model='FFANet', test_batch=4,initial_pretrain=None):
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
        self.model = model
        self.initialize()
        
    # Get Dataset Here
    def _prepare_dataset(self):
        if self.dataset == 'sceneflow':
            train_transform_list = [complex_transforms.RandomCrop(320,400),
                                    complex_transforms.RandomVerticalFlip(),
                            complex_transforms.ToTensor()
                            ]
            train_transform =complex_transforms.Compose(train_transform_list)

            val_transform_list = [complex_transforms.ToTensor()
                         ]
            val_transform = complex_transforms.Compose(val_transform_list)
            
            train_dataset = ComplexStereoDataset(data_dir=self.datapath,train_datalist=self.trainlist,test_datalist=self.vallist,
                                    dataset_name='SceneFlow',mode='train',transform=train_transform,visible_list=complete_data)
            test_dataset = ComplexStereoDataset(data_dir=self.datapath,train_datalist=self.trainlist,test_datalist=self.vallist,
                                    dataset_name='SceneFlow',mode='val',transform=val_transform,visible_list=complete_data)

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
        if self.model == 'FFANet':
            self.net = FFA(gps=3,blocks=19)
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

    def _set_loss_function(self):
        self.recover_loss = nn.L1Loss(size_average=True,reduction='mean')


    def initialize(self):
        self._prepare_dataset()
        self._build_net()
        self._build_optimizer()
        self._set_loss_function()

    def adjust_learning_rate(self, epoch):
        if epoch>=0 and epoch<=10:
            cur_lr = 2e-4
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
        
        # Loss Function Designer
        losses_meter = AverageMeter()
        
        # PSNR
        psnr_meter = AverageMeter()
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
            clear_left = torch.autograd.Variable(sample_batched['clear_left_image'].cuda(), requires_grad=False)
            target_disp_left = torch.autograd.Variable(sample_batched['left_disp'].cuda(), requires_grad=False).unsqueeze(1)
            focal_length = torch.autograd.Variable(sample_batched['focal_length'].cuda(), requires_grad=False)
            baseline = torch.autograd.Variable(sample_batched['baseline'].cuda(), requires_grad=False)
            beta = torch.autograd.Variable(sample_batched['beta'].cuda(), requires_grad=False)
            airlight = torch.autograd.Variable(sample_batched['airlight'].cuda(), requires_grad=False)
            
            left_depth = convert_disp_to_depth(baseline=baseline,focal_length=focal_length,disp=target_disp_left)
            left_trans = depth2trans(left_depth,beta=beta)
            
            haze_left = recover_haze_images(clean_images=clear_left,beta=beta,A=airlight,depth=left_depth)
            
            haze_left = haze_left.float()
            target_disp_left = target_disp_left.float()
            focal_length = focal_length.float()
            airlight = airlight.float()
            beta = beta.float()
            left_depth = left_depth.float()
            left_trans = left_trans.float()
        
            data_time.update(time.time() - end)
            self.optimizer.zero_grad()

            # Inference Here 
            if self.model =="FFANet":
                # get the disparity and
                dehaze_image = self.net(haze_left) 
            
            # Disparity Loss: Target Disparity
            recover_loss = self.recover_loss(dehaze_image,clear_left)
            total_loss = recover_loss
            total_loss = total_loss.float()

            
            # Evaluation 
            # disp_epe = self.epe(disparity_output,target_disp_left)
            img_loss = img2mse(dehaze_image,clear_left)
            psnr = mse2psnr(img_loss)
            
            
            losses_meter.update(total_loss.data.item(),clear_left.size(0))
            psnr_meter.update(psnr.data.item(),clear_left.size(0))            

            
            summary_writer.add_scalar("Total_loss",losses_meter.val,iterations+1)
            summary_writer.add_scalar("PSNR",psnr_meter.val,iterations+1)


            # Save Some Images
            if i_batch in [0,nums_samples//2,nums_samples//4*3]:

                img_summary = dict()
                img_summary['Haze_left'] = haze_left.detach()
                img_summary['clear_left'] = clear_left.detach()
                img_summary['dehaze_left'] = dehaze_image.detach()

                save_images(summary_writer, 'train' + str(train_count), img_summary, epoch)
                train_count = train_count +1
            
            # compute gradient and do SGD step
            with torch.autograd.detect_anomaly():
                total_loss.backward()
                
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
                'PSNR {psnr_m.val:.3f} ({psnr_m.avg:.3f})\t'
                .format(
                epoch, i_batch, self.num_batches_per_epoch, 
                batch_time=batch_time,
                loss  = losses_meter,
                psnr_m = psnr_meter,
                data_time=data_time))


        return losses_meter.avg, psnr_meter.avg,iterations
    

    # Validation One Epoch
    def validate(self,summary_writer,epoch,vis=False):
          
        batch_time = AverageMeter()

        psnr_meters = AverageMeter()
        
        # switch to evaluate mode
        self.net.eval()
        end = time.time()
        inference_time = 0
        img_nums = 0
        nums_samples = len(self.test_loader)
        test_count = 0
        
        for i, sample_batched in enumerate(self.test_loader):
            
    
            clear_left = torch.autograd.Variable(sample_batched['clear_left_image'].cuda(), requires_grad=False)
            target_disp_left = torch.autograd.Variable(sample_batched['left_disp'].cuda(), requires_grad=False).unsqueeze(1)
            focal_length = torch.autograd.Variable(sample_batched['focal_length'].cuda(), requires_grad=False)
            baseline = torch.autograd.Variable(sample_batched['baseline'].cuda(), requires_grad=False)
            beta = torch.autograd.Variable(sample_batched['beta'].cuda(), requires_grad=False)
            airlight = torch.autograd.Variable(sample_batched['airlight'].cuda(), requires_grad=False)
            
            left_depth = convert_disp_to_depth(baseline=baseline,focal_length=focal_length,disp=target_disp_left)
            left_depth_l = F.interpolate(left_depth,size=[clear_left.shape[-2],clear_left.shape[-1]],mode='bilinear',
                                         align_corners=False)
            
            left_trans = depth2trans(left_depth,beta=beta)
            haze_left = recover_haze_images(clean_images=clear_left,beta=beta,A=airlight,depth=left_depth_l)

            haze_left = haze_left.float()
            target_disp_left = target_disp_left.float()
            focal_length = focal_length.float()
            airlight = airlight.float()
            beta = beta.float()
            left_depth = left_depth.float()
            left_trans = left_trans.float()

        
            with torch.no_grad():
                start_time = time.perf_counter()
                # Get the predicted disparity
                if self.model=="FFANet":
                    dehaze_image = self.net(haze_left)

                    
                inference_time += time.perf_counter() - start_time
                img_nums += haze_left.shape[0]

                img_loss = img2mse(dehaze_image,clear_left)
                psnr = mse2psnr(img_loss)
                
 
            psnr_meters.update(psnr.data.item(),haze_left.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % 10 == 0:
                logger.info('Test: [{0}/{1}]\t Time {2}\t PSNR {3}\t '
                      .format(i, len(self.test_loader), batch_time.val, psnr_meters.val))
            

            if i in [0,nums_samples//2,nums_samples//4*3]:
                img_summary = dict()
                img_summary['Haze_left'] = haze_left.detach()
                img_summary['clear_left'] = clear_left.detach()
                img_summary['dehaze_left'] = dehaze_image.detach()
                img_summary = dict()
           

                # img_summary['dehaze_left'] = dehazed_left.detach()
                save_images(summary_writer, 'test' + str(test_count), img_summary, epoch)
                test_count = test_count +1
                
        # logger.info(' * PSNR {:.3f}'.format(psnr_meters.avg))
        logger.info(' * PSNR meter {:.3f}'.format(psnr_meters.avg))
        logger.info(' * avg inference time {:.3f}'.format(inference_time / img_nums))
        return psnr_meters.avg


    def get_model(self):
        return self.net.state_dict()