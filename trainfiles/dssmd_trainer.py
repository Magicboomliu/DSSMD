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
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).type_as(x))

# Keys 
complete_data=['clear_left_image','clear_right_image','left_disp','right_disp',
                    'focal_length','baseline','beta','airlight']



        
def RecoveredCleanFromTrans(transmission_map,airlight,haze_image):
    
    '''
    transmision: [B,1,H,W]
    haze image: [B,3,H,W]
    airlight: [B,1]
    '''
    airlight = airlight.unsqueeze(-1).unsqueeze(-1) #[B,1,1,1]
    
    # if transmission = 0, how to real with?
    if transmission_map.min==0:
        recovered_clean = (haze_image-airlight*(1-transmission_map))/(transmission_map+1e-4)
    else:
        recovered_clean = (haze_image-airlight*(1-transmission_map))/(transmission_map)
        
    recovered_clean = torch.clamp(recovered_clean,min=0,max=1.0)
    
    return recovered_clean




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

    def _set_loss_function(self):
        
        self.disp_loss = Disparity_Loss(type='smooth_l1',weights=[0.6,0.8,1.0,1.0])
        
        self.transmission_loss = TransmissionMap_Loss(type='smooth_l1')
        
        self.airlight_loss = Airlight_Loss(type='l1_loss')
        
        self.recovered_left_loss = RecoveredCleanImagesLoss(type='normal')


    def initialize(self):
        self._prepare_dataset()
        self._build_net()
        self._build_optimizer()
        self._set_loss_function()

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
        
        # Loss Function Designer
        losses_meter = AverageMeter()
        
        disp_loss_meter = AverageMeter()
        transmision_loss_meter = AverageMeter()
        
        airlight_loss_meter = AverageMeter()
        recovered_rgb_loss_meter = AverageMeter()
        
        disp_EPEs = AverageMeter()
        psnr_meters = AverageMeter()


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
            # right_trans = depth2trans(right_depth,beta=beta)
            
            haze_left = recover_haze_images(clean_images=clear_left,beta=beta,A=airlight,depth=left_depth)
            haze_right = recover_haze_images(clean_images=clear_right,beta=beta,A=airlight,depth=right_depth)
            # get left trans and right trans
            # generate left haze image and right haze image.
            
            haze_left = haze_left.float()
            haze_right = haze_right.float()
            target_disp_left = target_disp_left.float()
            target_disp_right = target_disp_right.float()
            focal_length = focal_length.float()
            airlight = airlight.float()
            beta = beta.float()
            left_depth = left_depth.float()
            right_depth = right_depth.float()
            left_trans = left_trans.float()
        
            data_time.update(time.time() - end)
            self.optimizer.zero_grad()

            # Inference Here 
            if self.model =="DSSMD":
                # get the disparity and 
                disparity_pyramid, predicted_transmission,predicted_airlight = self.net(haze_left,haze_right)
                disparity_output = disparity_pyramid[-1]
                
                dehazed_left = RecoveredCleanFromTrans(transmission_map=predicted_transmission,
                                                       airlight=predicted_airlight,
                                                       haze_image=haze_left)
            
            # Disparity Loss: Target Disparity
            disp_loss =self.disp_loss(disparity_pyramid,target_disp_left)
            # Predicted Transmission.

            transmission_loss = F.smooth_l1_loss(predicted_transmission,left_trans,size_average=True) 
            # Predicted Airlight.
            airlight_loss = self.airlight_loss(predicted_airlight,airlight)
            # recovered disparity loss
            # transmission_map,airlight,haze_image,clean_image
            recovered_loss = self.recovered_left_loss(predicted_transmission,predicted_airlight,haze_left,clear_left)
            # Total Loss
            disp_loss = disp_loss.float()
            transmission_loss = transmission_loss.float()
            airlight_loss = airlight_loss.float()
            recovered_loss = recovered_loss.float()
            total_loss = disp_loss*2.0 + transmission_loss*1.0 + airlight_loss*1.0 + recovered_loss*0.8
            total_loss = total_loss.float()
            
            
            # Evaluation 
            disp_epe = self.epe(disparity_output,target_disp_left)
            img_loss = img2mse(dehazed_left,clear_left)
            psnr = mse2psnr(img_loss)
            
            
            losses_meter.update(total_loss.data.item(),clear_left.size(0))
            disp_EPEs.update(disp_epe.data.item(),clear_left.size(0))
            disp_loss_meter.update(disp_loss.data.item(),clear_left.size(0))
            transmision_loss_meter.update(transmission_loss.data.item(),clear_left.size(0))
            airlight_loss_meter.update(airlight_loss.data.item(),clear_left.size(0))
            recovered_rgb_loss_meter.update(recovered_loss.data.item(),clear_left.size(0))
            psnr_meters.update(psnr.data.item(),clear_left.size(0))            

            
            summary_writer.add_scalar("Total_loss",losses_meter.val,iterations+1)
            summary_writer.add_scalar("Disp Loss Meter",disp_loss_meter.val,iterations+1)
            summary_writer.add_scalar("Transmission Meter",transmision_loss_meter.val,iterations+1)
            summary_writer.add_scalar("Airlight loss Meter",airlight_loss_meter.val,iterations+1)
            summary_writer.add_scalar("Disp Loss",disp_EPEs.val,iterations+1)
            summary_writer.add_scalar("Recovered PSNR",psnr_meters.val,iterations+1)


            # Save Some Images
            if i_batch in [0,nums_samples//2,nums_samples//4*3]:
                pred_disp = disparity_output.squeeze(1).detach()
                gt_disp = target_disp_left.squeeze(1).detach()

                img_summary = dict()
                img_summary['disp_error'] = disp_error_img(pred_disp, gt_disp)
                img_summary['left'] = haze_left.detach()
                img_summary['right'] = haze_left.detach()
                img_summary['gt_disp'] = gt_disp.detach()
                img_summary['pred_disp'] = pred_disp.detach()
                img_summary['dehaze_img'] = dehazed_left.detach()

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
                'Disp_Loss {disp_loss.val:.3f} ({disp_loss.avg:.3f})\t'
                'transmission_loss {transmission_l.val:.3f} ({transmission_l.avg:.3f})\t'
                'airlight_loss {airlight_l.val:.3f} ({airlight_l.avg:.3f})\t'
                'recovered_loss {recover_l.val:.3f} ({recover_l.avg:.3f})\t'
                'Disp_EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})\t'
                'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})\t'
                .format(
                epoch, i_batch, self.num_batches_per_epoch, 
                batch_time=batch_time,
                disp_loss = disp_loss_meter,
                transmission_l = transmision_loss_meter,
                airlight_l = airlight_loss_meter,
                recover_l = recovered_rgb_loss_meter,
                data_time=data_time, loss=losses_meter,flow2_EPE=disp_EPEs,
                psnrs=psnr_meters))


        return losses_meter.avg, disp_EPEs.avg,iterations
    

    # Validation One Epoch
    def validate(self,summary_writer,epoch,vis=False):
          
        batch_time = AverageMeter()
        flow2_EPEs = AverageMeter()
        P1_errors = AverageMeter()
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
            # right_trans = depth2trans(right_depth,beta=beta)
            
            left_depth_l = F.interpolate(left_depth,size=[clear_left.shape[-2],clear_left.shape[-1]],mode='bilinear',
                                         align_corners=False)
            right_depth_l = F.interpolate(right_depth,size=[clear_left.shape[-2],clear_left.shape[-1]],mode='bilinear',
                                         align_corners=False)
            haze_left = recover_haze_images(clean_images=clear_left,beta=beta,A=airlight,depth=left_depth_l)
            haze_right = recover_haze_images(clean_images=clear_right,beta=beta,A=airlight,depth=right_depth_l)

            haze_left = haze_left.float()
            haze_right = haze_right.float()
            target_disp_left = target_disp_left.float()
            target_disp_right = target_disp_right.float()
            focal_length = focal_length.float()
            airlight = airlight.float()
            beta = beta.float()
            left_depth = left_depth.float()
            right_depth = right_depth.float()
            left_trans = left_trans.float()

        
            with torch.no_grad():
                start_time = time.perf_counter()
                # Get the predicted disparity
                if self.model=="DSSMD":
                    disparity_pyrmaid,pred_trans,pred_airlght = self.net(haze_left,haze_right)
                    output = disparity_pyrmaid[-1]
                    # output = self.net(haze_left,haze_right)
                    output = scale_disp(output, (output.size()[0], self.img_height, self.img_width))  

                    dehazed_left = RecoveredCleanFromTrans(transmission_map=pred_trans,airlight=pred_airlght,
                                                                            haze_image=haze_left)
                    
                inference_time += time.perf_counter() - start_time
                img_nums += haze_left.shape[0]
                flow2_EPE = self.epe(output, target_disp_left)
                P1_error = self.p1_error(output, target_disp_left)
                img_loss = img2mse(dehazed_left,clear_left)
                psnr = mse2psnr(img_loss)
                
            if flow2_EPE.data.item() == flow2_EPE.data.item():
                flow2_EPEs.update(flow2_EPE.data.item(), haze_left.size(0))
            if P1_error.data.item() == P1_error.data.item():
                P1_errors.update(P1_error.data.item(), haze_left.size(0))
                psnr_meters.update(psnr.data.item(),haze_left.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % 10 == 0:
                logger.info('Test: [{0}/{1}]\t Time {2}\t EPE {3}\t PSNR {3}\t '
                      .format(i, len(self.test_loader), batch_time.val, flow2_EPEs.val,psnr_meters.val))
            

            if i in [0,nums_samples//2,nums_samples//4*3]:
                pred_disp = output.squeeze(1).detach()
                gt_disp = target_disp_left.squeeze(1).detach()

                img_summary = dict()
                img_summary['disp_error'] = disp_error_img(pred_disp, gt_disp)
                img_summary['left'] = haze_left.detach()
                img_summary['right'] = haze_right.detach()
                img_summary['gt_disp'] = gt_disp.detach()
                img_summary['pred_disp'] = pred_disp.detach()
                img_summary['dehaze_left'] = dehazed_left.detach()
                save_images(summary_writer, 'test' + str(test_count), img_summary, epoch)
                test_count = test_count +1
                
        logger.info(' * PSNR {:.3f}'.format(psnr_meters.avg))
        logger.info(' * DISP EPE {:.3f}'.format(flow2_EPEs.avg))
        logger.info(' * P1_error {:.3f}'.format(P1_errors.avg))
        logger.info(' * avg inference time {:.3f}'.format(inference_time / img_nums))
        return flow2_EPEs.avg


    def get_model(self):
        return self.net.state_dict()