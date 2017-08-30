import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np

def _make_anno(value):

    anno = torch.zeros(1, 40)
    for i in value:
        anno[0, i-1] = 1
    anno = torch.cat((anno, (1 - anno)), dim=1).cuda()
    return anno.resize_([1, 80, 1, 1])

# hyli: this file is of VITAL importance
class FaderGANModel(BaseModel):
    def name(self):
        return 'FaderGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.gpu_mode = (len(opt.gpu_ids) > 1) | (opt.gpu_ids != -1)

        # load/define networks
        self.fader_encoder = networks.define_fader_encoder(ngf=opt.ngf,
                                                           which_structure=opt.which_structure,
                                                           gpu_ids=opt.gpu_ids)
        self.fader_decoder = networks.define_fader_decoder(opt.ngf, opt.which_structure, gpu_ids=opt.gpu_ids)

        if self.isTrain:
            self.united_optim = opt.united_optim
            self.factor = opt.factor
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.which_structure,
                                          use_sigmoid=use_sigmoid, gpu_ids=self.gpu_ids, attri_n=opt.attri_n)
        else:
            # test mode
            self.hyli_test_dict = {
                'all': list(range(1, 41)),
                'young': [40],
                'attractive': [3],
                'big lips': [7],
                'blond hair': [10],
                'eyeglasses': [16],
                'male': [21],
                'mustache': [23],
                'young, mustache': [40, 23],
                'young, attractive': [40, 3],
                'young, attractive, blond hair': [40, 3, 10]
            }

        # resume or test
        # TODO: for resume lr, epoch_iter index are NOT udpated (still from zero-index)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.fader_decoder, 'fader_decoder', which_epoch)
            self.load_network(self.fader_encoder, 'fader_encoder', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'netD', which_epoch)

        # optimizer, loss
        if self.isTrain:
            self.old_lr = opt.lr
            # self.fake_A_pool = ImagePool(opt.pool_size)
            # self.fake_B_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterion_mse = torch.nn.MSELoss()
            self.criterion_bce = torch.nn.BCEWithLogitsLoss()

            # initialize optimizers
            self.optimizer_encoder = torch.optim.Adam(self.fader_encoder.parameters(),
                                                      lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_decoder = torch.optim.Adam(self.fader_decoder.parameters(),
                                                      lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

        if self.gpu_mode:
            if len(self.gpu_ids) > 1:
                # TODO: for now only apply single GPU mode for the decoder (got error "Tensors on multiple GPUs")
                self.fader_decoder.cuda()
                # self.fader_decoder = torch.nn.DataParallel(self.fader_decoder, device_ids=self.gpu_ids).cuda()
                self.fader_encoder = torch.nn.DataParallel(self.fader_encoder, device_ids=self.gpu_ids).cuda()
            else:
                self.fader_decoder.cuda()
                self.fader_encoder.cuda()

            if self.isTrain:
                self.criterion_bce.cuda()
                self.criterion_mse.cuda()
                if len(self.gpu_ids) > 1:
                    self.netD = torch.nn.DataParallel(self.netD, device_ids=self.gpu_ids).cuda()
                else:
                    self.netD.cuda()

        print('---------- Networks initialized -------------')
        networks.print_network(self.fader_encoder)
        networks.print_network(self.fader_decoder)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):

        self.image = input['im']    # 32 x 3 x 256 x 256
        self.image.resize_(self.image.size()).copy_(self.image)
        self.image = self.image.cuda()

        # to (32 x 80 x 1 x 1) TensorFloat
        temp_ = torch.t(torch.stack(input['anno']))
        mask = torch.eq(temp_, -1)
        temp_[mask] = 0
        self.target = Variable(temp_.float().cuda())    # shape [1 x 40]
        self.anno = torch.cat((temp_, (1 - temp_)), dim=1).float().cuda()
        actual_batch_size = self.image.shape[0]
        self.anno.resize_([actual_batch_size, 2*self.opt.attri_n, 1, 1])
        self.image_paths = input['im_path']


    def test(self):

        encoder_out = self.fader_encoder.forward(Variable(self.image, volatile=True))
        decoder_out = self.fader_decoder.forward(encoder_out, Variable(self.anno, volatile=True))
        self.decoder_out = decoder_out

        self.test_result = dict(self.hyli_test_dict)

        for key, value in self.hyli_test_dict.items():
            _anno = _make_anno(value)
            _out = self.fader_decoder.forward(encoder_out, Variable(_anno, volatile=True))
            self.test_result.update({key: _out})

    def optimize_parameters(self):

        self.optimizer_encoder.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_D.zero_grad()

        # forward the encoder-decoder first, then compute the loss
        encoder_out = self.fader_encoder(Variable(self.image))

        if self.united_optim:
            discri_out = self.netD(encoder_out)
            loss_bce_disc = self.criterion_bce(discri_out, self.target)

            decoder_out = self.fader_decoder(encoder_out, Variable(self.anno))
            loss_mse = self.criterion_mse(decoder_out, Variable(self.image))
            loss_bce_decoder = self.criterion_bce(discri_out, (1-self.target))

            self.loss_all = loss_mse + self.factor * loss_bce_decoder + loss_bce_disc
            self.loss_all.backward()

            self.optimizer_D.step()
            self.optimizer_encoder.step()
            self.optimizer_decoder.step()
        else:
            _input = encoder_out.detach()
            discri_out = self.netD(_input)
            loss_bce_disc = self.criterion_bce(discri_out, self.target)
            loss_bce_disc.backward()
            self.optimizer_D.step()

            decoder_out = self.fader_decoder(encoder_out, Variable(self.anno))
            loss_mse = self.criterion_mse(decoder_out, Variable(self.image))
            discri_out = self.netD(encoder_out)
            loss_bce_decoder = self.criterion_bce(discri_out, (1-self.target))

            _temp = loss_mse + loss_bce_decoder
            _temp.backward()
            self.optimizer_encoder.step()
            self.optimizer_decoder.step()

            # just for log purpose
            self.loss_all = loss_mse + self.factor * loss_bce_decoder + loss_bce_disc

        self.loss_all_value = self.loss_all.data[0]
        self.loss_mse_value = loss_mse.data[0]
        self.loss_bce_decoder_value = loss_bce_decoder.data[0]
        self.loss_bce_disc_value = loss_bce_disc.data[0]

    def get_current_errors(self):
        loss_all = self.loss_all_value
        loss_bce_decoder = self.loss_bce_decoder_value
        loss_bce_disc = self.loss_bce_disc_value
        loss_mse = self.loss_mse_value

        return OrderedDict([('loss_all', loss_all), ('loss_bce_decoder', loss_bce_decoder),
                            ('loss_bce_disc', loss_bce_disc), ('loss_mse', loss_mse)])

    def get_current_visuals(self):
        # for test only
        image = util.tensor2im(self.image)
        decoder_out = util.tensor2im(self.decoder_out.data)
        out_dict_list = [('image', image), ('decoder_out', decoder_out)]

        for key, value in self.test_result.items():
            _out_value = util.tensor2im(value.data)
            out_dict_list.append((key, _out_value))
        return OrderedDict(out_dict_list)

    def save(self, label):
        self.save_network(self.netD, 'netD', label, self.gpu_ids)
        self.save_network(self.fader_encoder, 'fader_encoder', label, self.gpu_ids)
        self.save_network(self.fader_decoder, 'fader_decoder', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_decoder.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_encoder.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
