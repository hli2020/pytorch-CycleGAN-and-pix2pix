import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np

# hyli: this file is of VITAL importance
class FaderGANModel(BaseModel):
    def name(self):
        return 'FaderGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.factor = opt.factor
        # load/define networks
        self.fader_encoder = networks.define_fader_encoder(ngf=opt.ngf,
                                                           which_structure=opt.which_structure,
                                                           gpu_ids=opt.gpu_ids)
        self.fader_decoder = networks.define_fader_decoder(opt.ngf, opt.which_structure, gpu_ids=opt.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.which_structure,
                                          use_sigmoid=use_sigmoid, gpu_ids=self.gpu_ids, attri_n=opt.attri_n)

        # resume, TODO
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch   # no such parameter
            self.load_network(self.net, '', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, '', which_epoch)

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

        if 1:
            # always use gpu mode
            self.criterion_bce.cuda()
            self.criterion_mse.cuda()
            self.fader_decoder.cuda()
            self.fader_encoder.cuda()
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
        self.target = Variable(temp_.float().cuda())
        self.anno = torch.cat((temp_, (1 - temp_)), dim=1).float().cuda()
        actual_batch_size = self.image.shape[0]
        self.anno.resize_([actual_batch_size, 2*self.opt.attri_n, 1, 1])
        self.image_paths = input['im_path']

    def test(self):
        # TODO
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        self.rec_B = self.netG_A.forward(self.fake_A)

    def compute_loss(self):

        # forward the encoder-decoder first, then compute the loss
        encoder_out = self.fader_encoder(Variable(self.image))
        decoder_out = self.fader_decoder(encoder_out, Variable(self.anno))
        discri_out = self.netD(encoder_out)

        loss_mse = self.criterion_mse(decoder_out, Variable(self.image))
        loss_bce_decoder = self.criterion_bce(discri_out, (1-self.target))
        loss_bce_disc = self.criterion_bce(discri_out, self.target)

        # factor = 0.0001
        self.loss_all = loss_mse + self.factor * loss_bce_decoder + loss_bce_disc

        self.loss_all_value = self.loss_all.data[0]
        self.loss_mse_value = loss_mse.data[0]
        self.loss_bce_decoder_value = loss_bce_decoder.data[0]
        self.loss_bce_disc_value = loss_bce_disc.data[0]


    def optimize_parameters(self):

        self.optimizer_encoder.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_D.zero_grad()

        self.compute_loss()
        self.loss_all.backward()

        self.optimizer_encoder.step()
        self.optimizer_decoder.step()
        self.optimizer_D.step()


    def get_current_errors(self):
        loss_all = self.loss_all_value
        loss_bce_decoder = self.loss_bce_decoder_value
        loss_bce_disc = self.loss_bce_disc_value
        loss_mse = self.loss_mse_value

        return OrderedDict([('loss_all', loss_all), ('loss_bce_decoder', loss_bce_decoder),
                            ('loss_bce_disc', loss_bce_disc), ('loss_mse', loss_mse)])


    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B = util.tensor2im(self.rec_B.data)
        if self.opt.identity > 0.0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

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
