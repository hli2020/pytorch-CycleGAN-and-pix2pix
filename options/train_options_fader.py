from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency iter of showing training images on web')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency iter of showing training loss on console') #100

        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency iter of saving latest model') # 5000
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency epoch of saving checkpoints at the end of epochs')

        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

        self.parser.add_argument('--batchSize', type=int, default=64, help='input batch size')  # 1 for models except faderNet
        self.parser.add_argument('--niter', type=int, default=100, help='# of epoch to remain at initial learning rate') # 100
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of epoch/intervals to linearly decay lr to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--train_size', type=int, default=162770) #162770
        self.parser.add_argument('--anno_file', type=str, default='datasets/celebA/Anno/list_attr_celeba_hyli.txt')
        self.parser.add_argument('--attri_n', type=int, default=40)
        self.parser.add_argument('--factor', type=float, default=0.0001)
        self.parser.add_argument('--united_optim', action='store_true')

        # trick to stabilize training
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.isTrain = True
