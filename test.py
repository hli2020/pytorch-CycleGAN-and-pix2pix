import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html

# --dataroot datasets/celebA/Img/img_align_celeba --name fader_debug
# --model fader_gan --dataset_mode celebrA --resize_or_crop scale_width_and_crop --gpu_id 1

# python test.py --dataroot datasets/consumer2shop_fuck/merge_blouse --name blouse_cycle_gan --model cycle_gan --resize_or_crop scale_width_and_crop --no_dropout

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    print('process image... %s' % model.image_paths)
    visualizer.save_images(webpage, visuals, model.image_paths)

webpage.save()
