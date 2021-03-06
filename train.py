import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from torch.autograd import Variable
from util.functional_zoo.visualize import make_dot

# RUN THE FOLLOWING FOR CLOTHE DATASET
# python train.py --dataroot datasets/consumer2shop_unpaired/merge_blouse_clean/set1
# --name blouse_set1 --resize_or_crop scale_width_and_crop --no_dropout --display_port 1130

# TO LAUNCH VISDOM: python -m visdom.server -port PORT_ID
# for gpu_ids run CUDA_VISIBLE_DEVICES in the terminal

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('# set A training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0
# does not work for now :\
visualize_net = False #True
if visualize_net:
    for i, data in enumerate(dataset):
        if i == 0:
            model.set_input(data)
            output = model.netD_A.forward(Variable(model.input_A))
            make_dot(output)
            break

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()
