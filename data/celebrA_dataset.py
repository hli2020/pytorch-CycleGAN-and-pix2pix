import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import is_image_file
from PIL import Image


class CelebrADataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        im_paths = []
        assert os.path.isdir(self.root), '%s is not a valid directory' % self.root
        with open(opt.anno_file, 'r') as F:
            anno_info = []
            for line in F:
                temp = [x for x in line[11:-1].split(' ')]
                anno = [int(x) for x in temp if x !='']
                anno_info.append(anno)

        if opt.phase == 'train':    # 1:162770
            start_id, end_id = 1, opt.train_size
            anno_info = anno_info[:opt.train_size]

        elif opt.phase == 'test':   # 182638:202599
            start_id, end_id = 182638, 202599
            anno_info = anno_info[start_id-1:end_id]

        for i in range(start_id, end_id+1):
            fname = '{:06d}.jpg'.format(i)
            if is_image_file(fname):
                path = os.path.join(self.root, fname)
                im_paths.append(path)

        # self.im_paths = sorted(im_paths)
        self.im_paths = im_paths
        self.anno_info = anno_info
        self.dataset_size = len(self.im_paths)       # self.A_size
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        im_path = self.im_paths[index % self.dataset_size]
        anno = self.anno_info[index % self.dataset_size]
        img = Image.open(im_path).convert('RGB')
        img = self.transform(img)

        return {'im': img, 'im_path': im_path, 'anno': anno}

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'CelebrADataset'
