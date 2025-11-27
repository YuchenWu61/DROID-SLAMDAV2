
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp

from lietorch import SE3
from .base import RGBDDataset
from .stream import RGBDStream
cur_path = osp.dirname(osp.abspath(__file__))
test_split = osp.join(cur_path, 'tartanv2_test.txt')
test_split = open(test_split).read().split()

def depth_rgba_float32(depth_rgba):
    depth = depth_rgba.view("<f4")
    return np.squeeze(depth, axis=-1)

class TartanAirV2(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(TartanAirV2, self).__init__(name='TartanAirV2', **kwargs)
        # self.get_mean()
    @staticmethod 
    def is_test_scene(scene):
        # print(scene, any(x in scene for x in test_split))
        return any(x in scene for x in test_split)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building TartanAirV2 dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*/*/*'))

        for scene in tqdm(sorted(scenes)):
            images = sorted(glob.glob(osp.join(scene, 'image_lcam_front/*.png')))
            depths = sorted(glob.glob(osp.join(scene, 'depth_lcam_front/*.png')))

            if len(images) != len(depths):
                print("Skipping scene with unequal images and depths:", scene)

        for scene in tqdm(sorted(scenes)):
            images = sorted(glob.glob(osp.join(scene, 'image_lcam_front/*.png')))
            depths = sorted(glob.glob(osp.join(scene, 'depth_lcam_front/*.png')))

            if len(images) != len(depths):
                continue
            
            poses = np.loadtxt(osp.join(scene, 'pose_lcam_front.txt'), delimiter=' ')
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
            poses[:,:3] /= TartanAirV2.DEPTH_SCALE
            intrinsics = [TartanAirV2.calib_read()] * len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics)

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

        return scene_info

    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 320.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        depth = depth_rgba_float32(depth)
        depth = depth / TartanAirV2.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        return depth


class TartanAirV2Stream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(TartanAirV2Stream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/TartanAir'

        scene = osp.join(self.root, self.datapath)
        image_glob = osp.join(scene, 'image_lcam_front/*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(scene, 'pose_lcam_front.txt'), delimiter=' ')
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)


class TartanAirV2TestStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(TartanAirV2TestStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/mono'
        image_glob = osp.join(self.root, self.datapath, '*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(self.root, 'mono_gt', self.datapath + '.txt'), delimiter=' ')
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)
