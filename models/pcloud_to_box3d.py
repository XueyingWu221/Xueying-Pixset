from utils import to_box3d_package
import layers

import numpy as np
import torch
import torch.nn as nn


class PCloudToBox3D(nn.Module):
    def __init__(self, cfg, in_channels=7):
        super(PCloudToBox3D, self).__init__()
        self.cfg = cfg
        self.nb_classes = len(self.cfg['PREPROCESSING']['BOX_3D']['CLASSIFICATION'])


        self.feature_extract = layers.DenseBlock2(in_channels=in_channels, out_channels=32, **self.cfg['NEURAL_NET']['ENCODER'])
        self.backbone = layers.DenseBlock2(in_channels=self.feature_extract.out_channels, out_channels=7, **self.cfg['NEURAL_NET']['BACKBONE'])
        self.detection_head = layers.DenseBlock2(in_channels=self.backbone.out_channels, out_channels=8+self.nb_classes, **self.cfg['NEURAL_NET']['DECODER'])

        # self.feature_extract = layers.DenseBlock(in_channels=in_channels, **self.cfg['NEURAL_NET']['ENCODER'])
        # self.backbone = layers.DenseBlock(in_channels=self.feature_extract.out_channels,
        #                                   **self.cfg['NEURAL_NET']['BACKBONE'])
        # self.detection_head = layers.DenseBlock(in_channels=self.backbone.out_channels,
        #                                         out_channels=8 + self.nb_classes, **self.cfg['NEURAL_NET']['DECODER'])

    def forward(self, x):

        pillars, indices = x

        features = self.feature_extract(pillars)
        features = torch.max(features, dim=-1)[0]

        grid = torch.zeros((
            features.shape[0],
            features.shape[1],
            self.cfg['PREPROCESSING']['POINT_CLOUD']['PILLARS']['GRID']['x'][2],
            self.cfg['PREPROCESSING']['POINT_CLOUD']['PILLARS']['GRID']['y'][2],
        ), device = features.device)

        for i in range(grid.shape[0]):
            grid[i,:,indices[i,:,0],indices[i,:,1]] = features[i]

        grid = self.backbone(grid)

        grid = self.detection_head(grid)

        return grid

    def post_process(self, raw_output):
        return to_box3d_package(raw_output, self.cfg)

    @property
    def size_of_net(self):
        out = 0
        for key in list(self.state_dict()):
            out += np.product(self.state_dict()[key].shape)
        return out