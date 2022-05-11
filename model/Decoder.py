import torch.nn as nn

cfg = {
    'Decoder': [256, 'U', 256, 256, 256, 128, 'U', 128, 64, 'U', 64, 3]
}

class Decoder(nn.Module):
    def __init__(self, vgg_name):
        super(Decoder, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])

    def forward(self, x):
        out = self.features(x)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 512
        for x in cfg:
            # 碰到U上采样一次，碰到3padding一次，卷积一次成3通道，其余的就是填充，按照x通道去卷积，激活
            if x == 'U':
                layers += [nn.Upsample(scale_factor=2, mode='nearest')]
            elif x==3:
                layers += [nn.ReflectionPad2d((1, 1, 1, 1)),
                           nn.Conv2d(in_channels, x, kernel_size=3)]
            else:
                layers += [nn.ReflectionPad2d((1, 1, 1, 1)),
                           nn.Conv2d(in_channels, x, kernel_size=3),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

if __name__  ==  "__main__":
    decoder = Decoder('Decoder')
    print(decoder)