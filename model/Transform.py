from model.EFDM_WATNet import *

class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        # 得到了F(CSC)4_1和得到了F(CSC)5_1
        self.sanet4_1 = EFDM_WATNet(in_planes = in_planes)
        self.sanet5_1 = EFDM_WATNet(in_planes = in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
        # self.transpose = nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=2, stride=1,padding=1)
    def forward(self, content4_1, style4_1, content5_1, style5_1):
        A = self.sanet4_1(content4_1, style4_1)
        B1, C1, W1, H1 = A.size(0), A.size(1), A.size(2), A.size(3)
        B = self.upsample5_1(self.sanet5_1(content5_1, style5_1))
        B2, C2, W2, H2 = B.size(0), B.size(1), B.size(2), B.size(3)
        C = B
        if (H1 != H2):
            C = nn.ReflectionPad2d((1, 0, 0, 0))(C)
        if (W1 != W2):
            C = nn.ReflectionPad2d((0, 0, 0, 1))(C)
        B3, C3, W3, H3 = C.size(0), C.size(1), C.size(2), C.size(3)
        M = self.merge_conv(self.merge_conv_pad(A + C))
        return M

class Transform2(nn.Module):
    def __init__(self, in_planes):
        super(Transform2, self).__init__()
        # 得到了F(CSC)4_1和得到了F(CSC)5_1
        self.sanet4_1 = EFDM_WATNet(in_planes = in_planes)
        self.sanet5_1 = EFDM_WATNet(in_planes = in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
    def forward(self, content4_1, style4_1, content5_1, style5_1):
        M = self.merge_conv(self.merge_conv_pad(self.sanet4_1(content4_1, style4_1) + self.upsample5_1(self.sanet5_1(content5_1, style5_1))))
        return M