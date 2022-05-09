from model import utils
from model.Transform import *
from model.EFDM_WATNet import *
from model.VGG import *
import numpy as np
from model.function import exact_feature_distribution_matching as efdm



def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


class Net(nn.Module):
    def __init__(self, encoder, decoder, start_iter):
        super(Net, self).__init__()
        vgg = encoder
        # self.enc_0 = nn.Sequential(*list(vgg.children())[:1])
        # enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1
        # transform
        self.transform = Transform(in_planes=512)
        self.transform2 = Transform2(in_planes=512)
        self.decoder = decoder
        if (start_iter > 0):
            self.transform.load_state_dict(torch.load('experiments/transformer_iter_' + str(start_iter) + '.pth'), strict = False)
            self.decoder.load_state_dict(torch.load('experiments/decoder_iter_' + str(start_iter) + '.pth'), strict = False)
        self.mse_loss = nn.MSELoss()
        self.variation_loss = nn.L1Loss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.dx_bias = np.zeros([256, 256])
        self.dy_bias = np.zeros([256, 256])
        for i in range(256):
            self.dx_bias[:, i] = i
            self.dx_bias[i, :] = i

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i+1))
            results.append(func(results[-1]))

        return results[1:]
    def encode_with_intermediate2(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target, norm=False):
        if (norm == False):
            return self.mse_loss(input, target)
        else:
            return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    def calc_style_loss2(self,input, target):
        B, C, W, H = input.size(0), input.size(1), input.size(2), input.size(3)
        value_content, index_content = torch.sort(input.view(B, C, -1))
        value_style, index_style = torch.sort(target.view(B, C, -1))
        inverse_index = index_content.argsort(-1)
        return self.mse_loss(input.view(B, C, -1), value_style.gather(-1, inverse_index))
    def calc_temporal_loss(self,  x1, x2):
        h = x1.shape[2]
        w = x1.shape[3]
        D = h*w
        return self.mse_loss(x1, x2)

    def compute_total_variation_loss_l1(self, inputs):
        h = inputs.shape[2]
        w = inputs.shape[3]
        h1 = inputs[:, :, 0:h-1, :]
        h2 = inputs[:, :, 1:h, :]
        w1 = inputs[:, :, :, 0:w-1]
        w2 = inputs[:, :, :, 1:w]
        return self.variation_loss(h1, h2)+self.variation_loss(w1, w2)

    def forward(self, content, style, content2=None, video=False, alpha=1):
        # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        content_feat = self.encode(content)
        B, C, W ,H = content_feat.size(0),content_feat.size(1),content_feat.size(2),content_feat.size(3)
        stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        style_feats2 = self.encode_with_intermediate2(style)
        t = efdm(content_feat,style_feats2[-1])
        # content_feat = wct(content_feat)
        t = alpha * t + (1-alpha)*content_feat
        #得到了F(CSC)_m
        p = 0.3*t+0.7*stylized
        # p = nn.Upsample(scale_factor=2, mode='nearest')(p)
        # 接着进入decoder
        g_t = self.decoder(p)
        # 变分项损失（平滑）
        loss_v = self.compute_total_variation_loss_l1(g_t)
        g_t_feats = self.encode_with_intermediate(g_t)
        g_t_feats2 = self.encode_with_intermediate2(g_t)
        # 内容损失
        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm=True) + self.calc_content_loss(
            g_t_feats[4], content_feats[4], norm=True)+self.calc_content_loss(g_t_feats2[-1], t)
        # 风格损失
        loss_s = self.mse_loss(utils.gram_matrix(g_t_feats[0]), utils.gram_matrix(style_feats[0]))+self.calc_style_loss2(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            gram_y = utils.gram_matrix(g_t_feats[i])
            gram_s = utils.gram_matrix(style_feats[i])
            loss_s += self.mse_loss(gram_y, gram_s)+self.calc_style_loss2(g_t_feats[i], style_feats2[i])


        """Back LOSSES"""
        Icc = self.decoder(self.transform2(content_feats[3], content_feats[3], content_feats[4], content_feats[4]))
        Iss = self.decoder(self.transform2(style_feats[3], style_feats[3], style_feats[4], style_feats[4]))
        l_back1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_back2 = self.calc_content_loss(Fcc[-2], content_feats[-2]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_back2 += self.calc_content_loss(Fss[i],style_feats[i])
        if video==False:
            return loss_c, loss_s, l_back1, l_back2, loss_v
        else:
            content_feats2 = self.encode_with_intermediate(content2)
            stylized2 = self.transform(content_feats2[3], style_feats[3], content_feats2[4], style_feats[4])
            g_t2 = self.decoder(stylized2)
            g_t2_feats = self.encode_with_intermediate(g_t2)
            # g_t2_feats[0] = nn.Upsample(scale_factor=2, mode='nearest')(g_t2_feats[0])

            temporal_loss = self.calc_temporal_loss(g_t_feats[0], g_t2_feats[0])


            return loss_c, loss_s, l_back1, l_back2, temporal_loss, loss_v



