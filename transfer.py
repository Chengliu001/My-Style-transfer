import argparse
from model.VGG import *
from model.Decoder import *
from model.Transform import *
import os
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from model.function import exact_feature_distribution_matching as efdm
import time

def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def test_transform2(size,crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize([size, size]))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform





time_start = time.time()
parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str, default='input/venice-boat.jpg',
                    help='File path to the content image')
parser.add_argument('--style', type=str, default='style/seated-nude.jpg',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--steps', type=str, default=1)
parser.add_argument('--vgg', type=str, default='weight/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='experiments/decoder_iter_1210000.pth')
parser.add_argument('--transform', type=str, default='experiments/transformer_iter_1210000.pth')

# Additional options
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--content_size', type=int, default=512,
                    help='new(minimum)size')
parser.add_argument('--style_size', type=int, default=512,
                    help='new(minimum)size')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')



# Advanced options

args = parser.parse_args('')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = Decoder('Decoder')
transform = Transform(in_planes=512)
vgg = VGG('VGG19')

decoder.eval()
transform.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
transform.load_state_dict(torch.load(args.transform))
vgg.features.load_state_dict(torch.load(args.vgg))

# norm = nn.Sequential(*list(vgg.features())[:1])
enc_1 = nn.Sequential(*list(vgg.features.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.features.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.features.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.features.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.features.children())[31:44])  # relu4_1 -> relu5_1


# norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
transform.to(device)
decoder.to(device)

content_tf = test_transform()
style_tf = test_transform()


content = content_tf(Image.open(args.content))
style = style_tf(Image.open(args.style))

style = style.to(device).unsqueeze(0)
content = content.to(device).unsqueeze(0)


with torch.no_grad():
    for x in range(args.steps):
        print('iteration ' + str(x))

        Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
        Content5_1 = enc_5(Content4_1)

        Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
        Style5_1 = enc_5(Style4_1)


        stylized = transform(Content4_1, Style4_1, Content5_1, Style5_1)

        t = efdm(Content4_1, Style4_1)
        t = 1* t + (1 - 1) * Content4_1
        # 得到了F(CSC)_m
        p =0.3*t+0.7*stylized

        content = decoder(p)

        content.clamp(0, 255)

    content = content.cpu()

    output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
        args.output, splitext(basename(args.content))[0],
        splitext(basename(args.style))[0], args.save_ext
    )
    save_image(content, output_name)
end_time = time.time()
print("time_cost：",end_time-time_start,"s")