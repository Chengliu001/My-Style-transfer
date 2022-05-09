import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm
from model.Decoder import *
from model.EFDM_WATNet import *
from model.Net import *
from dataset.dataset import *
from torch.utils import data

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='D:/CL/单模型多风格/（代码未测试） Multi-style Generative Network for Real-time Transfer/PyTorch-Multi-Style-Transfer-master/experiments/dataset/coco/coco',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='D:/CL/MyTransfer/style_image',
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='weight/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=1500000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--style_weight', type=float, default=3.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=0)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--start_iter', type=float, default=1210000)
args = parser.parse_args('')

device = torch.device('cuda')
decoder = Decoder('Decoder')
vgg = VGG('VGG19')

vgg.features.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.features.children())[:44])
network = Net(vgg, decoder, args.start_iter)

network.train()
network.to(device)
# 将内容图和风格图都随机裁剪成256*256的
content_tf = train_transform()
style_tf = train_transform()
# 加载并对内容图和风格图预处理
content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)
# 读取内容图片
content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
# 读取风格图片
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))
# 定义优化器及要优化的参数
optimizer = torch.optim.Adam([
                              {'params': network.decoder.parameters()},
                              {'params': network.transform.parameters()}], lr=args.lr)

if(args.start_iter > 0):
    optimizer.load_state_dict(torch.load('experiments/optimizer_iter_' + str(args.start_iter) + '.pth'))
# 定义一个记录损失的容器
writer = SummaryWriter('runs/loss4')

for i in tqdm(range(args.start_iter, args.max_iter)):
    # 随着迭代过程调整学习率
    adjust_learning_rate(optimizer, iteration_count=i)
    # 去对应的迭代的风格图和内容图
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    # 打印输出图片大小
    print(content_images.shape)
    print(style_images.shape)
    # 调用Net类中的forward 函数，得到了整个网络的算是
    loss_c, loss_s, l_back1, l_back2, temporal_loss,T_loss = network(content_images, style_images, content_images, True)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    # 总损失
    loss = loss_c + loss_s + l_back1 * 50 + l_back2 * 1
    # 迭代一次记录一次总损失
    writer.add_scalar('total loss', loss, global_step=i)
    # 初始化梯度
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 一个batch优化
    optimizer.step()

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = network.transform.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = optimizer.state_dict()
        torch.save(state_dict,
                   '{:s}/optimizer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

writer.close()