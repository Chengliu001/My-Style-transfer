#!/user/bin/env python
# coding=utf-8
"""
@author  : cl
@file   : change.py
@ide    : PyCharm
@time   : 2022-4-24 14:10:57
"""
import os
import uuid
from ffmpy import FFmpeg

exe = r'C:\Users\admin\Desktop\image_change_output\ffmpeg-master-latest-win64-lgpl-shared\bin\ffmpeg.exe'
# 垂直翻转
def vflip(image_path: str, output_dir: str):
    ext = _check_format(image_path)
    result = os.path.join(output_dir, '{}.{}'.format(uuid.uuid4(), ext))
    ff = FFmpeg(executable=exe, inputs={image_path: None},
                outputs={result: '-vf vflip -y'})
    print(ff.cmd)
    ff.run()
    return result


# 水平翻转
def hflip(image_path: str, output_dir: str):
    ext = _check_format(image_path)
    result = os.path.join(output_dir, '{}.{}'.format(uuid.uuid4(), ext))
    ff = FFmpeg(executable=exe,inputs={image_path: None},
                outputs={result: '-vf hflip -y'})
    print(ff.cmd)
    ff.run()
    return result


# 旋转
def rotate(image_path: str, output_dir: str, angle: int):
    ext = _check_format(image_path)
    result = os.path.join(output_dir, '{}.{}'.format(uuid.uuid4(), ext))
    ff = FFmpeg(executable=exe,inputs={image_path: None},
                outputs={result: '-vf rotate=PI*{}/180 -y'.format(angle)})
    print(ff.cmd)
    ff.run()
    return result


# 转置
'''
    type:0 逆时针旋转90度，对称翻转
    type:1 顺时针旋转90度
    type:2 逆时针旋转90度
    type:3 顺时针旋转90度，对称翻转
'''


def transpose(image_path: str, output_dir: str, type: int):
    ext = _check_format(image_path)
    result = os.path.join(output_dir, '{}.{}'.format(uuid.uuid4(), ext))
    ff = FFmpeg(executable=exe,inputs={image_path: None},
                outputs={result: '-vf transpose={} -y'.format(type)})
    print(ff.cmd)
    ff.run()
    return result


def _check_format(image_path: str):
    ext = os.path.basename(image_path).strip().split('.')[-1]
    if ext not in ['png', 'jpg']:
        raise Exception('format error')
    return ext

if __name__ == '__main__':
    print(vflip('C:/Users/admin/Desktop/wave.jpg', 'C:/Users/admin/Desktop/image_change_output/'))
    print(hflip('C:/Users/admin/Desktop/wave.jpg', 'C:/Users/admin/Desktop/image_change_output/'))
    print(rotate('C:/Users/admin/Desktop/wave.jpg', 'C:/Users/admin/Desktop/image_change_output/', 203))
    print(transpose('C:/Users/admin/Desktop/wave.jpg', 'C:/Users/admin/Desktop/image_change_output/', 2))
