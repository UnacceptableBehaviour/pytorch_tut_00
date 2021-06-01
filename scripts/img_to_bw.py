#! /usr/bin/env python

# load image convert to B&W

# AIM:
# steps
# load phone colour image 3264 x 2448 ~3Mb
# Shrink to 64px wide
# make greyscale

# derived from
# https://www.blog.pythonlibrary.org/2017/10/11/convert-a-photo-to-black-and-white-in-python/
#
# this is probably worth a quick experiment too
# https://datacarpentry.org/image-processing/07-thresholding/

from PIL import Image

def black_and_white(input_image_path, output_image_path):

    color_image = Image.open(input_image_path)
    bw = color_image.convert('L')
    bw.save(output_image_path)

# https://www.blog.pythonlibrary.org/2017/10/11/convert-a-photo-to-black-and-white-in-python/
def black_and_white_dithering(input_image_path,
    output_image_path,
    dithering=True):

    color_image = Image.open(input_image_path)
    if dithering:
        bw = color_image.convert('1')
    else:
        bw = color_image.convert('1',dither=Image.NONE)

    bw.save(output_image_path)


def create_thumbnail(input_image_path, output_image_path, size):
    try:
        img = Image.open(input_image_path)
        img.thumbnail(size, Image.ANTIALIAS)
        img.save(output_image_path, "JPEG")
    except IOError:
        print(f"cannot create thumbnail for {output_image_path}")


def bw_thumb(input_image_path, output_image_path, size, blak_gray, dithering=True):
    try:
        img = Image.open(input_image_path)

        img.thumbnail(size, Image.ANTIALIAS)

        if dithering:
            bw = img.convert(blak_gray)
        else:
            bw = img.convert(blak_gray,dither=Image.NONE)

        bw.save(output_image_path, "JPEG")

    except IOError:
        print(f"cannot create thumbnail for {output_image_path}")


if __name__ == '__main__':
    black_and_white('./data/scales/s0.jpg',
        './data/scales/s0_bw.jpg')

    black_and_white_dithering('./data/scales/s0.jpg',
        './data/scales/s0_bw_d.jpg')

    black_and_white_dithering('./data/scales/s0.jpg',
        './data/scales/s0_bw_Nd.jpg', False)

    BOW  = '1'              # Black or White
    GREY = 'L'              # Greay scale 256?
    thumb_size = 64, 64     # goes with width :)
    create_thumbnail('./data/scales/s0.jpg',
        './data/scales/s0_thumb.jpg', thumb_size)

    bw_thumb('./data/scales/s0.jpg',
        './data/scales/s0_bw_d_thumb.jpg', thumb_size, BOW)

    bw_thumb('./data/scales/s0.jpg',
        './data/scales/s0_bw_Nd_thumb.jpg', thumb_size, BOW, False)

    bw_thumb('./data/scales/s0.jpg',
        './data/scales/s0_bw_d_thumb.jpg', thumb_size, GREY)
