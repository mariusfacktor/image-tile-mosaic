
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from PIL import Image, ImageDraw
from skimage import exposure
import random
import os, sys
import numpy as np
import time
from tqdm import tqdm


from skimage.util.shape import view_as_blocks




######################## FILL IN THESE FIELDS ########################
BLOCK_SIZE = 85
SCALE_TARGET_IMAGE = 1.0
SCALE_SOURCE_IMAGES = 1.0
######################## FILL IN THESE FIELDS ########################



############################### USAGE ################################
# python3 image_tile_mosaic.py target.jpg source1.jpg source2.jpg source3.jpg
# jpg or png are fine
############################### USAGE ################################



def visualize(**kwargs):

    num_images = len(kwargs)

    for i, (key, value) in enumerate(kwargs.items()):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(value)
        plt.axis('off')
        plt.title(key)

    plt.show()



def stitch_blocks_into_image(blocks, num_h, num_w):
    # Stitch the reconstructed image together
    stitch_rows = [None]*num_h

    # Stitch rows first
    for t_i in range(num_h):
        for t_j in range(num_w):
            if t_j == 0:
                stitch_rows[t_i] = blocks[t_i][0]
            else:
                stitch_rows[t_i] = np.hstack((stitch_rows[t_i], blocks[t_i][t_j]))


    reconstruction = stitch_rows[0]
    # Stitch rows together
    for t_i in range(num_h):
        if t_i != 0:
            reconstruction = np.vstack((reconstruction, stitch_rows[t_i]))

    return reconstruction



def convert_1d_index_to_2d(i, width):

    row = int(i / width)
    col = i % width

    return row, col


def convert_2d_index_to_1d(row, col, width):

    i = (width * row) + col

    return i


def convert_1d_index_to_3d(i, source_num_h, source_num_w):

    accum = 0
    frame = 0
    row = 0
    col = 0

    while accum < i:

        num_blocks_frame = source_num_h[frame] * source_num_w[frame]

        if num_blocks_frame + accum <= i:
            accum += num_blocks_frame
            frame += 1

        else:
            num_still_needed = i - accum

            row = int(num_still_needed / source_num_w[frame])
            col = num_still_needed % source_num_w[frame]

            accum += row * source_num_w[frame]
            accum += col

    return frame, row, col


def convert_3d_index_to_1d(frame, row, col, source_num_h, source_num_w):

    i = 0

    for frame_i in range(frame):
        i += source_num_h[frame_i] * source_num_w[frame_i]

    i += (row * source_num_w[frame]) + col

    return i



# bl_i vertical index, bl_i horizontal index
def draw_box(image, bl_i, bl_j, bl_h, bl_w, color=255, text=None):
    image = np.copy(image)

    # top line
    image[bl_i * bl_h, bl_j * bl_w : (bl_j+1) * bl_w] = color
    image[bl_i * bl_h + 1, bl_j * bl_w : (bl_j+1) * bl_w] = color
    image[bl_i * bl_h + 2, bl_j * bl_w : (bl_j+1) * bl_w] = color
    
    # bottom line
    image[(bl_i+1) * bl_h - 1, bl_j * bl_w : (bl_j+1) * bl_w] = color
    image[(bl_i+1) * bl_h - 2, bl_j * bl_w : (bl_j+1) * bl_w] = color
    image[(bl_i+1) * bl_h - 3, bl_j * bl_w : (bl_j+1) * bl_w] = color

    # left line
    image[bl_i * bl_h : (bl_i+1) * bl_h, bl_j * bl_w] = color
    image[bl_i * bl_h : (bl_i+1) * bl_h, bl_j * bl_w + 1] = color
    image[bl_i * bl_h : (bl_i+1) * bl_h, bl_j * bl_w + 2] = color
    
    # right line
    image[bl_i * bl_h : (bl_i+1) * bl_h, (bl_j+1) * bl_w - 1] = color
    image[bl_i * bl_h : (bl_i+1) * bl_h, (bl_j+1) * bl_w - 2] = color
    image[bl_i * bl_h : (bl_i+1) * bl_h, (bl_j+1) * bl_w - 3] = color

    if text is not None:
        # Add white background
        image[bl_i * bl_h + 3, bl_j * bl_w : bl_j* bl_w + 45] = 255
        image[bl_i * bl_h + 4, bl_j * bl_w : bl_j* bl_w + 45] = 255
        image[bl_i * bl_h + 5, bl_j * bl_w : bl_j* bl_w + 45] = 255
        image[bl_i * bl_h + 6, bl_j * bl_w : bl_j* bl_w + 45] = 255
        image[bl_i * bl_h + 7, bl_j * bl_w : bl_j* bl_w + 45] = 255
        image[bl_i * bl_h + 8, bl_j * bl_w : bl_j* bl_w + 45] = 255
        image[bl_i * bl_h + 9, bl_j * bl_w : bl_j* bl_w + 45] = 255
        image[bl_i * bl_h + 10, bl_j * bl_w : bl_j* bl_w + 45] = 255

        # Add text
        image = Image.fromarray(image)
        text_image = ImageDraw.Draw(image)
        text_image.text((bl_j*bl_w + 1, bl_i*bl_h), text, (0,0,0))
        image = np.asarray(image)

    return image

def mse(img_A, img_B):

    difference_array = np.subtract(img_A, img_B)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()

    return mse


def block_function(arr, block_h, block_w, func):
    # Reshape the array into blocks
    blocks = view_as_blocks(arr, block_shape=(block_h, block_w))

    # Apply the function to each block
    result = np.apply_over_axes(func, blocks, axes=(2,3))

    return result.squeeze()



def mse_whole_image(img_A, img_B, block_h, block_w, num_block_h, num_block_w):


    difference_array = np.subtract(img_A, img_B)
    squared_array = np.square(difference_array)

    mse_array = block_function(squared_array, block_h, block_w, np.mean)


    return mse_array




# divide all by max value
def normalize(image):
    max_val = np.max(image)
    image = np.log((image / max_val)+1) * 256

    return image

def contrast_stretch(image):
    min_val = np.min(image)
    max_val = np.max(image)
    image = np.asarray([((pix - min_val) / (max_val - min_val)) * 255 for pix in image])
    image = image.astype(np.uint8)

    return image



def main():

    # Get start time
    start_time = time.time()

    blck_size = BLOCK_SIZE
    return_num = 150

    random.seed(0)

    num_source = len(sys.argv) - 2 # number of source images

    target_path = './' + sys.argv[1]

    source_path = [None]*num_source
    for s_n in range(num_source):
        source_path[s_n] = './' + sys.argv[2 + s_n]

    target_image = plt.imread(target_path)

    source_image = [None]*num_source
    original_source_image = [None]*num_source
    for s_n in range(num_source):
        source_image[s_n] = plt.imread(source_path[s_n])


    # Contrast stretch
    target_image = contrast_stretch(target_image)
    for s_n in range(num_source):
        source_image[s_n] = contrast_stretch(source_image[s_n])


    # Resize target image
    target_image = Image.fromarray(target_image)

    height = target_image.height
    width = target_image.width

    if SCALE_TARGET_IMAGE is not None:
        height = int(SCALE_TARGET_IMAGE * height)
        width = int(SCALE_TARGET_IMAGE * width)
        target_image = target_image.resize((width, height)) # w, h

    target_image = np.asarray(target_image)



    # Resize source image
    for s_n in range(num_source):
        source_image[s_n] = Image.fromarray(source_image[s_n])

        height = source_image[s_n].height
        width = source_image[s_n].width

        if SCALE_SOURCE_IMAGES is not None:
            height = int(SCALE_SOURCE_IMAGES * height)
            width = int(SCALE_SOURCE_IMAGES * width)
            source_image[s_n] = source_image[s_n].resize((width, height)) # w, h


        source_image[s_n] = np.asarray(source_image[s_n])



    # Keep copy of unchanged source image
    for s_n in range(num_source):
        original_source_image[s_n] = source_image[s_n]



    # make black and white
    target_image = target_image[:,:,0]
    for s_n in range(num_source):
        source_image[s_n] = source_image[s_n][:,:,0]



    # Test histogram equalization
    target_image = exposure.equalize_adapthist(target_image)
    #target_image = exposure.equalize_hist(target_image)
    target_image = contrast_stretch(target_image)

    '''
    for s_n in range(num_source):
        source_image[s_n] = exposure.equalize_hist(source_image[s_n])
        source_image[s_n] = contrast_stretch(source_image[s_n]) # Contrast stretch again
    '''


    # Get dimensions
    targ_h = target_image.shape[0] # 1080
    targ_w = target_image.shape[1] # 1920

    print('targ_h %d targ_w %d' %(targ_h, targ_w))

    source_h = [None]*num_source
    source_w = [None]*num_source


    for s_n in range(num_source):
        source_h[s_n] = source_image[s_n].shape[0]
        source_w[s_n] = source_image[s_n].shape[1]
        print('source_h %d source_w %d s_n %d' %(source_h[s_n], source_w[s_n], s_n))


    # Block size 
    blck_h = blck_size #
    blck_w = blck_size #

    print('block size %d' %blck_size)

    # Number of blocks
    targ_num_h = (int)(targ_h / blck_h) # 36
    targ_num_w = (int)(targ_w / blck_w) # 64

    source_num_h = [None]*num_source
    source_num_w = [None]*num_source

    for s_n in range(num_source):
        source_num_h[s_n] = (int)(source_h[s_n] / blck_h)
        source_num_w[s_n] = (int)(source_w[s_n] / blck_w)


    # Get target blocks
    targ_blocks = [[None]*targ_num_w for i in range(targ_num_h)]
    targ_dct_blocks = [[None]*targ_num_w for i in range(targ_num_h)]

    for i in range(targ_num_h):
        for j in range(targ_num_w):
            targ_blocks[i][j] = np.asarray(target_image[i*blck_h: i*blck_h + blck_h , j*blck_w: j*blck_w + blck_w])

            targ_dct_blocks[i][j] = dct(targ_blocks[i][j])


    # Get source blocks
    source_blocks = [None]*num_source
    source_dct_blocks = [None]*num_source 
    original_source_blocks = [None]*num_source # Keep to stitch together at the end


    for s_n in range(num_source):
        source_blocks[s_n] = [[None]*source_num_w[s_n] for i in range(source_num_h[s_n])]
        source_dct_blocks[s_n] = [[[None]*source_num_w[s_n] for i in range(source_num_h[s_n])] for j in range(4)] # 4 rotations

        original_source_blocks[s_n] = [[None]*source_num_w[s_n] for i in range(source_num_h[s_n])]




    for s_n in range(num_source):
        for i in range(source_num_h[s_n]):
            for j in range(source_num_w[s_n]):
                source_blocks[s_n][i][j] = np.asarray(source_image[s_n][i*blck_h: i*blck_h + blck_h , \
                    j*blck_w: j*blck_w + blck_w])
                original_source_blocks[s_n][i][j] = np.asarray(original_source_image[s_n][i*blck_h: i*blck_h + blck_h , \
                    j*blck_w: j*blck_w + blck_w])

                im = Image.fromarray(source_blocks[s_n][i][j])
                for rot in range(4):
                    source_dct_blocks[s_n][rot][i][j] = dct(im.rotate(90*rot))



    stitch_source = [[None]*4 for i in range(num_source)]
    for s_n in range(num_source):
        for rot in range(4):
            stitch_source[s_n][rot] = stitch_blocks_into_image(source_dct_blocks[s_n][rot], source_num_h[s_n], source_num_w[s_n])



    target_idx_array_list = [[[None] for i in  range(targ_num_w)] for j in range(targ_num_h)]
    target_rot_array_list = [[[None] for i in  range(targ_num_w)] for j in range(targ_num_h)]


    for t_i in tqdm(range(targ_num_h)):
        for t_j in range(targ_num_w):

            dct_block = Image.fromarray(targ_dct_blocks[t_i][t_j])

            mse_source_list = []
            rot_source_list = []

            for s_n in range(num_source):

                tile_target_block = np.tile(dct_block, (source_num_h[s_n], source_num_w[s_n]))

                mse_array_rot = [None]*4


                for rot in range(4):


                    mse_array = mse_whole_image(tile_target_block, stitch_source[s_n][rot], 
                                                blck_h, blck_w, source_num_h[s_n], source_num_w[s_n])
                    mse_array_rot[rot] = mse_array


                min_mse_array = np.minimum.reduce(mse_array_rot)
                rot_array = np.argmin(mse_array_rot, axis=0)


                flattened_mse_list = [item for sublist in min_mse_array for item in sublist]
                flattened_rot_list = [item for sublist in rot_array for item in sublist]


                mse_source_list.extend(flattened_mse_list)
                rot_source_list.extend(flattened_rot_list)


            sorted_mse_indices = sorted(range(len(mse_source_list)), key=lambda i: mse_source_list[i])

            target_idx_array_list[t_i][t_j] = sorted_mse_indices
            target_rot_array_list[t_i][t_j] = rot_source_list




    used = []

    # Choose source blocks closest to target blocks
    chosen_blocks = [[None]*targ_num_w for i in range(targ_num_h)]
    # Gives index of source block at each point in target image
    index_blocks = [[(None, None, None, None)]*targ_num_w for i in range(targ_num_h)] # s_n, s_i, s_j, rot

    targ_i_list = list(range(targ_num_h))
    random.shuffle(targ_i_list)

    targ_j_list = list(range(targ_num_w))
    random.shuffle(targ_j_list)

    for t_i in targ_i_list:
        for t_j in targ_j_list:
            for sorted_idx in target_idx_array_list[t_i][t_j]:
                if sorted_idx not in used:
                    
                    used.append(sorted_idx)
                    rot = target_rot_array_list[t_i][t_j][sorted_idx]

                    frame, row, col = convert_1d_index_to_3d(sorted_idx, source_num_h, source_num_w)
                    
                    im = Image.fromarray(original_source_blocks[frame][row][col])
                    chosen_blocks[t_i][t_j] = im.rotate(90 * rot) 
                    index_blocks[t_i][t_j] = (frame, row, col, rot)

                    break # chosen



    tot_num_source_blocks = 0
    for s_n in range(num_source):
        tot_num_source_blocks += source_num_h[s_n]*source_num_w[s_n] 

    percent_used = float(len(used) / tot_num_source_blocks)
    print('Percent Used: %.2f' %percent_used)

    reconstruction = stitch_blocks_into_image(chosen_blocks, targ_num_h, targ_num_w)

    duration = (int)(time.time() - start_time)
    print('Duration (s) %d' %duration)

    save_out_reconstruction = Image.fromarray(reconstruction)
    save_out_reconstruction.save('reconstruction.png')
    #visualize(reconstruction)

    print('Saving out annotated source images...')
    
    # Show grid on source image
    for s_n in range(num_source):
        boxes = original_source_image[s_n]

        # Grid all blocks in black
        for s_i in range(source_num_h[s_n]):
            for s_j in range(source_num_w[s_n]):
                boxes = draw_box(boxes, s_i, s_j, blck_h, blck_w, \
                    color=0)


        # Grid used blocks in white
        for t_i in range(targ_num_h):
            for t_j in range(targ_num_w):

                # Text: targ_i, targ_j, rotation
                if index_blocks[t_i][t_j][0] == s_n:

                    rotation_str = '' 
                    if index_blocks[t_i][t_j][3] == 0:
                        rotation_str = 'U'
                    if index_blocks[t_i][t_j][3] == 1:
                        rotation_str = 'L'
                    if index_blocks[t_i][t_j][3] == 2:
                        rotation_str = 'D'
                    if index_blocks[t_i][t_j][3] == 3:
                        rotation_str = 'R'
                    boxes = draw_box(boxes, index_blocks[t_i][t_j][1], index_blocks[t_i][t_j][2], blck_h, blck_w, \
                        text=str(t_i) + ' ' + str(t_j) + ' ' + rotation_str)

        save_out_img = Image.fromarray(boxes)
        save_out_img.save('gridded_source_image' + str(s_n) + '.png')
        
        # boxes.save('gridded_source_image' + str(s_n) + '.png')
        # visualize(boxes)




if __name__ == '__main__':
    main()

    


