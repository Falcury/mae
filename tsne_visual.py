'''
Description: feature visualization use tsne
version: 1.0
Author: Jia Li
Email: j.li@liacs.leidenuniv.nl
Date: 2023-02-03 10:20:06
LastEditTime: 2023-05-15 18:43:59
'''
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,60).__str__()
import csv, cv2
import h5py, gc
import numpy as np
from PIL import Image, TarIO
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import colorsys
import tqdm


def compare_features_year(feature_root, model_name,year, max_images=0, max_patches_per_image=0):
    to_compare_data = []
    to_compare_coor = []
    to_compare_label = []
    img_list = []
    images_included = 0
    for root,dirs,files in os.walk(feature_root): 
        for i, name in enumerate(files) :
            if year is None or name[1:3] == year:   # .startswith("T01")

                if images_included == max_images and max_images > 0:
                    break
                images_included += 1

                file_name, file_type = os.path.splitext(name)
                slide_name = name.replace("_%s_features"%(model_name), "")
                img_list.append(slide_name)
                stain_label = file_name.split("_")[1]
                full_path = os.path.join(feature_root, name)
                with h5py.File(full_path, "r") as hdf5_file:

                    if max_patches_per_image == 0:
                        index = slice(0,None)
                    else:
                        index = slice(0,max_patches_per_image)

                    features = hdf5_file["features"][index]
                    coords = hdf5_file["coords"][index]
                    for j,em in enumerate(features):
                        to_compare_data.append(em)
                        to_compare_label.append(stain_label)
                        to_compare_coor.append(coords[j])
    return to_compare_data,to_compare_coor, to_compare_label, img_list


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
    return hls_colors


def get_nRGB_colors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors


def get_image_array(img_root, img_list, max_patches_per_image=0):
    img_arr_list = []
    num = len(img_list)
    rgb_colors = get_nRGB_colors(num)
    for i, img_name in enumerate(img_list):
        slide_path = os.path.join(img_root, img_name)
        with h5py.File(slide_path, "r") as hdf5_file:

            # Get a lookup index for each patch image, so we can read the patches later when they are needed.
            num_patch_images = len(hdf5_file["imgs"])
            if max_patches_per_image > 0:
                num_patch_images = min(max_patches_per_image, num_patch_images)
            img_indices = range(num_patch_images)

            # Store the information we need to assemble the final image after t-SNE clustering.
            for j in img_indices:
                img_description = (slide_path, j, rgb_colors[i])
                img_arr_list.append(img_description)

    return img_arr_list


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    height, width, _ = image.shape
    center_x = int(image_centers_area_size * x) + offset
    center_y = int(image_centers_area_size * (1 - y)) + offset
    tl_x = center_x - int(width / 2)
    tl_y = center_y - int(height / 2)
    br_x = tl_x + width
    br_y = tl_y + height
    return tl_x, tl_y, br_x, br_y


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def tsne_patches(features, img_arr, model_name, data_name, year, suffix):
    plot_size = 20000
    size = 128
    features = np.array(features)
    tsne = TSNE(n_components=2, init='random', verbose=True).fit_transform(features)

    print("Creating output image...")
    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    # print("tx: ", tx)

    hdf5_filename = ""
    hdf5_file: h5py.File = None
    for (img_hdf5_filename, img_index, rgb_color), x, y in tqdm.tqdm(zip(img_arr, tx, ty), total=len(img_arr)):
        # Open HDF5 file (reuse if already opened)
        if img_hdf5_filename != hdf5_filename:
            if hdf5_file is not None:
                hdf5_file.close()
            hdf5_file = h5py.File(img_hdf5_filename, "r")

        # Read image from HDF5 file
        img = hdf5_file["imgs"][img_index]

        # Add colored outline
        image_height, image_width = img.shape[:2]
        img = cv2.rectangle(img, (0, 0), (image_width - 1, image_height - 1), rgb_color, 5)

        new_img = np.array( Image.fromarray(img).resize((size, size)))
        offset = size // 2
        image_centers_are_size = plot_size - 2 * offset
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(new_img, x, y, image_centers_are_size, offset)
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = new_img

    if hdf5_file is not None:
        hdf5_file.close()

    final_img = Image.fromarray(tsne_plot)
    output_filename = '%s_%s_%s_tsne_%s.jpg'%(model_name, data_name, year, suffix)
    final_img.save(output_filename)
    print(f"Saved image to {output_filename}")
    return tsne

configs = [

    {"arch": "mae_vit_base_patch16", "name": "mae_pretrain_vit_base",
     "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/mae_pretrain_vit_base.pth",
     "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_pretrain_vit_base"},

    {"arch": "mae_vit_large_patch16", "name": "mae_pretrain_vit_large",
     "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/mae_pretrain_vit_large.pth",
     "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_pretrain_vit_large"},

    {"arch": "mae_vit_huge_patch14", "name": "mae_pretrain_vit_huge",
     "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/mae_pretrain_vit_huge.pth",
     "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_pretrain_vit_huge"},

    {"arch": "mae_vit_large_patch16", "name": "mae_visualize_vit_large",
     "checkpoint_path": "demo/mae_visualize_vit_large.pth",
     "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_visualize_vit_large"},

    {"arch": "mae_vit_large_patch16", "name": "mae_visualize_vit_large_ganloss",
     "checkpoint_path": "demo/mae_visualize_vit_large_ganloss.pth",
     "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_visualize_vit_large_ganloss"},

    {"arch": "mae_vit_huge_patch14", "name": "mae_scratch_checkpoint-0",
     "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/mae_scratch_checkpoint-0.pth",
     "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_scratch_checkpoint-0"},

    {"arch": "mae_vit_huge_patch14", "name": "mae_scratch_checkpoint-1",
     "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/mae_scratch_checkpoint-1.pth",
     "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_scratch_checkpoint-1"},

    {"arch": "mae_vit_huge_patch14", "name": "mae_scratch_checkpoint-2",
     "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/mae_scratch_checkpoint-2.pth",
     "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_scratch_checkpoint-2"},

    {"arch": "mae_vit_huge_patch14", "name": "mae_scratch_checkpoint-3",
     "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/mae_scratch_checkpoint-3.pth",
     "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_scratch_checkpoint-3"},

    {"arch": "mae_vit_huge_patch14", "name": "mae_scratch_checkpoint-4",
     "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/mae_scratch_checkpoint-4.pth",
     "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_scratch_checkpoint-4"},

    {"arch": "mae_vit_huge_patch14", "name": "checkpoint-298_scratch",
     "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/checkpoint-298_scratch.pth",
     "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/checkpoint-298_scratch"},

    {"arch": "mae_vit_huge_patch14", "name": "checkpoint-299_imagenet_retrained",
     "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/checkpoint-299_imagenet_retrained.pth",
     "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/checkpoint-299_imagenet_retrained"},
]

def main():
    # Expected filename structure:
    # X<year>-XXXX_<stain_name>_<data_name>_<model_name>.h5            (= file containing patch images)
    # X<year>-XXXX_<stain_name>_<data_name>_<model_name>_features.h5   (= file containing features)

    data_name = "AMC"
    model_name = "MAE"  # "moco","simclr", "simclr", "swav", "byol", "dino", "pirl", "barlow_twins", "seer"
    year = "19" # Set to None to disable filter on year

    # combine_feature_file = "/data/h5/combined_features/%s_%s_features_T%s.npy"%(model_name, data_name, year)
    # img_arr_file = "/data/h5/combined_features/%s_%s_img_T%s.npy"%(model_name, data_name, year)
    # # if not os.path.exists(combine_feature_file):
    # feature_root = "/data/h5/feature_extract_results/%s_%s_h5feature/"%(model_name, data_name)

    feature_root = "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_cls/mae_pretrain_vit_huge/"
    img_root = "/run/media/pieter/T7-Pieter/ssl/PATCHES"

    # Set max_images or max_patches_per_image to 0 to disable the limit
    max_images = 0
    max_patches_per_image = 50

    for config in configs:
        print(f"Visualizing for checkpoint: {config['checkpoint_path']}")
        feature_root = f"/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/{config['name']}/"

        print("Loading features...")
        to_compare_data, to_compare_coor, to_compare_label, img_list = \
            compare_features_year(feature_root, model_name, year, max_images, max_patches_per_image=max_patches_per_image)

        img_arr = get_image_array(img_root, img_list, max_patches_per_image=max_patches_per_image)

        print("Running t-SNE...")
        suffix = config['name'] #"mae_pretrain_vit_huge"
        tsne_patches(to_compare_data, img_arr, model_name, data_name, year, suffix)

    return 0


if __name__ == "__main__" :
    main()

