'''
Description: 
version: 1.0
Author: Jia Li
Email: j.li@liacs.leidenuniv.nl
Date: 2023-05-18 10:29:52
LastEditTime: 2023-05-22 15:40:05
'''

import models_mae
import numpy as np
import torch
import os
import h5py
import tqdm
from PIL import Image
import argparse
import cv2

def get_h5_list(dir_name):
    h5_files: list[str] = []
    for root, _, files in os.walk(dir_name):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.endswith("h5") and not file[0] == ".":
                h5_files.append(file)
    h5_files.sort()
    return h5_files

def prepare_model(device, chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location=device)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    model.to(device)
    model.eval()
    return model


def extract_features_one_image(device, img, model, global_pool=True):
    x = torch.tensor(img)

    if device == "cuda":
        x = x.cuda()

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    features = model.forward_features(x.float(), global_pool=global_pool)
    return features


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def prepare_image(image_from_h5_file):
    img = Image.fromarray(image_from_h5_file)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std

    return img


def extract_features_from_h5(device, h5_filename, model, max_patches, out_features_root, model_name, global_pool):
    all_features = []
    with h5py.File(h5_filename, "r") as hdf5_file:

        if max_patches == 0:
            index = slice(0, None)
        else:
            index = slice(0, max_patches)

        patches = hdf5_file["imgs"][index]
        coords = hdf5_file["coords"][index]

        for patch in patches:
            img = prepare_image(patch)

            with torch.no_grad():
                patch_features = extract_features_one_image(device, img, model, global_pool).cpu().detach().squeeze(0).numpy()

            all_features.append(patch_features)

    if len(all_features) > 0:
        all_features = np.array(all_features)
        basename, _ = os.path.splitext(os.path.basename(h5_filename))
        feature_path = os.path.join(out_features_root, basename + "_" + model_name + "_features.h5")

        file = h5py.File(feature_path, 'w')
        dset = file.create_dataset('features', shape=all_features.shape, maxshape=all_features.shape, chunks=all_features.shape, dtype=np.float32)
        coord_dset = file.create_dataset('coords', shape=coords.shape, maxshape=coords.shape, chunks=coords.shape, dtype=np.int32)
        dset[:] = all_features
        coord_dset[:] = coords
        file.close()


def main(img_root, out_features_root, max_patches_per_image, model_name, global_pool):

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = prepare_model(device, "/exports/path-nefro-hpc/ssl_trained_models/mae_visualize_vit_large.pth", "mae_vit_large_patch16")
    # model = prepare_model(device, "demo/mae_visualize_vit_large_ganloss.pth", "mae_vit_large_patch16")
    # model = prepare_model(device, "checkpoints/mae_scratch_checkpoint-4.pth", "mae_vit_huge_patch14")

    h5_files = get_h5_list(img_root)
    for h5_filename in tqdm.tqdm(h5_files):
        full_filename = os.path.join(img_root, h5_filename)
        # print(f"Extracting features for {full_filename}")
        extract_features_from_h5(device, full_filename, model, max_patches_per_image, out_features_root, model_name, global_pool)

def main_folder():
    '''
    description: 提取png的特征
    return {*}
    '''    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    global_pool = False
    parser = argparse.ArgumentParser('Feature extraction using MAE')

    parser.add_argument('--ckpt_dir',default='/exports/path-nefro-hpc/ssl_trained_models/mae_visualize_vit_large.pth',
                        help='experiment configure file name',
                        type=str)

    parser.add_argument('--model_name', type=str, help='model used', default= "mae_visual")
    parser.add_argument('--data_path', type=str, help='data used', default= "/exports/path-nefro-hpc/PATCHES_LANCET/TRAIN_folder/")
    parser.add_argument('--result_fea_dir', type=str, help='data used', default= "/exports/path-nefro-hpc/SSL/TRAIN/FEATURES/mae_visual/")
    args = parser.parse_args()
    
    os.makedirs(args.result_fea_dir, exist_ok=True)
    model = prepare_model(device, args.ckpt_dir, "mae_vit_large_patch16")
    all_wsi = os.listdir(args.data_path)  # 列举所有的case文件夹
    # print(all_wsi[0:10])
    for i, dir in enumerate(all_wsi):  # dir的名字为case名
    # for root,dirs,files in os.walk(args.data_path):
        # for i, every_file in enumerate(files):
        file_name = dir.split(".")[0]
        # if file_name[1:3] == "19": #  and file_name.split('_')[-1] == "umcu":
        all_features = []
        all_coords = []
        feature_path = args.result_fea_dir+file_name+"_%s_features.h5"%(args.model_name)
        if os.path.exists(feature_path):
            continue
        img_path = os.path.join(args.data_path, dir+"/train/")
        for each_img in os.listdir(img_path):
            patch = cv2.imread(img_path+each_img)
            img = prepare_image(patch)
            with torch.no_grad():
                patch_features = extract_features_one_image(device, img, model, global_pool).cpu().detach().squeeze(0).numpy()

                all_features.append(patch_features)
                coor = np.array([int(x) for x in (os.path.splitext(each_img)[0].split("_"))])
                all_coords.append(coor)
        if len(all_features) > 0:
            all_features = np.array(all_features)
            # basename, _ = os.path.splitext(os.path.basename(h5_filename))
            feature_path = os.path.join(args.result_fea_dir, file_name + "_" + args.model_name + "_features.h5")

            file = h5py.File(feature_path, 'w')
            dset = file.create_dataset('features', shape=all_features.shape, maxshape=all_features.shape, chunks=all_features.shape, dtype=np.float32)
            coord_dset = file.create_dataset('coords', shape=all_coords.shape, maxshape=all_coords.shape, chunks=all_coords.shape, dtype=np.int32)
            dset[:] = all_features
            coord_dset[:] = all_coords
            file.close()
    return 0

def main_h5():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    global_pool = False
    parser = argparse.ArgumentParser('Feature extraction using MAE')

    parser.add_argument('--ckpt_dir',default='/exports/path-nefro-hpc/ssl_trained_models/mae_visualize_vit_large.pth',
                        help='experiment configure file name',
                        type=str)

    parser.add_argument('--model_name', type=str, help='model used', default= "mae_visual")
    parser.add_argument('--arch', type=str, help='model used', default= "mae_vit_large_patch16")
    parser.add_argument('--data_path', type=str, help='data used', default= "/exports/path-nefro-hpc/PATCHES_LANCET/TRAIN_folder/")
    parser.add_argument('--result_fea_dir', type=str, help='data used', default= "/exports/path-nefro-hpc/SSL/TRAIN/FEATURES/mae_visual/")
    args = parser.parse_args()
    os.makedirs(args.result_fea_dir, exist_ok=True)
    model = prepare_model(device, args.ckpt_dir, args.arch)
    all_wsi = os.listdir(args.data_path)  # 列举所有的case文件夹
    # print(all_wsi[0:10])
    for i, every_file in enumerate(all_wsi):  # every_file 的名字为case名
    # for root,dirs,files in os.walk(args.data_path):
        # for i, every_file in enumerate(files):
        file_name = every_file.split(".")[0]
        # if file_name[1:3] == "19": #  and file_name.split('_')[-1] == "umcu":
        all_features = []
        all_coords = []
        feature_path = args.result_fea_dir+file_name+"_%s_features.h5"%(args.model_name)
        if os.path.exists(feature_path):
            continue
        img_path = os.path.join(args.data_path, every_file)
        with h5py.File(img_path, "r") as hdf5_file:
            patches = hdf5_file["imgs"][:]
            coords = hdf5_file["coords"][:]

            for patch in patches:
                img = prepare_image(patch)
        # for each_img in os.listdir(img_path):
            # patch = cv2.imread(img_path+each_img)
            # img = prepare_image(patch)
                with torch.no_grad():
                    patch_features = extract_features_one_image(device, img, model, global_pool).cpu().detach().squeeze(0).numpy()

                    all_features.append(patch_features)
            if len(all_features) > 0:
                all_features = np.array(all_features)
                # basename, _ = os.path.splitext(os.path.basename(h5_filename))
                feature_path = os.path.join(args.result_fea_dir, file_name + "_" + args.model_name + "_features.h5")

                file = h5py.File(feature_path, 'w')
                dset = file.create_dataset('features', shape=all_features.shape, maxshape=all_features.shape, chunks=all_features.shape, dtype=np.float32)
                coord_dset = file.create_dataset('coords', shape=coords.shape, maxshape=coords.shape, chunks=coords.shape, dtype=np.int32)
                dset[:] = all_features
                coord_dset[:] = coords
                file.close()
    return 0

if __name__=="__main__":
    # img_root = "/exports/path-nefro-hpc/PATCHES_LANCET/TRAIN_folder/"   # TRAIN_folder umcu2019
    # out_features_root = "/exports/path-nefro-hpc/SSL/TRAIN/FEATURES/mae_visualize_vit_large-cls/"
    # max_patches_per_image = 0
    # global_pool = False

    # os.makedirs(out_features_root, exist_ok=True)

    # main(img_root, out_features_root, max_patches_per_image, "MAE", global_pool)
    # main_folder()
    main_h5()