
import models_mae
import numpy as np
import torch
import os
import h5py
import tqdm
from PIL import Image


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
    # upload model to the GPU
    model.to(device)
    # put model in evaluation mode
    model.eval()
    return model


def extract_features_one_image(device, img, model, global_pool=True):
    x = torch.tensor(img)

    # upload to the GPU
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


def main(img_root, out_features_root, checkpoint_path, model_arch, max_patches_per_image, model_name, global_pool):

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = prepare_model(device, checkpoint_path, model_arch)

    h5_files = get_h5_list(img_root)
    for h5_filename in tqdm.tqdm(h5_files):
        full_filename = os.path.join(img_root, h5_filename)
        # print(f"Extracting features for {full_filename}")
        extract_features_from_h5(device, full_filename, model, max_patches_per_image, out_features_root, model_name, global_pool)


if __name__=="__main__":
    img_root = "/run/media/pieter/T7-Pieter/ssl/PATCHES"

    configs = [

        {"arch": "mae_vit_base_patch16",
         "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/mae_pretrain_vit_base.pth",
         "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_pretrain_vit_base"},

        {"arch": "mae_vit_large_patch16",
         "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/mae_pretrain_vit_large.pth",
         "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_pretrain_vit_large"},

        {"arch": "mae_vit_huge_patch14",
         "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/mae_pretrain_vit_huge.pth",
         "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_pretrain_vit_huge"},

        {"arch": "mae_vit_large_patch16",
         "checkpoint_path": "demo/mae_visualize_vit_large.pth",
         "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_visualize_vit_large"},

        {"arch": "mae_vit_large_patch16",
         "checkpoint_path": "demo/mae_visualize_vit_large_ganloss.pth",
         "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_visualize_vit_large_ganloss"},

        {"arch": "mae_vit_huge_patch14",
         "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/mae_scratch_checkpoint-0.pth",
         "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_scratch_checkpoint-0"},

        {"arch": "mae_vit_huge_patch14",
         "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/mae_scratch_checkpoint-1.pth",
         "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_scratch_checkpoint-1"},

        {"arch": "mae_vit_huge_patch14",
         "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/mae_scratch_checkpoint-2.pth",
         "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_scratch_checkpoint-2"},

        {"arch": "mae_vit_huge_patch14",
         "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/mae_scratch_checkpoint-3.pth",
         "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_scratch_checkpoint-3"},

        {"arch": "mae_vit_huge_patch14",
         "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/mae_scratch_checkpoint-4.pth",
         "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/mae_scratch_checkpoint-4"},

        {"arch": "mae_vit_huge_patch14",
         "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/checkpoint-298_scratch.pth",
         "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/checkpoint-298_scratch"},

        {"arch": "mae_vit_huge_patch14",
         "checkpoint_path": "/run/media/pieter/T7-Pieter/ssl/mae/checkpoint-299_imagenet_retrained.pth",
         "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/mae_max50_globalpool/checkpoint-299_imagenet_retrained"},
    ]


    # Set to 0 to extract all images
    max_patches_per_image = 50

    # Global pooling: if True, use mean of all tokens (excluding cls token); if False, use cls token
    global_pool = True

    for config in configs:
        out_features_root = config["out_features_root"]
        checkpoint_path = config["checkpoint_path"]
        arch = config["arch"]

        os.makedirs(out_features_root, exist_ok=True)

        print(f"Extracting features using checkpoint: {checkpoint_path}...")
        main(img_root, out_features_root, checkpoint_path, arch, max_patches_per_image, "MAE", global_pool)
