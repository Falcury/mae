import os
import h5py
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging

import numpy as np



class KidneyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.is_initialized = False
        self.image_dataset = []
        self._load_h5(data_path)
        self.data_length = len(self.image_dataset)
        self.data_path = data_path

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        # print("get item==========")
        if not self.is_initialized:
            self._load_h5(self.data_path)
            self.is_initialized = True
        try:
            filename, extension = os.path.splitext(self.image_dataset[idx])
            case = filename + ".h5"
            patch_id = extension.split("_")[1]
            # case, patch_id = self.image_dataset[idx].split('_')
            f = h5py.File(case, "r")
            img = f['imgs'][int(patch_id)]
            # img = Image.fromarray(img)
            # img = torch.from_numpy(img)

            # case, patch_id = self.image_dataset[idx].split('\t')
            # # abs_p = os.path.abspath(case)
            # # case_id = os.path.splitext(abs_p)[0].split('\\')[-1]
            # f = h5py.File(case, "r")
            # img = f['imgs'][int(patch_id)]
            # img = Image.fromarray(img)

            img = self.transform(img)
            # img = torch.from_numpy(img).unsqueeze(0)
            coords = f['coords'][int(patch_id)]
            # if is_success:
            #     self.on_sucess(img)
            return img, coords  # , os.path.splitext(abs_p)[0].split('\\')[-1] # self.h5data[idx]
        except Exception as e:
            logging.warning(
                f"Couldn't load: {self.image_dataset[idx]}. Exception: \n{e}"
            )

    def _load_h5(self, data_path):
        for root, dirs, files in os.walk(data_path):
            for every_file in files:
                try:
                    f = h5py.File(root + every_file, "r")
                    # file_name, file_type = os.path.splitext(every_file)
                    for i in range(len(f['imgs'])):
                        self.image_dataset.append(root + every_file + "_%d" % (i))
                except Exception as e:
                    logging.warning(
                        f"Couldn't load: {every_file}. Exception: \n{e}"
                    )
        self.is_initialized = True


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
def show_image(image, unnormalize=True, title=''):
    # image is [H, W, 3]
    if image.shape[2] != 3:
        image = torch.einsum('chw->hwc', image)
    if unnormalize:
        plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    else:
        plt.imshow(torch.clip((image) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

if __name__=="__main__":
    import util.misc as misc
    from util.misc import NativeScalerWithGradNormCount as NativeScaler
    import matplotlib.pyplot as plt
    import time
    import tqdm

    patch_size = 256
    input_size = 224
    data_path = "/run/media/pieter/T7-Pieter/ssl/PATCHES/"
    out_dir = "/run/media/pieter/T7-Pieter/ssl/dataloader_test_result_crop/"
    distributed = False
    visualize = False

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomResizedCrop(input_size, scale=(input_size/patch_size, 1.0), interpolation=InterpolationMode.BICUBIC),  # 3 is bicubic

        # transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    batch_size = 16



    transform_none = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_colorjitter = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # minscale = input_size/patch_size
    minscale = 0.2
    transform_randomresizedcrop = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(minscale, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_randomflip = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_all = transforms.Compose([
        transforms.ColorJitter(),
        transforms.RandomResizedCrop(input_size, scale=(minscale, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_train = KidneyDataset(data_path, transform_train)


    if distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        # num_workers=args.num_workers,
        # pin_memory=args.pin_mem,
        drop_last=True,
    )

    print(dataset_train)

    os.makedirs(out_dir, exist_ok=True)

    transform_tensor_to_pil = transforms.ToPILImage()

    start = time.perf_counter()


    for i, samples in enumerate(tqdm.tqdm(data_loader_train, total=500)):
        # print(f"batch {i}")

        coords = samples[1]
        samples = samples[0]

        if visualize:

            # samples = torch.einsum('nchw->nhwc', samples)
            img = samples[0, :]
            img_colorjitter = transform_colorjitter(img)
            img_randomresizedcrop = transform_randomresizedcrop(img)
            img_randomflip = transform_randomflip(img)
            img_all = transform_all(img)

            # crops = []
            # for i in range(4):
            #     crop = transform_randomresizedcrop(img)
            #     crops.append(crop)


            # img = transform_tensor_to_pil(img)
            # img.show()
            # img.save(os.path.join(out_dir, f"batch{i}_0.png"))

            plt.rcParams['figure.figsize'] = [15, 4]

            plt.subplot(1, 5, 1)
            show_image(img, unnormalize=False, title="original")

            plt.subplot(1, 5, 2)
            show_image(img_colorjitter, unnormalize=True, title="colorjitter")

            plt.subplot(1, 5, 3)
            show_image(img_randomresizedcrop, unnormalize=True, title="randomresizedcrop")

            plt.subplot(1, 5, 4)
            show_image(img_randomflip, unnormalize=True, title="randomflip")

            plt.subplot(1, 5, 5)
            show_image(img_all, unnormalize=True, title="all")

            # plt.subplot(1, 5, 2)
            # show_image(crops[0], unnormalize=True, title="crop1")
            #
            # plt.subplot(1, 5, 3)
            # show_image(crops[1], unnormalize=True, title="crop2")
            #
            # plt.subplot(1, 5, 4)
            # show_image(crops[2], unnormalize=True, title="crop3")
            #
            # plt.subplot(1, 5, 5)
            # show_image(crops[3], unnormalize=True, title="crop4")

            plt.show()
            # plt.savefig(os.path.join(out_dir, f"batch{i}_0.png"))

        #print(samples)
        if i == 500:
            break

    print(f"Completed Execution in {time.perf_counter() - start} seconds")


