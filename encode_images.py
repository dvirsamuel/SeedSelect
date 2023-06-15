import os
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class VaeDataset(Dataset):
    def __init__(
            self,
            data_root,
            size=512,
            interpolation="bicubic",
            center_crop=False,
    ):

        self.data_root = data_root
        self.size = size
        self.center_crop = center_crop

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.image_paths = list(filter(lambda x: not x.endswith("pt"), self.image_paths))

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def vae_encode(vae, folder_path):
    if "vae_latents.pt" in os.listdir(folder_path):
        return torch.load(f"{folder_path}/vae_latents.pt")
    else:
        with torch.no_grad():
            train_dataset = VaeDataset(
                data_root=folder_path,
                size=512,
                center_crop=False
            )
            data_loader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=5,
                                                      shuffle=False)
            all_vae_latents = []
            for batch in data_loader:
                latents = vae.encode(batch["pixel_values"].cuda()).latent_dist.sample().detach()
                latents = latents * 0.18215

                all_vae_latents += [latents]

            all_vae_latents = torch.cat(all_vae_latents)
            torch.save(all_vae_latents, f"{folder_path}/vae_latents.pt")
            return all_vae_latents


def clip_encode(clip, img_transform, folder_path):
    if "vae_latents.pt" in os.listdir(folder_path):
        return torch.load(f"{folder_path}/clip_embeddings.pt")
    else:
        with torch.no_grad():
            clip_embeddings = []
            for x in tqdm(os.listdir(folder_path)):
                if not x.endswith("pt"):
                    # image_pp = torch.from_numpy(np.array(Image.open(f"{folder_path}/{x}").convert("RGB")).astype(np.uint8)).permute(2, 0, 1)
                    image_preproceesed = img_transform(Image.open(f"{folder_path}/{x}")).cuda().unsqueeze(dim=0)
                    clip_img_features = clip.get_image_features(image_preproceesed)
                    clip_embeddings += [clip_img_features]

            clip_embeddings = torch.cat(clip_embeddings)
            torch.save(clip_embeddings, f"{folder_path}/clip_embeddings.pt")
            return clip_embeddings
