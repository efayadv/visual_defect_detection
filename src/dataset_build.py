import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch


def buildDataset(root, subdirectory):
        images = []
        labels = []
        binary_masks = []

        if subdirectory == "train":
            good_dir = os.path.join(root, "train", "good")
            for img in os.listdir(good_dir):
                if not img.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                images.append(os.path.join(good_dir, img))
                labels.append(0)
                binary_masks.append(None)

        elif subdirectory == "test":
            test_dir = os.path.join(root, "test")
            gt_dir = os.path.join(root, "ground_truth")

            for defect_type in os.listdir(test_dir):
                defect_path = os.path.join(test_dir, defect_type)

                if not os.path.isdir(defect_path):
                    continue

                for img in os.listdir(defect_path):
                    if not img.endswith((".png", ".jpg")):
                        continue

                    img_path = os.path.join(defect_path, img)
                    images.append(img_path)

                    if defect_type == "good":
                        labels.append(0)
                        binary_masks.append(None)
                    else:
                        labels.append(1)
                        name, ext = os.path.splitext(img)
                        mask_name = f"{name}_mask{ext}"
                        mask_path = os.path.join(gt_dir, defect_type, mask_name)
                        binary_masks.append(mask_path)

        else:
            raise ValueError("subdirectory must be 'train' or 'test'")

        return images, labels, binary_masks


class CapsuleDataset(Dataset):

    def __len__(self):
        return len(self.images)

    def __init__(self, root, split="train", image_transform=None, mask_transform=None):
        self.images, self.labels, self.binary_masks = buildDataset(root, split)
        self.split = split

        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.mask_transform = mask_transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        mask_path = self.binary_masks[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask > 0).float()
        else:
            mask = torch.zeros((1, 256, 256), dtype=torch.float32)

        return image, label, mask, image_path


if __name__ == "__main__":
    train_dataset = CapsuleDataset("./capsule", split="train")
    test_dataset = CapsuleDataset("./capsule", split="test")

    print("Train size:", len(train_dataset))
    print("Test size:", len(test_dataset))