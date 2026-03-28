import os
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset


dataset_path = "./capsule"

def loadDataset(root, subdirectory):
    images = []
    labels = []
    binary_masks = []

    if subdirectory == "train":
        good_dir = os.path.join(root, "train", "good")
        for img in os.listdir(good_dir):
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
                    # masks are basically the ground truth
                    name, ext = os.path.splitext(img) # bc path names are different
                    mask_name = f"{name}_mask{ext}"
                    mask_path = os.path.join(gt_dir, defect_type, mask_name)
                    binary_masks.append(mask_path)






    






    

