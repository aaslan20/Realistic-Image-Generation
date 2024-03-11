from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

path = "GSTRB/Training/"


class CustomTrafficSignDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.data = self.load_data()

    def load_data(self):
        data = []
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root_dir, class_name)
            csv_filename = f"GT-{class_name}.csv"

            # we skip non dictionaries (readme.txt)
            if not os.path.isdir(class_path):
                continue
            
            csv_path = os.path.join(class_path, csv_filename)
            # use delimiter, because it's treated as a single column and not separate ones
            df = pd.read_csv(csv_path, delimiter=';')

            for index, row in df.iterrows():
                data.append({
                    'image_path': os.path.join(class_path, row['Filename']),
                    'width': row['Width'],
                    'height': row['Height'],
                    'roi': (row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']),
                    'class_id': int(row['ClassId']),
                })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path, width, height, roi, class_id = (
            sample['image_path'],
            sample['width'],
            sample['height'],
            sample['roi'],
            sample['class_id']
        )

        image = Image.open(image_path)

        target_size = (30, 30)  # Set your desired target size
        resize_transform = transforms.Resize(target_size, interpolation=Image.NEAREST)
        image = resize_transform(image)

        if self.transform:
            image = self.transform(image)

        return image, class_id


root_dir = "Training"

transform = transforms.Compose([
    transforms.ToTensor()
])


dataset = CustomTrafficSignDataset(root_dir=root_dir, transform=transform)

def split_dataset(dataset, train_size):
    indices = list(range(len(dataset)))
    labels = [dataset[i][1] for i in indices]  

    train_indices, test_indices = train_test_split(indices, test_size=1.0 - train_size, stratify=labels, shuffle=True, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # Get numerical labels for the training and test datasets
    train_labels = [dataset[i][1] for i in train_indices]
    test_labels = [dataset[i][1] for i in test_indices]

    # Convert numerical labels to class names
    train_class_names = [dataset.classes[label] for label in train_labels]
    test_class_names = [dataset.classes[label] for label in test_labels]

    return train_dataset, test_dataset, train_class_names, test_class_names

train_dataset, test_dataset, train_class_names, test_class_names = split_dataset(dataset, 0.8)