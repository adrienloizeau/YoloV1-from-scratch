import torch
import os 
import pandas as pd
from PIL import Image
from src import config
import argparse


# Questions : 
# Pourquoi 5 fois le nombre de boxes ? -> Chaque bounding box a 5 éléments

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, transform = None) -> None:
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = config.S
        self.B = config.B
        self.C = config.C
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []

        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])
            
            img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
            image = Image.open(img_path)
            boxes = torch.tensor(boxes) 

            if self.transform:
                image, boxes = self.transform(image, boxes)
            
            label_matrix = torch.zeros((self.S, self.S,self.C +5 * self.B))

            for box in boxes:
                class_label, x, y, width, height = box.tolist()
                class_label = int(class_label)
                # Divide the image in the number of cells
                i, j = int(self.S * y), int(self.S * x) 
                x_cell, y_cell = self.S * x - j,self.S * y - i
                width_cell, height_cell = (
                    width * self.S,
                    height * self.S
                )

                if label_matrix[i,j,20]==0:
                    label_matrix[i,j,20]= 1
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                        )
                    label_matrix[i,j, 21:25] = box_coordinates
                    label_matrix[i,j,class_label] = 1

            return image, label_matrix


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--csv_file', type=str, default='8examples.csv', help='Path of the csv file')
  parser.add_argument('--img_dir', type=str, default='images/', help='Path to the images')
  parser.add_argument('--label_dir', type=str, default='labels/', help='Path to the labels')
  args = parser.parse_args()

  CSV_DIR = args.csv_file
  IMG_DIR = args.img_dir
  LABEL_DIR = args.label_dir

  train_dataset = VOCDataset(
      CSV_DIR,
      transform = None,
      img_dir = IMG_DIR,
      label_dir= LABEL_DIR
  )
  image, label_matrix = train_dataset[0]

  print("####### Dataset ###### ")
  print(f"{image= }")
  print(f"{label_matrix.shape = }")
  print(f"{label_matrix = }")
  print("______________________")
  print(f"{label_matrix[1].shape = }")
  print(f"{label_matrix[1] = }")

