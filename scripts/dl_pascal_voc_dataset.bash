# Download dataset
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d aladdinpersson/pascalvoc-yolo

# Unzip dataset and store it
unzip pascalvoc-yolo.zip dataset/voc_dataset
rm pascalvoc-yolo.zip