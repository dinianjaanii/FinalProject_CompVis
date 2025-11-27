import os
import shutil
import random

target_class_folder = r'6_muda'
train_ratio = 0.8

def process_single_class_split():
    if not os.path.exists(target_class_folder):
        print(target_class_folder)
        print("Selesai")
        return

    files = [
        f for f in os.listdir(target_class_folder)
        if os.path.isfile(os.path.join(target_class_folder, f))
        and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]

    total_files = len(files)
    if total_files == 0:
        print(target_class_folder)
        print("Selesai")
        return

    random.shuffle(files)

    train_count = int(total_files * train_ratio)
    files_train = files[:train_count]
    files_test = files[train_count:]

    train_dir = os.path.join(target_class_folder, 'train')
    test_dir = os.path.join(target_class_folder, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for f in files_train:
        shutil.move(os.path.join(target_class_folder, f), os.path.join(train_dir, f))

    for f in files_test:
        shutil.move(os.path.join(target_class_folder, f), os.path.join(test_dir, f))

    print(target_class_folder)
    print("Selesai")

if __name__ == '__main__':
    process_single_class_split()
