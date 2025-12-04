import os
import cv2
import shutil

sumber_dataset = r'dataset/test/pecah'
folder_output = r'dataset/test_preprocessing/pecah'
IMG_SIZE = 224

def process_and_save():

    if not os.path.exists(sumber_dataset):
        return

    if os.path.exists(folder_output):
        shutil.rmtree(folder_output)

    processed_count = 0

    for root, dirs, files in os.walk(sumber_dataset):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, sumber_dataset)
                
                dest_rgb_dir = os.path.join(folder_output, 'ShuffleNet_RGB', rel_path)
                os.makedirs(dest_rgb_dir, exist_ok=True)

                dest_gray_dir = os.path.join(folder_output, 'GLCM_Grayscale', rel_path)
                os.makedirs(dest_gray_dir, exist_ok=True)

                img = cv2.imread(src_path)
                if img is None:
                    continue

                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(os.path.join(dest_rgb_dir, file), img_resized)

                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(dest_gray_dir, file), img_gray)

                processed_count += 1

if __name__ == '__main__':
    process_and_save()
