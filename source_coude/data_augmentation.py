import cv2
import os
import random

folder_target = r'6_muda\train'
target_jumlah_akhir = 150

def get_random_transform(image):
    opsi = ['flip_horizontal', 'flip_vertical', 'rot_90', 'rot_180', 'rot_270']
    pilihan = random.choice(opsi)
    
    if pilihan == 'flip_horizontal':
        return cv2.flip(image, 1)
    elif pilihan == 'flip_vertical':
        return cv2.flip(image, 0)
    elif pilihan == 'rot_90':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif pilihan == 'rot_180':
        return cv2.rotate(image, cv2.ROTATE_180)
    elif pilihan == 'rot_270':
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return image

def run_augmentation():
    if not os.path.exists(folder_target):
        print(folder_target)
        print("Selesai")
        return

    files = [f for f in os.listdir(folder_target) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    jumlah_sekarang = len(files)
    kekurangan = target_jumlah_akhir - jumlah_sekarang

    if kekurangan <= 0:
        print(folder_target)
        print("Selesai")
        return

    count = 0
    while count < kekurangan:
        random_file = random.choice(files)
        img_path = os.path.join(folder_target, random_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_aug = get_random_transform(img)
        new_filename = f"aug_{count+1}_{random_file}"
        save_path = os.path.join(folder_target, new_filename)
        cv2.imwrite(save_path, img_aug)

        count += 1

    print(folder_target)
    print("Selesai")

if __name__ == '__main__':
    run_augmentation()
