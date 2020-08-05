import os
import cv2

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def save_image(img_num, image, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    image_path = './{}/{}-number.jpg'.format(out_dir, img_num)
    cv2.imwrite(image_path, image)
    print('Saved to {}'.format(image_path, ))