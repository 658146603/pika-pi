from PIL import Image
import os


def is_valid_image(img_path):
    b_valid = True
    try:
        Image.open(img_path).verify()
    except:
        b_valid = False
    return b_valid


def trans_img(_path):
    for filename in os.listdir(_path):
        img_path = _path + '/' + filename
        if is_valid_image(img_path):
            try:
                s = img_path.rsplit(".")
                if s[-1] == 'jpg' or s[-1] == 'jpeg' or s[-1] == 'JPG' or s[-1] == 'JPEG':
                    pass
                else:
                    s = img_path.rsplit(".", 1)
                    output_img_path = s[0] + ".jpg"
                    print(output_img_path)
                    im = Image.open(img_path)
                    rgb_im = im.convert('RGB')
                    rgb_im.save(output_img_path)
            except:
                print("error1:", img_path)
        else:
            print("error2:", img_path)


if __name__ == '__main__':
    path = './test'
    print(trans_img(path))
