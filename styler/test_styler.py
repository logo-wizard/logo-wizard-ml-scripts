import PIL
from canny_styler import Canny

if __name__ == "__main__":
    styler = Canny()
    image = PIL.Image.open("test_data/test.jpg")
    result = styler(image=image, style="modern logo style").images[0]
    result.save("test_data/result.jpg")
