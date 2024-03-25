import torchvision.transforms as transforms
from PIL import Image
import time
import itt

Image.MAX_IMAGE_PIXELS = 1000000000

size = (256, 256)
batch = ["random_image10MB.jpg" for i in range(5)]

for i, image_file in enumerate(batch):
    image = Image.open(image_file)
    image = image.convert('RGB')
    crop_transform = transforms.RandomCrop(size)

    time.sleep(1)

    if i == 4:
        itt.resume()
    image = crop_transform(image)

    if i == 4:
        itt.detach()