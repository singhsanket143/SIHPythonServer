# Imports for basic python libraries
import re # Package for regular expression
import base64 # Package for base64 conversion
import sys # Package for system manipulation
import  pandas as pd # Package for scientific computation
import cv2 #Package for opencv
from utils.utils import * # Importing utils folder
from utils.utils2 import * # Importing utils folder
from PIL import Image # Importing package for image manipulation
from flask import Flask, request, send_file # Importing package for flask web server
from time import time # Package for time manipulation
import urllib
import requests
import time
import base64
# from flask_cors import CORS
from flask_restful import Resource, Api

# Imports for NN implementations
from fastai.vision.transform import get_transforms
from fastai.vision.data import bb_pad_collate
from dgrec.networks.unet import Unet
from dgrec.networks.dunet import Dunet
from dgrec.networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool

torch.cuda.set_device(-1) # Sets the current divice

app = Flask(__name__) # Creating instance of flask class
# CORS(app)
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
mean = 255 * torch.tensor(imagenet_stats[0])
std  = 255 * torch.tensor(imagenet_stats[1])
sz = 224 # Size of image 

norm = lambda x: (x-mean)/ std
denorm = lambda x: x * std + mean


PATH = Path(r'data/train')
JPEGS = 'jpegs'
IMG_PATH = PATH/JPEGS

bb = pd.read_csv(PATH / 'mbb_noempties.csv')
cc = pd.read_csv(PATH / 'mc_noempties.csv')

def get_y_func(x):
    classes = cc[cc.fn == x.name].clas.values[0].split()
    bboxes = get_bbox(bb[bb.fn == x.name].bbox.values[0])
    return [bboxes, classes]

def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)', imgData1).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/')
def index():
    return "Hello ISRO"

@app.route('/predict_road')
def predict_road():
    source = 'prediction_road/'
    imgs = os.listdir(source)
    # tic = time()
    target = 'road_output/'
    if not os.path.exists(target):
        os.mkdir(target)
    for i, name in enumerate(imgs):
        print("Name:"+name)
        # if i % 10 == 0:
            # print(i / 10, '    ', '%.2f' % (time() - tic))
        mask = solver.test_one_img_from_path(source + name)
        mask[mask > 4.0] = 255
        mask[mask <= 4.0] = 0
        mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)
        cv2.imwrite(target + name[:-7] + 'mask.png', mask.astype(np.uint8))
    return "Done: Image Saved at"


@app.route('/predict_coconut', methods=['POST'])
def predict_coconut():
    batch_size = 32
    batch = []
    submission = {}
    # print(request.form)
    # print(request.files['file'])
    filename = ''
    files = []
    if 'file' not in request.files:
        img_url = request.form["link"]
        print(img_url)
        filename = '/Users/sanketsingh/Desktop/amstrong/prediction_coconut/'+str(time.time())+'.jpg'
        f = open(filename,'wb')
        f.write(requests.get(img_url).content)
        f.close()
        img = Image.open(filename)
        img.resize((224, 224), Image.ANTIALIAS).save(filename)
        files = [filename]
        # urllib.urlretrieve(img_url, "00000001.jpg")
        # return "No file "
    else:
        file_input = request.files['file']
        filename = file_input.filename
        print(filename)
        file_input.save(os.path.join('/Users/sanketsingh/Desktop/amstrong/prediction_coconut', filename))
        files = ['prediction_coconut/'+filename]
    # files = ['prediction_coconut/2.jpg']
    name_of_file = ''
    target = 'coconout_output/'
    for i, file in enumerate(files):
        if file == ".DS_Store":
            continue;
        batch.append(file)
        if (i + 1) % batch_size == 0 or (i + 1) > (len(files) - batch_size):
            batch_tensor = torch.cat(
                [torch.tensor(np.array(Image.open(image)), dtype=torch.float32).unsqueeze(0) for image in batch])
            normed_batch_tensor = norm(batch_tensor)
            normed_batch_tensor = normed_batch_tensor.permute(0, 3, 1, 2)
            preds = ssd.learn.model(normed_batch_tensor) 
            for i in range(len(batch)):
                bboxes, classes, scores = analyze_pred((preds[0][i].cpu(), preds[1][i].cpu()), thresh=0.2, ssd=ssd)
                name_of_file = show_images_with_labels([batch[i]], ((bboxes + 1) / 2 * sz).tolist(), target)
            batch = []
    image = open(name_of_file, 'rb')
    image_read = image.read() 
    image_64_encode = base64.encodestring(image_read)
    return image_64_encode
    # return str(bboxes)+str(classes)+str(scores)

solver = TTAFrame(DinkNet34)
solver.load('weights/log01_dink34.th')

data = (SSDObjectItemList.from_folder(PATH/JPEGS)
        .random_split_by_pct(0.1)
        .label_from_func(get_y_func)
        .transform(get_transforms(), tfm_y=True, size=224)
        .databunch(bs=32, collate_fn=bb_pad_collate)
        .normalize(imagenet_stats))
sys.setrecursionlimit(10000)
ssd = SingleShotDetector(data, [5,4], [1.0], [[1.0,1.0]])
ssd.load('model')

if __name__ == "__main__":
    # decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    # run the app locally on the givn port
    app.run(host='0.0.0.0', port=port)
# optional if we want to run in debugging mode
# app.run(debug=True)

