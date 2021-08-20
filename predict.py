# import tensorflow as tf
import sys
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model

model = load_model("TCH.h5")

def preprocess_img(gray):
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
    binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY)[1]
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))
    dilation = cv2.dilate(binary, element, iterations = 1)

    cv2.imwrite("binary.png", binary)
    cv2.imwrite("dilation.png", dilation)

    return dilation

def find_region(dilation, img):
    region = []
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        x, y, w, h = cv2.boundingRect(box)
        try:
            crop = img[y:y+h, x:x+w]
            crop = cv2.resize(crop, (40, 40))
            crop = cv2.copyMakeBorder(crop, 5, 5, 5, 5, borderType=cv2.BORDER_REPLICATE)
            cv2.imwrite('./crop/crop'+str(i)+'.jpg', crop)
            crop = crop.astype("float32") / 255.0
            
            region.append((crop, (x, y, w, h)))
        except:
            continue
 
    return region

def paint_chinese_opencv(im, chinese, pos, color):
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('NotoSansTC-Regular.otf',25)
    fillColor = color 
    position = pos 
    
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position,chinese,font=font,fill=fillColor)
 
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    return img

def predict(img, label_name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dilation = preprocess_img(gray)
    region = find_region(dilation, img)

    boxes = [r[1] for r in region]
    words = np.array([r[0] for r in region], dtype="float32")
    
    preds = model.predict(words)

    for (pred, (x, y, w, h)) in zip(preds, boxes):
        i = np.argmax(pred)
        print(label_name[i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        #img = paint_chinese_opencv(img, label_name[i],(x-10, y-10),(0, 255, 0))

        img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        font = ImageFont.truetype('NotoSansTC-Regular.otf',30)
        draw = ImageDraw.Draw(img_PIL)
        draw.text((x, y-40), label_name[i], font=font, fill=(0, 255, 0))
        img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
        #cv2.putText(img, label_name[i],(x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
 
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
 
    cv2.imwrite("result.png", img)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if sys.argv is None:
        img_path = "test4.png"
    else:
        img_path = sys.argv[1]
    img = cv2.imread(img_path)
    
    with open('data.json', 'r') as fp:
        label_name = json.load(fp)
    
    label_name = {v: k for k, v in label_name.items()}
    predict(img, label_name)