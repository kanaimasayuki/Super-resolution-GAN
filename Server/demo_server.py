# -*- coding: utf-8 -*-
"""
    SRGAN デモ用
    (C) Masayuki Kanai 2021/2/25
"""

#%% インポートなど

import sys, os
import numpy as np
import pandas as pd
import glob
import flask as flk
# request フォームから送信した情報を扱うためのモジュール
# redirect  ページの移動
# url_for アドレス遷移
from flask import request, redirect, url_for
# 画像のダウンロード
from flask import send_from_directory

import cv2
from datetime import datetime
import string
import random
import tensorflow as tf

sys.path.append(os.path.abspath(".."))
import srgan_pre


#%% 準備

app = flk.Flask(__name__)

# # ベーシック認証->
# from flask import Flask
# from flask_httpauth import HTTPBasicAuth

# auth = HTTPBasicAuth()

# users = {
#     "heroz": "HEROZ",
#     "aruhi": "ARUHI"
# }

# @auth.get_password
# def get_pw(username):
#     if username in users:
#         return users.get(username)
#     return None
# # ベーシック認証<-

# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = './uploads'
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
# # アップロードされる拡張子の制限
# ALLOWED_EXTENSIONS = set(['png', 'jpg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def SRGAN(img):
    img = cv2.resize(img, None, fx=2.0, fy=2.0)
    imheight, imwidth, channels = img.shape[:3]
    # Network Setup
    model = srgan_pre.Generator(input_shape=(imheight, imwidth, 3))
    model.build((None, imheight, imwidth, 3))
    model.load_weights('srgan.h5')

    # Image processing
    img = tf.convert_to_tensor(img, np.float32) 
    img = tf.convert_to_tensor(img, np.float32) 
    img = (img - 127.5) / 127.5
    img = img[np.newaxis, :, :, :]

    # Super-resolution
    re = model.predict(img)
    # Super-resolution Image processing
    re = np.reshape(re, (imheight, imwidth, 3))
    re = re * 127.5 + 127.5
    re = np.clip(re, 0.0, 255.0)

    return re

def random_str(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])

@app.route('/')
def index():
    return flk.render_template('index.html', images=os.listdir(UPLOAD_FOLDER)[::-1])

@app.route('/uploads/<path:path>')
def send_js(path):
    return send_from_directory(UPLOAD_FOLDER, path)
    
@app.route('/upload', methods=['POST'])
def upload():
    if request.files['image']:
        #フォルダ内のデータ削除
        file_list = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
        for file in file_list:
            os.remove(file)

        # 画像として読み込み
        stream = request.files['image'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        oriimg = cv2.imdecode(img_array, 1)

        # 変換
        img = SRGAN(oriimg)

        # 保存
        dt_now = datetime.now().strftime("%Y_%m_%d%_H_%M_%S_") + random_str(5)
        upload_path = os.path.join(UPLOAD_FOLDER, dt_now + '_SRGAN' + ".png")
        cv2.imwrite(upload_path, img)
        upload_path = os.path.join(UPLOAD_FOLDER, dt_now + ".png")
        cv2.imwrite(upload_path, oriimg)

        print("save", upload_path)

        return redirect('/')

        

#%% API定義


#%% サーバ起動

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    #app.run(host="0.0.0.0", port=8080, debug=True)
