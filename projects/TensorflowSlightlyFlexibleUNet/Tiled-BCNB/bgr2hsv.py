# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# bgr2hsv.py
# 2024/08/01

import os
import glob
import cv2
import numpy as np
import shutil
import traceback

def bgr2hsv(images_dir, output_dir):
  image_files = glob.glob(images_dir + "/*.jpg")
  for image_file in image_files:
    basename = os.path.basename(image_file)
    image = cv2.imread(image_file)
    hsvimage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    output_file = os.path.join(output_dir, basename)

    cv2.imwrite(output_file, hsvimage)
    print("--- Saved {}".format(output_file))

def sharpen(images_dir, output_dir):
  k = 1
  kernel = np.array([[-k, -k, -k], 
                       [-k, 1+8*k, -k], 
                       [-k, -k, -k]])
  image_files = glob.glob(images_dir + "/*.jpg")
  for image_file in image_files:
    basename = os.path.basename(image_file)
    image = cv2.imread(image_file)
    output_file = os.path.join(output_dir, basename)
    image = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    cv2.imwrite(output_file, image)
    print("--- Saved {}".format(output_file))
   
 

def gamma_correction(images_dir, output_dir, gamma=2):  
  image_files = glob.glob(images_dir + "/*.jpg")
  for image_file in image_files:
    basename = os.path.basename(image_file)
    image = cv2.imread(image_file)
    table = (np.arange(256) / 255) ** gamma * 255
    table = np.clip(table, 0, 255).astype(np.uint8)
    image =  cv2.LUT(image, table)  
    output_file = os.path.join(output_dir, basename)

    cv2.imwrite(output_file, image)
    print("--- Saved {}".format(output_file))

if __name__ == "__main__":
  try:
     images_dir = "./mini_test/images"
     #output_dir = "./hsv-image"
     output_dir = "./sharpend-images"
     output_dir = "./gamma_correction"
     if os.path.exists(output_dir):
       shutil.rmtree(output_dir)
     os.makedirs(output_dir)
     #bgr2hsv(images_dir, output_dir)
     #sharpen(images_dir, output_dir)
     gamma_correction(images_dir, output_dir)

  except:
    traceback.print_exc()
 
