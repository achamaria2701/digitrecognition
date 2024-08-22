from keras.models import load_model
import os
import h5py
import cv2
import numpy as np

class PredictImage:
    def __init__(self, image_path):
        self.image_path = image_path
        self.model_file_path = os.path.join('artifacts','model.keras')
    
    def predict(self):
        image = cv2.imread(self.image_path, 0)
        image = 255 - image
        
        final_image_arr = self.compatible_image_converter(image)
        model = load_model(self.model_file_path)
        predicted_prob = model.predict(final_image_arr)
        predicted_digits = self.final_prediction(predicted_prob)
        return predicted_digits
        
    
    
    def compatible_image_converter(self, image):
        image_arr = self.image_localizer(image)
        number_of_images = len(image_arr)
        final_arr = np.zeros((number_of_images, 28, 28, 1))
        for i in range(number_of_images):
            resize_image = cv2.resize(image_arr[i], (28,28))
            final_arr[i,:,:,0] = resize_image
        return final_arr
    
    def final_prediction(self, predicted_prob):
        prediction = []
        for prob_arr in predicted_prob:
            prediction.append(np.argmax(prob_arr))
        return prediction
    
    def image_localizer(self, img):
        vsum = np.sum(img, axis=0)
        vdiff = np.diff(vsum.astype('float'))
        localization = []
        right = left = 0
        while vdiff[right] == 0:
            right += 1
        leftmid = right // 2
        pxl_x = right
        while pxl_x < vdiff.shape[0]:
            if vdiff[pxl_x] == 0:
                left = pxl_x
                while pxl_x < vdiff.shape[0] and vdiff[pxl_x] == 0:
                    pxl_x += 1
                right = pxl_x
                if right-left <= 5:
                    continue
                else:
                    rightmid = (left+right) // 2
                    temp = img[:,leftmid:rightmid]
                    if self.check_localized_image_validity(temp):
                        temp_3d = np.zeros((temp.shape[0], temp.shape[1], 1))
                        temp_3d[:,:,0] = temp
                        localization.append(temp_3d)
                    leftmid = rightmid
            pxl_x += 1
        return localization
    
    def check_localized_image_validity(self, image):
        rows = image.shape[0]
        cols = image.shape[1]
        hsum = np.sum(image, axis=1)
        vsum = np.sum(image, axis=0)
        hzero = len(hsum[hsum == 0])
        vzero = len(vsum[vsum == 0])
        if hzero > 0.8*rows or vzero > 0.8*cols:
            return False
        else:
            return True