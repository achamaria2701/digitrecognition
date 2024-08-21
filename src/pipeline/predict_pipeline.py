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
        image = cv2.imread(self.image_path)
        
        final_image = self.compatible_image_converter(image)
        
        model = load_model(self.model_file_path)
        
        predicted_digit = self.final_prediction(model, final_image)

        return predicted_digit
        
    
    
    def compatible_image_converter(self, image):
        resize_image = cv2.resize(image, (28,28))
        grey_image = resize_image[:,:,0]

        final = np.zeros((100,28,28,1))
        for i in range(100):
            final[i,:,:,0] = 255-grey_image
        
        return final
    
    def final_prediction(self, model, image):
        prediction = list(model.predict(image)[0])
        temp = 0
        for i in range(len(prediction)):
            if prediction[i] > temp:
                temp = prediction[i]
                ans = i
        
        return ans