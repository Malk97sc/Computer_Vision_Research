import cv2 as cv
import numpy as np
import pandas as pd
import os

from img_data.mapping import SHEET_CLASSES

class SheetRecognition():
    def __init__(self, data_path):
        self.data_path = data_path
        self.X = None
        self.Y = None
        self.feature_names = [
            'major_axis', 'minor_axis', 'width', 'height', 'area', 'perimeter',
            'aspect_ratio', 'form_factor', 'rectangularity', 'per_diam_ratio'
        ] + [f'hu_moment_{i+1}' for i in range(7)]
    
    def preprocess_image(self, path):
        img = cv.imread(path, cv.IMREAD_COLOR_BGR)
        if img is None:
            print(f"Fail to load {path}")
            return None, None, None, None
            
        img = cv.resize(img, (300, 300))
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        
        cnts, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv.contourArea)
        
        x, y, w, h = cv.boundingRect(c)
        crop = thresh[y:y+h, x:x+w]
        
        binary_full = thresh.copy()
        
        contour_adjusted = c - np.array([x, y])
        
        return crop, contour_adjusted, c, img_rgb, binary_full
    
    def search_axis(self, contour):
        pts = contour.reshape(-1, 2).astype(np.float32)
            
        mean, eigenvectors, eigenvalues = cv.PCACompute2(pts, np.array([]))
        projected = cv.PCAProject(pts, mean, eigenvectors)
        
        major_length = np.max(projected[:, 0]) - np.min(projected[:, 0])
        minor_length = np.max(projected[:, 1]) - np.min(projected[:, 1])
        
        return float(major_length), float(minor_length)
    
    def extract_features(self, binary_img, contour):
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)

        #Bounding box
        x, y, w, h = cv.boundingRect(contour)

        #axis by PCA
        major, minor = self.search_axis(contour)

        aspect_ratio = w / (h + 1e-6)
        form_factor = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        rectangularity = (w * h) / (area + 1e-6)
        per_diam_ratio = perimeter / (w + 1e-6)

        #Hu moments
        moments = cv.moments(binary_img)
        hu = cv.HuMoments(moments).flatten()

        #results
        features = [
            major,
            minor,
            float(w),
            float(h),
            area,
            perimeter,
            aspect_ratio,
            form_factor,
            rectangularity,
            per_diam_ratio
        ] + list(hu)

        return np.array(features, dtype=np.float32)
    
    def build_dataset(self):
        X = []
        y = []

        for cls_name, cls_info in SHEET_CLASSES.items():
            folder = os.path.join(self.data_path, cls_name)
            for img_name in os.listdir(folder):
                img_path = os.path.join(folder, img_name)

                result = self.preprocess_image(img_path)
                if result[0] is None:
                    print(f"Warning: Failed to process {img_path}")
                    continue
                    
                binary, cnt_adjusted, cnt_org, img_rgb, binary_full = result
                feats = self.extract_features(binary, cnt_org)

                X.append(feats)
                y.append(cls_info.get('label'))

        self.X = np.array(X)
        self.Y = np.array(y)
        
        return self.X, self.Y
    
    def get_info(self):
        if self.X is None or self.Y is None:
            print("Build the dataset first. Run build_dataset()")
            return None
        
        df_features = pd.DataFrame(self.X, columns=self.feature_names)        
        class_mapping = {info.get('label'): name for name, info in SHEET_CLASSES.items()}
        df_labels = pd.DataFrame({
            'class_label': self.Y,
            'class_name': [class_mapping[label] for label in self.Y]
        })
        
        df_combined = pd.concat([df_features, df_labels], axis=1)
        
        return df_combined
