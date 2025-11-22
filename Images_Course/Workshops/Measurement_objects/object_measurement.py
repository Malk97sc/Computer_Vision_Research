import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

from Plot import plot_cv, plot_img

class ObjectMeasurement():
    def __init__(self, image, method = "min_area_rect"):
        self.img = image
        self.img_gray = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)
        self.contours = None
        self.objects_info = []
        self.reference_obj = None
        self.pixel_cm = 5.0
        self.method = method

    def preprocess_image(self, blur_kernel=(7,7), morph_kernel=(5,5)):
        self.blur = cv.GaussianBlur(self.img_gray, blur_kernel, 0)
        
        _, self.thresh = cv.threshold(self.blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        
        kernel = cv.getStructuringElement(cv.MORPH_RECT, morph_kernel)
        self.morph = cv.morphologyEx(self.thresh, cv.MORPH_CLOSE, kernel, iterations=2)
        self.morph = cv.morphologyEx(self.morph, cv.MORPH_OPEN, kernel, iterations=1)
        
        return self.morph
    
    def find_contours(self, min_area=5000):
        self.contours, _ = cv.findContours(self.morph, cv.RETR_EXTERNAL,  cv.CHAIN_APPROX_SIMPLE)
        self.contours = [c for c in self.contours if cv.contourArea(c) > min_area]
        return self.contours

    def _pca_analysis(self, contour):
        pts = contour.reshape(-1, 2).astype(np.float32)
        
        mean, eigenvectors, eigenvalues = cv.PCACompute2(pts, np.array([]))
        projected = cv.PCAProject(pts, mean, eigenvectors)
        
        major_length = np.max(projected[:, 0]) - np.min(projected[:, 0])
        minor_length = np.max(projected[:, 1]) - np.min(projected[:, 1])
        
        return float(major_length), float(minor_length), mean, eigenvectors
    
    def _min_area_rect_analysis(self, contour):
        rect = cv.minAreaRect(contour)
        (cx, cy), (bw, bh), angle = rect
        length = max(bw, bh)
        width = min(bw, bh)
        return length, width, (cx, cy), rect
    
    def analyze_objects(self):
        if self.contours is None:
            raise ValueError("You need to find the contours first.")
            
        self.objects_info = []
        
        for contour in self.contours:
            if self.method == "min_area_rect":
                length, width, center, rect = self._min_area_rect_analysis(contour)
                box = cv.boxPoints(rect)
                box = np.intp(box)
                
                info = {
                    'method': 'min_area_rect',
                    'Lpx': length,
                    'Wpx': width,
                    'area': cv.contourArea(contour),
                    'center': center,
                    'box': box,
                    'rect': rect
                }
            else:
                major, minor, mean, eigenvectors = self._pca_analysis(contour)
                length = major
                width = minor
                
                rect = cv.minAreaRect(contour)
                box = cv.boxPoints(rect)
                box = np.intp(box)
                
                info = {
                    'method': 'pca',
                    'Lpx': length,
                    'Wpx': width,
                    'area': cv.contourArea(contour),
                    'center': rect[0],
                    'box': box,
                    'rect': rect,
                    'pca_mean': mean,
                    'pca_eigenvectors': eigenvectors
                }
                
            self.objects_info.append(info)
        
        return self.objects_info
    
    def find_reference_object(self, reference_size_cm=5.0):
        if not self.objects_info:
            raise ValueError("Analyze the results first")
            
        candidates = []
        h, w = self.img_gray.shape
        
        for info in self.objects_info:
            length = info['Lpx']
            width = info['Wpx']
            ar = length / width if width > 0 else 999
            
            cx, cy = info['center']
            
            score = 0
            score += abs(ar - 1.0) * 2.0
            score += (cx / w) * 0.8 + (cy / h) * 0.8
            score += (info['area'] / (w * h)) * 5.0
            
            candidates.append((score, info))
        
        if candidates:
            self.reference_obj = sorted(candidates, key=lambda x: x[0])[0][1]
            px = (self.reference_obj['Lpx'] + self.reference_obj['Wpx']) / 2.0
            self.pixel_cm = px / reference_size_cm ##OBJ OF REFERENCE
        else:
            self.reference_obj = None
            self.pixel_cm = None
            
        return self.reference_obj
    
    def measure_objects(self, reference_size_cm=5.0):
        if not self.objects_info:
            self.analyze_objects()
            
        if self.reference_obj is None:
            self.find_reference_object(reference_size_cm)
            
        results = []
        
        for info in self.objects_info:
            if info is self.reference_obj:
                continue
                
            Lcm = info['Lpx'] / self.pixel_cm
            Wcm = info['Wpx'] / self.pixel_cm
            
            if Wcm > Lcm:
                Lcm, Wcm = Wcm, Lcm
                
            result = info.copy()
            result.update({
                'Lcm': Lcm,
                'Wcm': Wcm
            })
            results.append(result)
        
        df = pd.DataFrame(results)
        
        if not df.empty:
            display_df = df[['Lcm', 'Wcm', 'Lpx', 'Wpx', 'area']].copy()
            display_df = display_df.rename(columns={
                'Lcm': 'Length_cm',
                'Wcm': 'Width_cm', 
                'Lpx': 'Length_px',
                'Wpx': 'Width_px',
                'area': 'Area_px'
            })
            
            display_df['Length_cm'] = display_df['Length_cm'].round(2)
            display_df['Width_cm'] = display_df['Width_cm'].round(2)
            display_df['Length_px'] = display_df['Length_px'].round(1)
            display_df['Width_px'] = display_df['Width_px'].round(1)
            display_df['Area_px'] = display_df['Area_px'].round(1)
            
            display_df.index = range(1, len(display_df) + 1)
            display_df.index.name = 'Object'
            
            return results, display_df
    
    def draw_pca_axes(self, img, info, max_color=(255, 0, 0), min_color=(0, 255, 0), scale=0.5):
        if 'pca_mean' not in info or 'pca_eigenvectors' not in info:
            return img
            
        mean = info['pca_mean'][0]
        eigenvectors = info['pca_eigenvectors']
        
        major_axis = mean + eigenvectors[0] * info['Lpx'] * 0.5 * scale
        minor_axis = mean + eigenvectors[1] * info['Wpx'] * 0.5 * scale
        major_axis_neg = mean - eigenvectors[0] * info['Lpx'] * 0.5 * scale
        minor_axis_neg = mean - eigenvectors[1] * info['Wpx'] * 0.5 * scale
        
        mean_pt = tuple(mean.astype(int))
        major_pt = tuple(major_axis.astype(int))
        minor_pt = tuple(minor_axis.astype(int))
        major_neg_pt = tuple(major_axis_neg.astype(int))
        minor_neg_pt = tuple(minor_axis_neg.astype(int))
        
        cv.line(img, major_neg_pt, major_pt, min_color, 3)  #major axis
        cv.line(img, minor_neg_pt, minor_pt, max_color, 2)  #minor axis
        cv.circle(img, mean_pt, 5, (0, 0, 255), -1)  #center
        
        return img
    
    def draw_image(self, results, show_axes=True, show_ids=True):
        annot = self.img.copy()
        
        if self.reference_obj is not None:
            cv.drawContours(annot, [self.reference_obj['box']], 0, (0, 255, 0), 4)
            if show_axes and self.method == "pca":
                self.draw_pca_axes(annot, self.reference_obj, max_color=(0, 255, 0), min_color=(0, 255, 0), scale=0.5)
            
            x, y = map(int, self.reference_obj['center'])
            cv.putText(annot, "REF (5x5 cm)", (x-60, y-20), cv.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 3)
        
        for idx, info in enumerate(results, 1):
            cv.drawContours(annot, [info['box']], 0, (0, 255, 0), 4)
            
            if show_axes and self.method == "pca":
                self.draw_pca_axes(annot, info, max_color=(255, 0, 0), min_color=(255, 0, 0), scale=0.5)
            
            x, y = map(int, info['center'])
            
            cv.putText(annot, f"{info['Lcm']:.1f} x {info['Wcm']:.1f} cm", (x-80, y), cv.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 3)
            
            if show_ids:
                box_points = info['box']
                x_min = min(box_points[:, 0])
                y_min = min(box_points[:, 1])
                
                x_id = x_min + 10
                y_id = y_min + 40
                
                text = f"ID:{idx}"
                text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                cv.rectangle(annot, (x_id - 5, y_id - text_size[1] - 5), (x_id + text_size[0] + 5, y_id + 5), (0, 0, 0), -1)
                
                cv.putText(annot, text,  (x_id, y_id), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return annot
    
    def run_measurement(self, min_area = 5000, draw_id = True):
        self.preprocess_image()
        self.find_contours(min_area = min_area)
        self.analyze_objects()
        self.find_reference_object()

        results, results_df = self.measure_objects()
        img_result = self.draw_image(results=results, show_ids = draw_id)

        return img_result, results, results_df
