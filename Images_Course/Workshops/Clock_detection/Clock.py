import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

class ClockDetection:
    def __init__(self, img):
        self.img = img

    #Preprocess
    def _denoise(self, gray, k=5):
        return cv.GaussianBlur(gray, (k, k), 0)
    
    def _binarize(self, blur):
        th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 41, 11)
        return cv.bitwise_not(th)
    
    #Find Circle
    def find_center_radius_contour(self, gray):
        H, W = gray.shape
        cx0, cy0 = W//2, H//2
        mask = np.zeros_like(gray, np.uint8)
        max_range = 0.48
        cv.circle(mask, (cx0, cy0), int(max_range*min(H, W)), 255, -1)

        _, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        th = cv.bitwise_and(th, mask)

        cnts, _ = cv.findContours(th, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        best, best_score = None, -1.0
        for c in cnts:
            area = cv.contourArea(c)
            if area < 0.001 * (H*W): 
                continue
            peri = cv.arcLength(c, True)
            if peri == 0:
                continue
            circ = 4 * np.pi * area / (peri * peri) #circularity
            (x,y), r = cv.minEnclosingCircle(c)

            center_pen = 1.0 / (1.0 + np.hypot(x-cx0, y-cy0))
            score = circ * np.sqrt(area) * center_pen

            if score > best_score:
                best_score, best = score, (int(x), int(y), int(r))

        return best
    
    def mask_inner_disk(self, shape, cx, cy, r, inner=0.90, outer=0.98):
        h, w = shape
        mask = np.zeros((h, w), np.uint8)
        cv.circle(mask, (cx, cy), int(r * inner), 255, -1, cv.LINE_AA)
        rim = np.zeros((h, w), np.uint8)
        cv.circle(rim, (cx, cy), int(r * outer), 255, -1, cv.LINE_AA)
        cv.circle(rim, (cx, cy), int(r * inner), 0, -1, cv.LINE_AA)
        return mask, rim
    
    def apply_mask(self, th_img, mask_disk):
        return cv.bitwise_and(th_img, mask_disk)
    
    #Find Clock Hands
    def search_clock_hands(self, th_img, cx, cy, r, min_white = 127,rin_frac=0.16, rout_frac=0.97, gap_frac=0.01, fan=1):
        h, w = th_img.shape
        R = int(r)
        interal_radius = int(R * rin_frac)
        external_radius = int(R * rout_frac)
        gap_tol = max(1, int(R * gap_frac))

        response = np.zeros(360, np.float32)
        end_r    = np.zeros(360, np.int32)     

        for angle in range(360): #360 degress
            best, best_end = 0, interal_radius
            cur, gaps = 0, 0

            angs = [ (angle + d) % 360 for d in range(-fan, fan+1) ]

            for search_r in range(interal_radius, external_radius):
                is_white = False
                for a in angs:
                    theta = math.radians(a)
                    x = int(round(cx + search_r * math.cos(theta)))
                    y = int(round(cy + search_r * math.sin(theta)))
                    if 0 <= x < w and 0 <= y < h and th_img[y, x] > min_white:
                        is_white = True
                        break

                if is_white: #continue with the search
                    cur += 1 + gaps
                    gaps = 0
                else:
                    if cur > 0 and gaps < gap_tol: #little gaps inside the hands
                        gaps += 1
                    else:
                        if cur > best:
                            best = cur
                            best_end = search_r - 1
                        cur, gaps = 0, 0
            if cur > best:
                best = cur
                best_end = external_radius - 1
            
            response[angle] = best
            end_r[angle] = best_end
        
        return response, end_r
    
    #Pick hands
    def ang_dist_deg(self, a, b): #min distance in degress
        d = abs((a - b) % 360)
        return d if d <= 180 else 360 - d
    
    def top_peaks(self, response, window_deg=18):
        resp = np.asarray(response, np.float32)
        idx_sorted = np.argsort(resp)[::-1]
        taken = []

        for i in idx_sorted:
            a = int(i) % 360
            if all(self.ang_dist_deg(a, t) > window_deg for t in taken):
                taken.append(a)
                if len(taken) == 2:
                    break
        taken.sort()
        return taken 
    
    #Find Angles
    def ang_norm_deg(self, a):
        return float(a) % 360.0
    
    def angles_to_time(self, angles, lengths, r):
        if len(angles) < 2: 
            return None, None
        
        Lnorm = [l / max(1.0, float(r)) for l in lengths]
        idx_min = int(np.argmax(Lnorm))       #long: minutes
        idx_hr  = 1 - idx_min

        aM = self.ang_norm_deg(angles[idx_min])
        aH = self.ang_norm_deg(angles[idx_hr])

        phiM = self.ang_norm_deg(aM + 90.0) #12 is 0°
        phiH = self.ang_norm_deg(aH + 90.0) #12 is 0°

        M = int(round(phiM / 6.0)) % 60
        hour_float = ((phiH - 0.5 * M) / 30.0) % 12.0
        H = int(round(hour_float)) % 12
        H = 12 if H == 0 else H
        return H, M
    
    #Plot
    def draw(self, base_gray, cx, cy, angles, lengths, r):
        out = cv.cvtColor(base_gray, cv.COLOR_GRAY2BGR)
        for i, ang in enumerate(angles):
            L = int(lengths[i])
            theta = math.radians(ang)
            x2 = int(round(cx + (L + max(3, int(0.02 * r))) * math.cos(theta)))
            y2 = int(round(cy + (L + max(3, int(0.02 * r))) * math.sin(theta)))
            color = (0,0,255) if i==0 else (255,0,0) # R and B
            cv.line(out, (cx, cy), (x2, y2), color, 3, cv.LINE_AA)
            cv.circle(out, (x2, y2), 4, color, -1, cv.LINE_AA)
        cv.circle(out, (cx, cy), 6, (255,255,255), -1, cv.LINE_AA)
        return out
    
    def detect_time(self, kernel_size = 5):
        #gray image and preprocess
        gray = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)
        blur = self._denoise(gray, kernel_size)
        cx, cy, r = self.find_center_radius_contour(blur)

        #Binarize image
        th_img = self._binarize(blur)
        mask_disk, _ = self.mask_inner_disk(gray.shape, cx, cy, r)
        bin_inner = self.apply_mask(th_img, mask_disk)

        #Search hands
        all_lines, _ = self.search_clock_hands(bin_inner, cx, cy, r, min_white=200)
        peaks = self.top_peaks(all_lines)

        #Draw
        lengths = [all_lines[line] for line in peaks]
        out_img = self.draw(gray, cx, cy, peaks, lengths, r)
        H, M = self.angles_to_time(peaks, lengths, r)

        return H, M, out_img
        



