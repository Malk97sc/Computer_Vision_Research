import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

class HourDetection:
    def __init__(self, img):
        self.img = img
        self.max_value = 255

    def _blurr_img(self, kernel_size):
        kernel = (kernel_size, kernel_size)
        blurred_img = cv.GaussianBlur(self.img, kernel, 0)
        return blurred_img

    def _threshold_img(self, blurred_img, thres_value=124 ,max_value = 255):
        _, thres_img = cv.threshold(blurred_img , thres_value, max_value, cv.THRESH_BINARY)
        return thres_img

    def _find_countours(self, thres_img):
        contours, _ = cv.findContours(thres_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        return contours

    def _search_center_contour(self, contours):
        center_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(center_contour)

        clock = self.img[y:y+h, x:x+w].copy() #new image
        c_shifted = center_contour - [x, y] #new contours
        return clock, c_shifted
    
    def _get_center(self, img):
        cx, cy = [], []
        height, width = img.shape

        for i in range(height):
            for j in range(width):
                if img[i, j] != 0:
                    cx.append(i)
                    cy.append(j) 
        return cx, cy
    
    def _get_radius(self, contours, center_x, center_y, point_index):
        if point_index > len(contours[0]):
            raise IndexError("Out limits")
        
        point = contours[0][point_index][0] 
        r = np.sqrt((center_x - point[0])**2 + (center_y - point[1])**2)
        return r, point
    
    def _detect_hands_lengths(self, img_bin, cx, cy, radius, amount_lines=2, start_skip=5, gap_tol=3):
        h, w = img_bin.shape
        results = []

        for ang in range(0, 360, 1):
            theta = math.radians(ang) #degree to radian
            white = 0 #amount of white pixels
            black = 0 #black pixels
            seen_wht = False 

            for r in range(start_skip, radius):
                x = int(round(cx + r * math.cos(theta)))
                y = int(round(cy + r * math.sin(theta)))
                if not (0 <= x < w and 0 <= y < h):
                    break

                if img_bin[y, x] == 255:
                    seen_wht = True
                    white += 1
                    black = 0
                else:
                    if seen_wht:
                        black += 1
                        if black >= gap_tol:
                            break
            
            length = white
            results.append((length, ang))

        results.sort(reverse=True, key=lambda t: t[0])
        return results[:amount_lines] #(length, angle)
    
    def _cluster_by_angle(self, canditates, interval_deg=10):
        if not canditates:
            return []

        cands = sorted(canditates, key=lambda t: t[1])

        clusters = []
        cur = [cands[0]]
        last_angle = cands[0][1]

        for length, angle in cands[1:]:
            if (angle - last_angle) <= interval_deg:
                cur.append((length, angle))
            else:
                clusters.append(cur)
                cur = [(length, angle)]
            last_angle = angle

        clusters.append(cur)

        return clusters #(length, angle)
    
    def _pick_biggest(self, cluster):
        #cluster = (length, angle)
        reps = []
        for group in cluster: 
            best = max(group, key = lambda x: x[0]) #pick the length
            angles = [a for _, a in group]
            reps.append({
                "angle": best[1] % 360.0,
                "length": best[0],
                "count": len(group),
                "min_angle": min(angles) % 360.0,
                "max_angle": max(angles) % 360.0,
            })
        return reps
    
    def _draw_hands_on_image(self, img, cx, cy, reps, start_skip=5, thickness=3):
        out_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        palette = [(0, 0, 255), (255, 0, 0)] #red and blue

        for i, d in enumerate(reps):
            ang = float(d["angle"])
            L = int(d["length"])
            L_draw = L + int(start_skip)

            theta = math.radians(ang)
            xe = int(round(cx + L_draw * math.cos(theta)))
            ye = int(round(cy + L_draw * math.sin(theta)))

            color = palette[i % len(palette)]
            cv.line(out_img, (int(cx), int(cy)), (xe, ye), color, thickness, cv.LINE_AA)
            cv.circle(out_img, (xe, ye), 4, color, -1, cv.LINE_AA) #end line

        cv.circle(out_img, (int(cx), int(cy)), 8, (255, 255, 255), -1, cv.LINE_AA) #center circle
        return out_img
    
    def _get_time(self, reps):
        if not reps or len(reps) < 2:
            return None, None

        a, b = sorted(reps, key=lambda d: d["length"], reverse=True)
        minute_rep = a["angle"]
        hour_rep = b["angle"]

        phiM = (float(minute_rep) + 90.0) % 360.0
        phiH = (float(hour_rep) + 90.0) % 360.0

        #minutes 
        M = int(round(phiM / 6.0)) % 60

        #hour
        hour_float = ((phiH - 0.5 * M) / 30.0) % 12.0
        H = int(round(hour_float)) % 12
        H = 12 if H == 0 else H

        return H, M
    
    def detect_time(self, kernel_size = 5, thres_value = 120, amount_lines = 20):
        blurred = self._blurr_img(kernel_size)
        thres_img = self._threshold_img(blurred, thres_value)
        
        contours = self._find_countours(thres_img)
        clock, center_contours = self._search_center_contour(contours)
        clock_thres = self._threshold_img(clock)
        clock_thres = cv.bitwise_not(clock_thres)

        xs, ys = self._get_center(clock)
        center_x = int(np.mean(ys))
        center_y = int(np.mean(xs))

        r, points = self._get_radius([center_contours], center_x, center_y, 0)

        hands = self._detect_hands_lengths(clock_thres, 
                                           center_x,
                                           center_y,
                                           int(r),
                                           amount_lines)
        hands.sort(key = lambda x: x[1])

        hands_clusters = self._cluster_by_angle(hands)
        best_values = self._pick_biggest(hands_clusters)
        output_img = self._draw_hands_on_image(clock_thres, center_x, center_y, best_values)

        H, M = self._get_time(best_values)
        return H, M, output_img
