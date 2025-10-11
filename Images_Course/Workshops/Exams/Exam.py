import cv2 as cv
import numpy as np

class ExamDetection():
    def __init__(self, img):
        self.img = img
        self.rows, self.rows_norm = self.get_rows()

    #Preprocess
    def _denoise(self, gray, k=7):
        return cv.GaussianBlur(gray, (k,k), 0)
    
    def _binarize(self, blur):
        _, thresh = cv.threshold(blur, 220, 255, cv.THRESH_BINARY)
        th = cv.bitwise_not(thresh)
        return th

    def _morphology(self, thres, k=3, struct = cv.MORPH_ELLIPSE):
        clean = cv.getStructuringElement(struct, (k,k))
        clean_img = cv.morphologyEx(thres, cv.MORPH_OPEN, clean)
        return clean_img

    def find_contours(self, clean):
        contours, _ = cv.findContours(clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return contours
    
    def find_bubbles(self, contours):
        bubbles = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if 3000 < area < 7000:
                x, y, w, h = cv.boundingRect(cnt)
                bubbles.append((x, y, w, h, cnt))
        return bubbles
    
    def find_filled(self, bubbles, clean):
        size = 3
        kernel = np.ones((size, size), np.uint8)
        dilate = cv.dilate(clean, kernel, iterations = 4)
        filled_bbs = []
        for (x, y, w, h, cnt) in bubbles:
            mask = np.zeros(dilate.shape, dtype=np.uint8)
            cv.drawContours(mask, [cnt], -1, 255, -1)
            bit = cv.bitwise_and(dilate, dilate, mask=mask)

            total_area = cv.contourArea(cnt)
            filled_area = cv.countNonZero(bit)

            fill_ratio = filled_area / float(total_area)
            if 0.6 < fill_ratio < 1.03:
                filled_bbs.append((x, y, w, h, cnt, fill_ratio, bit))
        return sorted(filled_bbs, key = lambda y: y[1])
    
    #Rows and Columns
    def to_rows(self, bubbles, n_rows = 10, Y_BIN_MARGIN = 6):
        centers_y = np.array([y + h/2 for (_, y, _, h, _, _, _) in bubbles])
        if len(centers_y) == 0:
            return [[] for _ in range(n_rows)]

        bins = np.linspace(centers_y.min() - Y_BIN_MARGIN, centers_y.max() + Y_BIN_MARGIN, n_rows + 1) #creates equally spaced along the vertical axis
        idxs = np.digitize(centers_y, bins) - 1  # 0..n_rows-1

        rows = [[] for _ in range(n_rows)]
        for idx, b in zip(idxs, bubbles):
            if 0 <= idx < n_rows:
                rows[idx].append(b)

        return rows
         
    def compute_centers(self, bubbles, n_cols = 4, sep = 0.5):
        xs = [x + w/2 for (x, _, w, _, _, _, _) in bubbles]
        xmin, xmax = min(xs), max(xs)
        step = (xmax - xmin) / n_cols
        centers = [xmin + (i+sep)*step for i in range(n_cols)]
        return np.array(centers)

    def normalize_rows(self, rows, centers):
        fill_idx = 5 #index of % fill in the bubbles
        rows_norm = []
        for r in rows:
            row_out = [None] * len(centers)   #[A, B, C, D]
            for cell in r:
                x,_, w, _,_,_,_ = cell
                cx = x + w/2
                c_idx = int(np.argmin(np.abs(centers - cx))) #finds the nearest column center
                if row_out[c_idx] is None or cell[fill_idx] > row_out[c_idx][fill_idx]: #it keeps only the one with the higher fill ratio
                    row_out[c_idx] = cell
            rows_norm.append(row_out)
        return rows_norm

    def decide_answer(self, row, rel_thresh = 0.75, abs_min = 0.30):
        ratios = []
        for cell in row:
            ratios.append(0.0 if cell is None else float(cell[5]))

        max_val = max(ratios) if ratios else 0.0
        if max_val < abs_min:
            return "BLANK", ratios, [None if c is None else c[6] for c in row]

        cand = [i for i,v in enumerate(ratios) if v >= rel_thresh * max_val]

        if len(cand) == 1:
            return "ABCD"[cand[0]], ratios, [None if c is None else c[6] for c in row]
        else:
            return "INVALID", ratios, [None if c is None else c[6] for c in row]    

    def draw_answers(self, img_rgb, rows, rows_norm, answers):
        abc = ['A', 'B', 'C', 'D']
        vis = img_rgb.copy()

        for r_idx, row in enumerate(rows):
            for c_idx, cell in enumerate(row):
                if cell is None:
                    continue
                x, y, w, h, _, _, _ = cell
                cv.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 1)
                cv.putText(vis, f"{r_idx+1}-{abc[c_idx] if c_idx < len(abc) else '?'}", (x, y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1, cv.LINE_AA)

        n = min(len(answers), len(rows_norm))
        for r_idx in range(n):
            a = answers[r_idx]
            row = rows_norm[r_idx]

            row_cells = [c for c in row if c is not None]
            if not row_cells:
                continue 
            xr, yr, wr, hr, *_ = row_cells[0]

            if a in abc:
                c_idx = abc.index(a)
                cell = row[c_idx] if c_idx < len(row) else None
                if cell is None:
                    cv.putText(vis, f"-> {a}*", (xr + wr + 8, yr + hr // 2),cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)
                    continue
                x, y, w, h, *_ = cell
                cv.rectangle(vis, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 2)
                cv.putText(vis, f"-> {a}", (x + w + 5, y + h // 2), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)
            else:
                cv.putText(vis, f"-> {a}", (xr + wr + 8, yr + hr // 2), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)

        return vis
    
    def get_rows(self):
        gray = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)
        
        blur = self._denoise(gray)
        thresh = self._binarize(blur)
        clean = self._morphology(thresh) #open morphology
        contours = self.find_contours(clean)

        bubbles = self.find_bubbles(contours)
        filled = self.find_filled(bubbles, clean)

        rows = self.to_rows(filled)
        centers = self.compute_centers(filled)
        rows_norm = self.normalize_rows(rows, centers)
        return rows, rows_norm
    
    def show_answers(self):
        answers = []
        for i, row in enumerate(self.rows_norm):
            choice, _, _ = self.decide_answer(row)
            print(f"row: {i}, choice: {choice}")
            answers.append(choice)
        
        return answers

    def show_result(self, answers):
        return self.draw_answers(self.img, self.rows, self.rows_norm, answers)