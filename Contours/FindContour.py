import cv2 as cv
import numpy as np

class FindContour():
    def __init__(self, img):
        self.height, self.width = img.shape
        self.img = img
        self.neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 1),
             (1, 0), (1, -1), (0, -1), (-1, -1)]

    def find_start(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.img[i, j] == 255:
                    return (i, j)
        return None
    
    def search_contour(self):
        start = self.find_start()
        if start is None:
            return []
        
        n_search = 8
        contour = [start]
        current = start
        prev_dir = 7

        while 1:
            found = False

            for i in range(n_search):
                idx = (prev_dir + i) % n_search
                dx, dy = self.neighbors[idx]
                neigh_i, neigh_j = current[0] + dx, current[1] + dy 

                if (0 <= neigh_i < self.height) and (0 <= neigh_j < self.width):
                    if self.img[neigh_i, neigh_j] == 255: 
                        contour.append((neigh_i, neigh_j))
                        current = (neigh_i, neigh_j)
                        prev_dir = (idx + 5) % n_search
                        found = True
                        break

            if not found or current == start:
                break
        return contour
        
    def contour2CV(self, contour):
        contour_xy = [(x,y) for (y,x) in contour]
        contour_np = np.array(contour_xy, dtype=np.int32).reshape((-1,1,2))
        return [contour_np]
