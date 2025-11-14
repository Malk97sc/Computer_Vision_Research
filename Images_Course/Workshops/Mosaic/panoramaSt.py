import cv2 as cv
import numpy as np

class Mosaic:
    def __init__(self, image_paths, ratio_test=0.7, min_matches=10):
        """
        image_paths: list of imgs
        ratio_test: ratio knn
        min_matches: min to accept the homography
        """
        if len(image_paths) < 2:
            raise ValueError("At least 2 images are needed to create a mosaic")
        self.image_paths = image_paths
        self.ratio_test = ratio_test
        self.min_matches = min_matches

        self.sift = cv.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)


    def _load_image(self, path):
        img = cv.imread(path, cv.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Can't load the img: {path}")
        return img

    def _detect_and_match(self, img1, img2):
        """
        detect sift and return kp1, kp2, good_matches
        """
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)

        matches = self.flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_test * n.distance:
                good_matches.append(m)

        return kp1, kp2, good_matches

    def _estimate_homography(self, kp1, kp2, good_matches):
        """
        kp2 to kp1
        """
        if len(good_matches) < self.min_matches:
            raise RuntimeError(f"There aren't enough good matches: {len(good_matches)} (Min: {self.min_matches})")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches] ).reshape(-1, 1, 2)#base img
        dst_pts = np.float32( [kp2[m.trainIdx].pt for m in good_matches] ).reshape(-1, 1, 2) #new img

        H, _ = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
        if H is None:
            raise RuntimeError("Can't find the homography")
        return H

    def _warp_and_blend(self, base, new_img, H):
        h1, w1 = base.shape[:2]
        h2, w2 = new_img.shape[:2]

        pano_w = w1 + w2
        pano_h = max(h1, h2)

        warped_new = cv.warpPerspective(new_img, H, (pano_w, pano_h))
        panorama = warped_new.copy()
        panorama[0:h1, 0:w1] = base

        panorama = self._crop_black(panorama)

        return panorama


    def _crop_black(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)

        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        x, y, w, h = cv.boundingRect(max(contours, key=cv.contourArea))
        return img[y:y + h, x:x + w]

    def _stitch_pair(self, base, new_img):
        #base is the actual img, "new_img" is the new to add at right.
        kp1, kp2, good_matches = self._detect_and_match(base, new_img)
        H = self._estimate_homography(kp1, kp2, good_matches)
        panorama = self._warp_and_blend(base, new_img, H)
        return panorama


    def stitch_all(self):
        base = self._load_image(self.image_paths[0])

        for path in self.image_paths[1:]:
            new_img = self._load_image(path)
            base = self._stitch_pair(base, new_img)

        return base

    def create_mosaic(self, output_path=None):
        mosaic = self.stitch_all()

        if output_path is not None:
            cv.imwrite(output_path, mosaic)

        return mosaic
