import numpy as np
import cv2 as cv
from LogoReplacer import LogoReplacer
import yaml
import pandas as pd

class OpenCVLogoReplacer(LogoReplacer):
    def __init__(self, input_frame, logo_path):
        self.frame = input_frame
        self.logo = logo_path
        self.parameters = {}
        self.corners = 0
        self.min_ratio = 100

    def __field_detection(self, kp_template, matcher, min_match_count, dst_threshold, n_features, rc_threshold):
        gray_frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        template = cv.imread(kp_template)
        gray_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

        sift = cv.xfeatures2d.SIFT_create(n_features)

        kp1, des1 = sift.detectAndCompute(gray_template, None)
        kp2, des2 = sift.detectAndCompute(gray_frame, None)

        index_params = {'algorithm': matcher['index_params'][0], 'trees': matcher['index_params'][1]}
        search_params = {'checks': matcher['search_params']}
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < dst_threshold * n.distance:
                good.append(m)

        field = None
        if len(good) >= min_match_count:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            m, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, rc_threshold)
            h, w = gray_template.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, m)

            for i in range(len(dst)):
                if dst[i][0][1] < 0:
                    dst[i][0][1] = 0

            # cv.polylines(frame, [np.int32(dst)], True, 255, 2, cv.LINE_AA)
            x_corner_list = [dst[i][0][0] for i in range(len(dst))]
            y_corner_list = [dst[j][0][1] for j in range(len(dst))]
            x_min, x_max = np.int64(min(x_corner_list)), np.int64(max(x_corner_list))
            y_min, y_max = np.int64(min(y_corner_list)), np.int64(max(y_corner_list))
            field = self.frame[y_min:y_max, x_min:x_max]
            return field, [x_min, y_min]
        else:
            return field, 0

    def __shape_detection(self, img, origin, kernel, area_threshold):
        frame_copy = self.frame.copy()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        blur = cv.GaussianBlur(gray, (kernel, kernel), 0)

        _, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        _, contours, _ = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        area_list = [cv.contourArea(cnt) for cnt in contours]
        max_area_index = area_list.index(max(area_list))
        required_contour = contours[max_area_index]
        for i in range(len(required_contour)):
            required_contour[i][0][0] = required_contour[i][0][0] + origin[0]
            required_contour[i][0][1] = required_contour[i][0][1] + origin[1]

        epsilon = area_threshold * cv.arcLength(required_contour, True)
        approx = cv.approxPolyDP(required_contour, epsilon, True)
        # cv.drawContours(frame, [approx], -1, (0, 255, 0), 2)

        index_max_list = np.ravel(np.argmax(approx, axis=0))
        index_min_list = np.ravel(np.argmin(approx, axis=0))

        top_left = approx[index_min_list[0]].tolist()[0]
        bot_left = approx[index_max_list[1]].tolist()[0]
        bot_right = approx[index_max_list[0]].tolist()[0]
        top_right = approx[index_min_list[1]].tolist()[0]

        if bot_right[0] == top_right[0]:
            new_array = approx[np.where(approx == frame_copy.shape[1] - 1)[0].tolist()]
            new_index_max = np.ravel(np.argmax(new_array, axis=0))
            bot_right = new_array[new_index_max[1]].tolist()[0]

        left_height = np.sqrt((top_left[0]-bot_left[0])**2+(top_left[1]-bot_left[1])**2)
        top_width = np.sqrt((top_left[0] - top_right[0])**2)
        bot_width = np.sqrt((bot_left[0] - bot_right[0])**2)
        ratio_top = left_height/top_width
        ratio_bot = left_height/bot_width

        if abs(ratio_top - 1.26) > 0.1 or abs(ratio_bot - 1.26) > 0.1:
            tmp_top_r_x = top_left[0] + left_height / 1.27
            tmp_bot_r_x = bot_left[0] + left_height / 1.22
            y_top = lambda x: (x - top_left[0]) * (top_right[1] - top_left[1]) / (top_right[0] - top_left[0]) + \
                              top_left[1]

            y_bot = lambda x: (x - bot_left[0]) * (bot_right[1] - bot_left[1]) / (bot_right[0] - bot_left[0]) + \
                              bot_left[1]

            top_right[1] = y_top(tmp_top_r_x)
            bot_right[1] = y_bot(tmp_bot_r_x)
            top_right[0] = tmp_top_r_x
            bot_right[0] = tmp_bot_r_x
        print(bot_right)

        self.corners = [top_left, bot_left, bot_right, top_right]
        pts = np.array(self.corners, np.int32)
        cv.fillPoly(frame_copy, [pts], (0, 0, 255), lineType=cv.LINE_AA)
        return frame_copy

    def __transform_logo(self):
        self.logo = cv.imread(self.logo)

        frame_h, frame_w, _ = self.frame.shape
        h, w, _ = self.logo.shape

        pts1 = np.float32([(0, 0), (0, (h - 1)), ((w - 1), (h - 1)), ((w - 1), 0)])
        pts2 = np.float32([self.corners[0], self.corners[1], self.corners[2], self.corners[3]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        self.logo = cv.warpPerspective(self.logo, matrix, (frame_w, frame_h), borderMode=1)

    def build_model(self, filename):
        with open(filename, 'r') as stream:
            self.parameters = yaml.safe_load(stream)

    def detect_object(self):
        p = self.parameters
        field, origin = self.__field_detection(p['kp_template'], p['matcher'], p['min_match_count'],
                                               p['dst_threshold'], p['n_features'], p['rc_threshold'])
        if field is not None:
            frame_copy = self.__shape_detection(field, origin, p['kernel'], p['area_threshold'])
            self.__transform_logo()
            return frame_copy
        else:
            return None

    def insert_logo(self, frame_copy):
        for i in range(self.frame.shape[0]):
            for j in range(self.frame.shape[1]):
                if list(frame_copy[i, j]) == [0, 0, 255]:
                    self.frame[i, j] = self.logo[i, j]


if __name__ == '__main__':
    logo = 'superman/1X_BET_2.png'
    video = 'superman/cut_test.mp4'
    # image = '/Users/oleksandr/Folder/superman/frame2210.png'
    write_video = True
    df = pd.DataFrame(columns=['x_top_left', 'y_top_left', 'x_top_right',
                               'y_top_right', 'x_bot_left', 'y_bot_left',
                               'x_bot_right', 'y_bot_right'])
    if write_video:
        capture = cv.VideoCapture(video)
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        four_cc = cv.VideoWriter_fourcc(*'MJPG')
        out = cv.VideoWriter('result.avi', four_cc, 30, (frame_width, frame_height), True)
        min_ratio = 100
        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                logo_replacer = OpenCVLogoReplacer(frame, logo)
                logo_replacer.build_model('parameters_setting.yml')
                logo_replacer.min_ratio = min_ratio
                detected = logo_replacer.detect_object()
                # top_left, bot_left, bot_right, top_right = logo_replacer.corners
                # df = df.append({"x_top_left": top_left[0], "y_top_left": top_left[1],
                #                 "x_bot_left": bot_left[0], "y_bot_left": bot_left[1],
                #                 "x_bot_right": bot_right[0], "y_bot_right": bot_right[1],
                #                 "x_top_right": top_right[0], "y_top_right": top_right[1]},
                #                 ignore_index=True)
                if logo_replacer.min_ratio < min_ratio:
                    min_ratio = logo_replacer.min_ratio
                if detected is not None:
                    logo_replacer.insert_logo(detected)
                cv.imshow('video', frame)
                out.write(frame)
            else:
                break
            # df.to_csv("superman.csv")
            key = cv.waitKey(1)
            if key == 27:
                cv.destroyAllWindows()
        capture.release()
        out.release()
    # else:
    #     image = cv.imread(image)
    #     logo_replacer = OpenCVLogoReplacer(image, logo)
    #     logo_replacer.build_model('parameters_setting.yml')
    #     detected = logo_replacer.detect_object()
    #     if detected is not None:
    #         logo_replacer.insert_logo(detected)
    #     cv.imshow('image', image)
    #     key = cv.waitKey()
    #     if key == 27:
    #         cv.destroyAllWindows()
