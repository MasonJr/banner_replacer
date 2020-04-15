import pandas as pd
import numpy as np
from scipy.spatial import distance
import cv2

df = pd.read_csv('data.csv')

periods = [[15270, 15399],
           [15554, 15686],
           [20863, 21084],
           [24272, 24600],
           [54543, 54758],
           [62616, 62738],
           [78811, 79357],
           [80010, 80206],
           [80590, 81038],
           [81238, 81412],
           [83794, 83955],
           [116280, 116444],
           [153080, 153171],
           [180115, 180214],
           [211879, 212085]]
           
def center(top_left, bot_right, bot_left, top_right):
    return (top_left + bot_right + bot_left + top_right) / 4
        
def get_smoothed_points(df):

    saved_corners = df.copy(deep=True)
    smooth_df = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2',
                                      'x3', 'y3', 'x4', 'y4'])

    while saved_corners.shape[0]:

        smooth_idx = []

        prev_frame_num = saved_corners.index[0]
        prev_points = saved_corners.loc[prev_frame_num]
        prev_center_x = center(prev_points[0], prev_points[6], prev_points[4], prev_points[2])
        prev_center_y = center(prev_points[1], prev_points[7], prev_points[5], prev_points[3])

        saved_corners.drop(prev_frame_num, inplace=True)

        smooth_df.loc[prev_frame_num[0]] = list(prev_points)
        smooth_idx.append(prev_frame_num)

        for frame_num, points in saved_corners.iterrows():
            if frame_num[0] - prev_frame_num[0] == 1:
                center_x = center(points[0], points[6], points[4], points[2])
                center_y = center(points[1], points[7], points[5], points[3])
                dist = distance.euclidean([prev_center_x, prev_center_y], [center_x, center_y])
                if dist < 10:
                    smooth_df.loc[frame_num[0]] = list(points)
                    smooth_idx.append(frame_num)
                    saved_corners.drop(frame_num, inplace=True)

                    prev_center_x = center_x
                    prev_center_y = center_y
                    prev_frame_num, prev_points = frame_num, points

            elif frame_num[0] - prev_frame_num[0] > 2:
                break

        smooth_idx = pd.MultiIndex.from_tuples(smooth_idx, names=('frame', 'id'))
        smooth_df.index = smooth_idx
        if len(smooth_idx) < 60:
            df.drop(smooth_idx, inplace=True)
        else:
            df.loc[smooth_idx] = smooth_df
        smooth_df.drop(smooth_idx, inplace=True)
        
cap = cv2.VideoCapture('Avengers.mkv')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('res.mp4', four_cc, fps, (frame_width, frame_height), True)
        
ids = []
for period in periods:

    print('period: ', period)

    start = df.loc[df['frame']==period[0]].index.min()
    stop = df.loc[df['frame']==period[1]].index.max()
#     ids.append((start, stop))
    
    cdf = df.loc[start:stop].copy()
    
    ids = cdf['frame'].value_counts()
    cdf.drop(columns=['frame'], inplace=True)
    
    idxs = []
    for frame, num in ids.iteritems():
        idxs+=[(frame, i) for i in range(num)]
        idxs = sorted(idxs)
    
    midx = pd.MultiIndex.from_tuples(idxs, names=['frame', 'id'])
    cdf.index = midx
    
    get_smoothed_points(cdf)
    
    prev_idx = 0
    
    for idx, rows in cdf.iterrows():
        if idx[0] - prev_idx != 0:
            cap.set(1, idx[0])
            ret, frame = cap.read()
            
        contour = [[[rows[0], rows[1]]],
                   [[rows[2], rows[3]]],
                   [[rows[4], rows[5]]],
                   [[rows[6], rows[7]]]]
        contour = np.array(contour)
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), -1)
#         center_x = center(rows[::2])
#         box = cv2.boxPoints(np.split(rect, 4)).astype(np.float16)
#         center_y = center(rows[1::2])
#         left_ids = np.argwhere(box[:, 0] < center_x).squeeze()
#         left = box[left_ids]
#         right = np.delete(box, np.s_[left_ids], 0)
#         top_left, bot_left = left[left[:, 1].argsort(axis=0)]
#         top_right, bot_right = right[right[:, 1].argsort(axis=0)]
        
#         cv2.rectangle(frame, top_left, bot_right, (0, 255, 255), 10)
        if idx[0] - prev_idx != 0:
            out.write(frame)

        prev_idx=idx[0]

cv2.destroyAllWindows()
out.release()
cap.release()

        
        
        
        
        
