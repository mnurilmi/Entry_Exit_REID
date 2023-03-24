# importing the module
import argparse
import os
import cv2
import numpy as np
import json

file_name = "tools/UI/first_frame.jpg"
points = []
entry_area_config = {}


def getFirstFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        cv2.imwrite("tools/UI/first_frame.jpg", image)  # save frame as JPEG file
        print("sukses terekstrak")
        return True
    else:
        print("gagal")
        return False

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        index = klik["index"]
        src = klik["src"]
        klik["index"]+=1
        print(index)
        if len(points) <=4:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)
            points.append((x,y))
            print(points)

            # Displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(new_img, str(x) + ',' +
                        str(y), (x,y), font,
                        1, (255, 0, 0), 2)
            # case 4 points entry exit area (polygon)
            cv2.circle(new_img, (x, y), radius=1, color=(0, 0, 255), thickness=10)
            if index > 0 and index <4 :
                 cv2.line(new_img, points[index-1],points[index],line_color, line_thickness)
                 cv2.line(src, points[index-1],points[index],(255), 3)
                 
            if len(points) == 4:
                cv2.line(new_img, points[index],points[0],line_color, line_thickness)
                cv2.line(src, points[index],points[0],( 255 ), 3)
                entry_area_config["img_h"] = h
                entry_area_config["img_w"] = w
                entry_area_config["points"] = points                
                # entry_area_config["point1"] = points[0]
                # entry_area_config["point2"] = points[1]
                # entry_area_config["point3"] = points[2]
                # entry_area_config["point4"] = points[3]
                # contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # entry_area_config["contours"] = contours
                # print(len(contours))
                # print(cv2.pointPolygonTest(contours[0], points[4], True))
                # cv2.imshow("a", src)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, new_img)
        

def get_points_2orthogonalplane():
    return [[], []]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', help='testing datas location')
    opt = parser.parse_args()
    # global index
    # index = 0
    # vid_name = opt.source.split("\\")[-1]

    # print(vid_name)
    video_file = os.path.join(opt.source)
    out = os.path.join(opt.source.split(".")[0] + ".json")

    print(video_file)
    print(out)

    if getFirstFrame(video_file):
        line_color = (0, 255, 0)
        line_thickness = 9

        window_name = file_name
        img = cv2.imread(file_name, 1)
        h, w, c = img.shape
        x_min, y_min = 0, 0
        x_max, y_max = w, h
        print(w, h)
        p1 = (0, 0)
        p2 = (100,100)

        print(h, w, c)
        print(
            x_min,
            y_min,
            x_max, 
            y_max
        )

        new_img = img
        klik = {
            "index" : 0,
            "src": np.zeros((new_img.shape[0], new_img.shape[1]), dtype=np.uint8) 
        }
        # new_img = cv2.circle(new_img, p2, radius=1, color=(0, 0, 255), thickness=10)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, new_img)
        cv2.setMouseCallback(window_name, click_event)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)
    
        # close the window
        cv2.destroyAllWindows()
        if len(entry_area_config["points"]) == 4:
            json_object = json.dumps(entry_area_config, indent=4)
            with open(out, "w") as outfile:
                outfile.write(json_object)
            print("CONFIGURATION SUCCEED")
        else:
            print("CONFIGURATION FAILED, need Entry area position!")
