# importing the module
import cv2
import numpy as np
import json

points = []
line = {}
# video_file = "test/jul2_miror.mp4"
# file_name = "tools/UI/first_frame.jpg"
# out = "configs/entry_line_config.json"

# video_file = "test/jul2.mp4"
# file_name = "tools/UI/first_frame.jpg"
# out = "configs/entry_line_config2.json"

video_file = "test/jul2.mp4"
file_name = "tools/UI/first_frame.jpg"
out = "configs/entry_line_config3.json"

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
        if len(points) >=2:
            print("JUMLAH TITIK HARUS 2!")
        elif len(points)<2:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)
            points.append([x, y])
            print(points)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(new_img, str(x) + ',' +
                        str(y), (x,y), font,
                        1, (255, 0, 0), 2)
                        
            cv2.circle(new_img, (x, y), radius=1, color=(0, 0, 255), thickness=10)

            if len(points) == 2:
                #Line 1
                # print(points)
                cv2.line(new_img, (points[0][0],points[0][1]), (points[1][0],points[1][1]), line_color, line_thickness)
                line["img_h"] = h
                line["img_w"] = w
                line["x1"] = points[0][0]
                line["y1"] = points[0][1]
                line["x2"] = points[1][0]
                line["y2"] = points[1][1]
                # print("gradien: ", -1*(line["y1"]-line["y2"])/(line["x1"]-line["x2"]))
                # line["gradien"] = -1*(line["y1"]-line["y2"])/(line["x1"]-line["x2"])
                print(line)
                json_object = json.dumps(line, indent=4)
                with open(out, "w") as outfile:
                    outfile.write(json_object)
                
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, new_img)

def get_points_2orthogonalplane():
    return [[], []]

if __name__ == "__main__":
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
        # new_img = cv2.circle(new_img, p2, radius=1, color=(0, 0, 255), thickness=10)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, new_img)
        cv2.setMouseCallback(window_name, click_event)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)
    
        # close the window
        cv2.destroyAllWindows()
        print("CONFIGURATION SUCCEED")
    else:
        print("CONFIGURATION FAILED")