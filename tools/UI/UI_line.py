# importing the module
import cv2
import numpy as np
import json

points = []
line = {
    "x1":0,
    "y1":0,
    "x2":0,
    "y2":0,
}
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
                line["x1"] = points[0][0]
                line["y1"] = points[0][1]
                line["x2"] = points[1][0]
                line["y2"] = points[1][1]
                print(line)
                json_object = json.dumps(line, indent=4)
                with open("entry_line_config.json", "w") as outfile:
                    outfile.write(json_object)
                

           
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, new_img)

def get_points_2orthogonalplane():
    return [[], []]


if __name__ == "__main__":
    file_name = "sample.png"
    line_color = (0, 255, 0)
    line_thickness = 9

    window_name = file_name
    img = cv2.imread(file_name, 1)
    h, w, c = img.shape
    x_min, y_min = 0, 0
    x_max, y_max = w, h
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


    # # # print(f"h: {h}, w: {w}, c:{c}")
    # # # displaying the image
    # if len(points) <4:
    #     # # setting mouse handler for the image
    #     # # and calling the click_event() function
    cv2.setMouseCallback(window_name, click_event)
    # else if len(points) == 4:
    #     pts = pts.reshape((-1, 1, 2))
    #     image = cv2.polylines(image, [pts],
    #                   isClosed, color,
    #                   thickness)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)
 
    # close the window
    cv2.destroyAllWindows()