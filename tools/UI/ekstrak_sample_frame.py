import cv2

def getFirstFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        cv2.imwrite("tools/UI/first_frame.jpg", image)  # save frame as JPEG file
        print("sukses terekstrak")
    else:
        print("gagal")
if __name__ == "__main__":
    video_file = "test/jul2.mp4"
    getFirstFrame(video_file)