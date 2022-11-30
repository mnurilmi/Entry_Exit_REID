import cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline  # if you are running this code in Jupyter notebook

tlbr = [
    [
        410, #left
        180, #top
        500, # right point from left
        500 # bottom point from top
        ],
    # [
    #     600, #left
    #     180, #top
    #     680, # right point from left
    #     1000 # bottom point from top
    #     ]
]

def getPatch(img, dets):
    # if dets is None or np.size(dets == 0):
    #     return []
    H, W, _ = np.shape(img)
    print(H, W)
    batch_patcehs = []
    patches = []

    for d in (dets):
        tlbr = d
        print(tlbr)
        tlbr[0] = max(0, tlbr[0])
        tlbr[1] = max(0, tlbr[1])
        tlbr[2] = min(W - 1, tlbr[2])
        tlbr[3] = min(H - 1, tlbr[3])
        patch = img[
            tlbr[1]:tlbr[3],
            tlbr[0]:tlbr[2],
            :
        ]
        print(tlbr)
        # patch = patch[:, :, ::-1] # Konversi Bgr2rgb
        print(patch)
        patch = cv2.resize(patch, (128, 256))

        # show image
        cv2.imshow('image',patch)
        # cv2.namedWindow("1", cv2.WINDOW_NORMAL)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    pass

if __name__ == "__main__":
    img = cv2.imread('tools/UI/first_frame.jpg') 
    getPatch(img, tlbr)
    print("hah")
    
