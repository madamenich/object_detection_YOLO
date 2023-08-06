import cv2
import matplotlib.pyplot as plt

def grid_display(n_grid,img_src):
    # im = cv2.imread(img_src)
    im = img_src
    imgheight=im.shape[0]
    imgwidth=im.shape[1]
    y1 = 0
    M = imgheight//n_grid
    N = imgwidth//n_grid

    for y in range(0,imgheight,M):
       for x in range(0, imgwidth, N):
            y1 = y + M
            x1 = x + N
            tiles = im[y:y+M,x:x+N]

            cv2.rectangle(im, (x, y), (x1, y1), (0, 0, 255))
       
            # cv2.imwrite("save/" + str(x) + '_' + str(y)+".png",tiles)

    # cv2.imwrite("gridded.png",im)