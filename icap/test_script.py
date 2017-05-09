import cv2

lourve = cv2.imread('./images/input.jpg')

cv2.imshow('Hello World', lourve)
cv2.waitKey(0)
cv2.destroyAllWindows()

quit()