import cv2
import numpy as np
from matplotlib import pyplot as plt
trian = cv2.imread('triangle.png')
gryscl = cv2.cvtColor(trian, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gryscl, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(
	threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
i = 0
for contour in contours:
	if i == 0:
		i = 1
		continue
	approx = cv2.approxPolyDP(
		contour, 0.01 * cv2.arcLength(contour, True), True)
	cv2.drawContours(trian, [contour], 0, (0, 0, 255), 5)
	M = cv2.moments(contour)
	if M['m00'] != 0.0:
		x = int(M['m10']/M['m00'])
		y = int(M['m01']/M['m00'])
	if len(approx) == 3:
		cv2.putText(trian, 'Triangle', (x, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
	else:
		cv2.putText(trian, 'No', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
cv2.imshow('detected', trian)
cv2.waitKey(0)
cv2.destroyAllWindows()

