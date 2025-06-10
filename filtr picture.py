import cv2
import numpy as np

def improve_image_with_median_filter(image_for_median):
    processed_image = cv2.medianBlur(image_for_median, 7)
    cv2.imshow('Median Filter', processed_image)
    cv2.imwrite('median_processed_image.jpg', processed_image)

def improve_image_with_mean_filter(image_for_mean):
    kernel = np.ones((5, 5), np.float32) / 25
    processed_image = cv2.filter2D(image_for_mean, -1, kernel)
    cv2.imshow('Mean Filter', processed_image)
    cv2.imwrite('mean_processed_image.jpg', processed_image)

def improve_image_with_erode(image_for_erode):
    kernel = np.ones((5, 5), np.uint8)
    processed_image = cv2.erode(image_for_erode, kernel, iterations=1)
    cv2.imshow('Erode Filter', processed_image)
    cv2.imwrite('erode_processed_image.jpg', processed_image)

first_image = cv2.imread('first.jpg')
second_image = cv2.imread('second.jpg')
third_image = cv2.imread('third.jpg')
improve_image_with_median_filter(second_image)
improve_image_with_mean_filter(second_image)
improve_image_with_erode(second_image)
cv2.waitKey(0)
cv2.destroyAllWindows()