import cv2

picture = cv2.imread("summer1.jpg")
cv2.imshow("Начальное изображение", picture )
cv2.waitKey(-1); cv2.destroyAllWindows()

def vP(picture, x):
    cv2.namedWindow(x, cv2.WINDOW_NORMAL); cv2.imshow(x, picture)
    cv2.waitKey(-1); cv2.destroyAllWindows()
vP(picture [15:480, 480:1900], "Обрезание изображения")

pict = cv2.imread("summer1.jpg")

d = (int(pict.shape[1] * 40 / 100), int(pict.shape[0] * 40 / 100))
vP(cv2.resize(pict, d, interpolation = cv2.INTER_AREA), "Изменение размера на 40 %")

(h, w, d) = pict.shape;  c = (w // 2, h // 2)
vP (cv2.warpAffine(pict, cv2.getRotationMatrix2D(c, 45, 1), (w, h)), "Поворот на 45 градусов")

o = pict.copy()
cv2.line(o, (810, 810), (110, 810), (20, 15, 255), 8)
vP(o, "Линия")

o = pict.copy()
cv2.putText(o, "YES!!!", (400, 400),cv2.FONT_HERSHEY_SIMPLEX, 9, (20, 205, 215), 25)
vP(o, "Текстом")
