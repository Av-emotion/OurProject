#импортируем библиотеку OpenCV
import cv2

#загрузка каскада для обнаружения лиц
face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

#открываем первую доступную камеру
capture = cv2.VideoCapture(0)

#бесконечный цикл обработки кадров
while True:
    #захватываем кадр с камеры: успешность захвата и кадр в виде массива пикселей
    success, img = capture.read()

    # преобразуем изображение в оттенки серого
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    #обнаружение лиц на изображении в серых оттенках; метод возвращает список прямоугольников, где обнаружены лица
    faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 19)

    #для каждого обнаруженного лица рисуем прямоугольник на исходном изображении
    for (x, y, h, w) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #отображаем изображение в окне с заголовком Result
    cv2.imshow('Result', img)

    #проверка нажатия клавиши
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

    #проверка закрытия окна
    if cv2.getWindowProperty('Result', cv2.WND_PROP_VISIBLE) < 1:
        break

#освобождаем ресурсы, связанные с камерой
capture.release()

#закрываем все окна OpenCV
cv2.destroyAllWindows()