#импортируем необходимые библиотеки
import cv2
import mediapipe as mp

#инициализируем объекты mediapipe для рисования лицевой сетки и контуров
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#создаем объект, определяющий стиль рисования для контуров лица
my_drawing_specs = mp_drawing.DrawingSpec(color = (0, 255, 0), thickness = 1)

#открываем первую доступную камеру
capture = cv2.VideoCapture(0)

#нициализируем модуль Face Mesh
mp_face_mesh = mp.solutions.face_mesh

#настраиваем модель для распознавания лицевой сетки
with mp_face_mesh.FaceMesh(
        #количество одновременно распознанных лиц на изображении
        max_num_faces = 1,
        #уточненные ключевые точки лица для детализированного результата
        refine_landmarks = True,
        #порог уверенности в том, что найденный объект является лицом
        min_detection_confidence = 0.5,
        #минимальный уровень уверенности, необходимый для отслеживания лица между кадрами
        min_tracking_confidence = 0.5
    ) as face_mesh:

    #основной цикл обработки кадров
    while capture.isOpened():
        #захватываем кадр с камеры; возвращаются успешность захвата и кадр в виде массива пикселей
        success, img = capture.read()

        #если возникли проблемы с захватом изображения с камеры, закрываем окно
        if not success:
            break

        #обработка захваченного кадра с помощью модели распознавания лицевой сетки
        results = face_mesh.process(img)

        #если на изображении распознаны лица
        if results.multi_face_landmarks:
            #то для каждого лица
            for face_landmarks in results.multi_face_landmarks:

                #рисуется лицевая сетка с использованием стиля FACEMESH_TESSELATION
                mp_drawing.draw_landmarks(
                    #изображение, на которое будут наноситься ключевые точки и их соединения
                    image = img,
                    #ключевые точки, содержащие координаты различных частей лица
                    landmark_list = face_landmarks,
                    #соединеняем ключевые точки лица в треугольники
                    connections = mp_face_mesh.FACEMESH_TESSELATION,
                    #ключевые точки рисоваться не будут
                    landmark_drawing_spec = None,
                    #используем предустановленный стиль для рисования лицевой сетки
                    connection_drawing_spec = mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )

                #рисуются контуры лица с использованием стиля рисования my_drawing_specs
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    #какие точки нужно соединить для рисования контуров (по краю лица, глаз, губ и т.д.)
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    #используем кастомный стиль для рисования контуров лица
                    connection_drawing_spec = my_drawing_specs
                    #.get_default_face_mesh_tesselation_style()
                )

                #получаем размер кадра
                h, w, _ = img.shape

                #инициализация начальных значений для границ прямоугольника
                #минимальное значение координаты x, чтобы получить самую левую границу лица
                x_min = w
                #минимальное значение координаты y, чтобы получить верхнюю границу лица
                y_min = h
                #максимальное значение координаты x, чтобы получить самую правую границу лица
                x_max = 0
                #максимальное значение координаты y, чтобы получить нижнюю границу лица
                y_max = 0

                #перебираем координаты всех ключевых точек лица и вычисляем границы прямоугольника
                for landmark in face_landmarks.landmark:
                    #вычисляем пиксельное значение координат x и у
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    #вычисляем минимальные и максимальные координаты
                    if x < x_min: x_min = x
                    if y < y_min: y_min = y
                    if x > x_max: x_max = x
                    if y > y_max: y_max = y

                #рисуем зеленый прямоугольник вокруг лица
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        #отображаем зеркальное изображение в окне с заголовком Result
        cv2.imshow('Result', cv2.flip(img, 1))

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