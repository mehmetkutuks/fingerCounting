import cv2
import mediapipe as mp

# kamerayı açıyoruz, çözünürlüğü ayarladık (genişlik 640, yükseklik 480)
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # genişlik ayarı
cap.set(4, 480)  # yükseklik ayarı

mpHands = mp.solutions.hands
hands = mpHands.Hands(1)  # el takibi
mpDraw = mp.solutions.drawing_utils  # çizim için araçlar
tipIds = [4, 8, 12, 16, 20]  # parmak uçlarının id numaraları

while True:
    success, img = cap.read()  # kameradan bir kare aldık
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr'dan rgb'ye çeviriyoruz
    results = hands.process(imgRGB)  # el tespiti

    xList, yList = [], []  # elin içinde x ve y koordinatlarını toplayacağız
    lmList = []  # landmark listesi, id ve koordinatlar buraya
    if results.multi_hand_landmarks:  # el varsa işleme devam
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # elimizi çiziyoruz

            for id, lm in enumerate(handLms.landmark):  # her landmark için
                h, w, _ = img.shape  # resmin boyutlarını aldık
                cx, cy = int(lm.x * w), int(lm.y * h)  # normalden piksele çeviriyoruz
                xList.append(cx)  # x'leri kaydettik
                yList.append(cy)  # y'leri kaydettik
                lmList.append([id, cx, cy])  # id ve koordinatları kaydettik

    if xList and yList:  # el koordinatları varsa
        x_min, x_max = min(xList), max(xList)  # en soldaki ve sağdaki noktalar
        y_min, y_max = min(yList), max(yList)  # en üstte ve alttaki noktalar
        # elin çevresine biraz boşluk bırakarak kare çiziyoruz
        cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

    if len(lmList) != 0:
        # sağ mı sol mu anlamak için bas parmak ve işaret parmağına bakıyoruz
        if lmList[4][1] > lmList[5][1]:
            cv2.putText(img, "sag el", (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 200), 2, cv2.LINE_AA)
        elif lmList[4][1] < lmList[5][1]:
            cv2.putText(img, "sol el", (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2, cv2.LINE_AA)

        fingers = []
        # baş parmak açık mı diye bakıyoruz, x eksenine göre
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # diğer parmaklar için y eksenine göre açık mı kapalı mı bakıyoruz
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalF = fingers.count(1)  # açık parmak sayısı
        cv2.putText(img, str(totalF), (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 215, 0), 3)

    cv2.imshow("img", img)
    cv2.waitKey(1)
