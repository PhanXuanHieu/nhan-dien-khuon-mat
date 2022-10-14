# chương trình phát hiện khuôn mặt trên video

import cv2
# đưa thư viện cv2 vào


face_cascade = cv2.CascadeClassifier(
    './cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    './cascades/haarcascade_eye.xml')
# sử dụng thuật toán phân loại khuôn mặt và mắt có sẵn của cv2
camera = cv2.VideoCapture(0)
# mở camera mặc định
while (cv2.waitKey(1) == -1):
    success, frame = camera.read()
    # đọc video từ camrea
    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # chuyển khung hình qua màu xám
        faces = face_cascade.detectMultiScale(
            gray, 1.3, 5, minSize=(120, 120))
        # 1.3 Tham số chỉ định kích thước hình ảnh được giảm bao nhiêu ở mỗi tỷ lệ hình ảnh.
        # 5 -Tham số chỉ định có bao nhiêu lân cận mỗi hình chữ nhật ứng viên để giữ lại nó.
        # Giá trị cao hơn dẫn đến ít phát hiện hơn nhưng với chất lượng cao hơn. 3 ~ 6 là một giá trị tốt cho nó.
        # MinSize - Kích thước đối tượng tối thiểu có thể. Các đối tượng nhỏ hơn bị bỏ qua.
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Lệnh này sẽ trả về một hình chữ nhật có tọa độ (x, y, w, h) xung quanh khuôn mặ  được phát hiện
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(
                roi_gray, 1.1, 5, minSize=(40, 40))
            # cùng một vùng hình chữ nhật của hình ảnh thang độ xám, chúng ta thực hiện phát hiện mắt
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey),
                              (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
            #Các hình chữ nhật mắt thu được và vẽ các đường viền màu xanh lá cây 	xung quanh chúng

        cv2.imshow('Face Detection', frame)
        # show video được phát hiện khuôn và mắt ra.
