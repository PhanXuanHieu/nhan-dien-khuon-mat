#
# Chương trình tạo dữ liệu khuôn mặt


import cv2
import os
# Đưa thư viện cv2 và os vào chuong trinhf

output_folder = '../data/at/hieu'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
#đưa data ra folder có tên output_folder nếu không tồn tại thì tạo output_folder

face_cascade = cv2.CascadeClassifier(
    './cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    './cascades/haarcascade_eye.xml')
# sử dụng phân loại khuôn mặt và mắt có sẵn của cv2

camera = cv2.VideoCapture(0)
# mở camera mặc định
count = 0
# tạo biến đếm
while (cv2.waitKey(1) == -1):
# khi không có phím ấn từ bên ngoài thì thực hiện chương trình bên dưới, nếu có nút ấn sẽ ngắt chương trình
    success, frame = camera.read()
# đọc video từ camrea
    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #chuyển khung hình qua màu xám
        faces = face_cascade.detectMultiScale(
            gray, 1.3, 5, minSize=(120, 120))
        # 1.3 Tham số chỉ định kích thước hình ảnh được giảm bao nhiêu ở mỗi tỷ lệ hình ảnh.
        # 5 -Tham số chỉ định có bao nhiêu lân cận mỗi hình chữ nhật ứng viên để giữ lại nó.
        # Giá trị cao hơn dẫn đến ít phát hiện hơn nhưng với chất lượng cao hơn. 3 ~ 6 là một giá trị tốt cho nó.
        #MinSize - Kích thước đối tượng tối thiểu có thể. Các đối tượng nhỏ hơn bị bỏ qua.
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            face_filename = '%s/%d.pgm' % (output_folder, count)
            cv2.imwrite(face_filename, face_img)
            count += 1
            #Chúng ta sẽ lặp qua từng hình chữ nhật (mỗi mặt được phát hiện)
            # bằng cách sử dụng tọa độ của nó được tạo ra bởi hàm mà chúng ta đã nói ở trên.
            # thay đổi kích thước hình ảnh chỉ lấy ảnh mỗi khuôn mặt màu xám
            # lưu lại các tệp dưới dạng pmg
        cv2.imshow('Capturing Faces...', frame)
        # show hình ảnh video ra
