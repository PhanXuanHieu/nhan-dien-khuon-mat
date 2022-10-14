# Đào tạo để nhận diện khuôn mặt


import os
import cv2
import numpy
# đưa thư viện os, cv, numpy vào

# tạo hàm đọc dữ liệu ảnh
def read_images(path, image_size):
    names = []
    # tạo mảng lưu tên người
    training_images, training_labels = [], []
    # tạo mảng mảng để lưu lại ID  được liên kết với ảnh
    label = 0
    for dirname, subdirnames, filenames in os.walk(path):
        for subdirname in subdirnames:
            names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                img = cv2.imread(os.path.join(subject_path, filename),
                                 cv2.IMREAD_GRAYSCALE)
                if img is None:
                   #Không thể tải được hình ảnh thì bỏ qua
                    continue
                img = cv2.resize(img, image_size)
                # thay đổi kích thước hình ảnh
                training_images.append(img)
                training_labels.append(label)
                # thêm dữ liệu vào mảng
            label += 1
    training_images = numpy.asarray(training_images, numpy.uint8)
    training_labels = numpy.asarray(training_labels, numpy.int32)
    #mảng dữ liệu có sẵn( dữ liệu đầu vào, loại dữ liệu)
    return names, training_images, training_labels


path_to_training_images = '../data/at'
training_image_size = (200, 200)
names, training_images, training_labels = read_images(
    path_to_training_images, training_image_size)

model = cv2.face.EigenFaceRecognizer_create()
model.train(training_images, training_labels)

face_cascade = cv2.CascadeClassifier(
    './cascades/haarcascade_frontalface_default.xml')
# sử dụng phân loại khuôn mặt và mắt có sẵn của cv2

camera = cv2.VideoCapture(0)
# mở camera mặc định

while (cv2.waitKey(1) == -1):
    # khi không có phím ấn từ bên ngoài thì thực hiện chương trình bên dưới, nếu có nút ấn sẽ ngắt chương trình.

    success, frame = camera.read()
    # đọc video từ camrea
    if success:
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        # 1.3 Tham số chỉ định kích thước hình ảnh được giảm bao nhiêu ở mỗi tỷ lệ hình ảnh.
        # 5 -Tham số chỉ định có bao nhiêu lân cận mỗi hình chữ nhật ứng viên để giữ lại nó.
        # Giá trị cao hơn dẫn đến ít phát hiện hơn nhưng với chất lượng cao hơn. 3 ~ 6 là một giá trị tốt cho nó.
        #MinSize - Kích thước đối tượng tối thiểu có thể. Các đối tượng nhỏ hơn bị bỏ qua.

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            #Chúng ta sẽ lặp qua từng hình chữ nhật (mỗi mặt được phát hiện)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # chuyển khung hình qua màu xám
            roi_gray = gray[x:x+w, y:y+h]
            if roi_gray.size == 0:
                #Nếu roi=0,  khuôn mặt ở rìa ảnh thì bỏ qua nó
                continue
            roi_gray = cv2.resize(roi_gray, training_image_size)
            # thay đổi kích thước về
            label, confidence = model.predict(roi_gray)
            text = '%s, confidence=%.2f' % (names[label], confidence)
            cv2.putText(frame, text, (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Recognition', frame)
