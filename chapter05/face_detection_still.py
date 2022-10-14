# Chương trình phát hiện khuôn mặt với ảnh tĩnh
import cv2
# đưa thư viện cv2 vào


face_cascade = cv2.CascadeClassifier(
    './cascades/haarcascade_frontalface_default.xml')
# sử dụng thuật toán phân loại khuôn mặt và mắt có sẵn của cv2
img = cv2.imread('../images/woodcutters.jpg')
# đọc hình ảnh  từ images
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# chuyển ảnh qua màu xám
faces = face_cascade.detectMultiScale(gray, 1.08, 5)
#Chức năng DiscoveryMultiScale được sử dụng để phát hiện các khuôn mặt.
# 1.08 Tham số chỉ định kích thước hình ảnh được giảm bao nhiêu ở mỗi tỷ lệ hình ảnh. Có thể tăng lên để phát hiện nhanh nhưng mà
#cơ bỏ lỡ các khuôn mặt
# 5 -Tham số chỉ định có bao nhiêu lân cận mỗi hình chữ nhật ứng viên để giữ lại nó.
# Giá trị cao hơn dẫn đến ít phát hiện hơn nhưng với chất lượng cao hơn. 3 ~ 6 là một giá trị tốt cho nó.
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Lệnh này sẽ trả về một hình chữ nhật có tọa độ (x, y, w, h) xung quanh khuôn mặt được phát hiện


cv2.imshow('Woodcutters Detected!', img)
# display hình ảnh ra
cv2.imwrite('./woodcutters_detected.jpg', img)
# lưu hình ảnh lại
cv2.waitKey(0)
# chờ khi có bấm phím thì kết thúc chương trình
