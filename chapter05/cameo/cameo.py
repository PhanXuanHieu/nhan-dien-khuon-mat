# chương trình hoán đổi khuôn mặt
import cv2
import depth
import filters
from managers import WindowManager, CaptureManager
import rects
from trackers import FaceTracker


class Cameo(object):
# tạo lớp

    def __init__(self): # hàm này sẽ được thực hiện đầu tiên
        self._windowManager = WindowManager('Cameo',
                                            self.onKeypress)
        # tạo một cửa sổ có tên Cameo và sử dụng phím nhấn khi cửa sổ này
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True)
        # mở camera mặc định ở cửa sổ mới tạo trên,
        self._faceTracker = FaceTracker()
        # gọi
        self._shouldDrawDebugRects = False
        self._curveFilter = filters.BGRPortraCurveFilter()

    def run(self):
        """Chạy vòng lặp"""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            if frame is not None:
            #khi có khung hình
                self._faceTracker.update(frame)
                faces = self._faceTracker.faces
            #  khuôn mặt sẽ thay đổi đi theo khung hình
                rects.swapRects(frame, frame,
                                [face.faceRect for face in faces])
                # thực hiện hoán đổi khuôn mặt
                filters.strokeEdges(frame, frame)
                self._curveFilter.apply(frame, frame)
                # thêm bộ lọc vào
                if self._shouldDrawDebugRects:
                    self._faceTracker.drawDebugRects(frame)
                # vẽ các nhận dạng mắt và mặt
            self._captureManager.exitFrame()
            self._windowManager.processEvents()
            # đóng khi có nhấn phím

    def onKeypress(self, keycode):
        """Khi nhấn một phím.

        space  -> Chụp ảnh lại.
        tab    -> Bắt đầu/ dừng ghi video màn hình
        x      -> Bật/tắt vẽ nhận diện mắt và mặt
        escape -> thoát.
        """
        if keycode == 32: # space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo(
                    'screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 120: # x
            self._shouldDrawDebugRects = \
                not self._shouldDrawDebugRects
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()
if __name__=="__main__":
    Cameo().run()
    # chạy chương trình chính.
