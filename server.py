import socket
import select
from concurrent.futures import ThreadPoolExecutor
import joblib
import cv2
import numpy as np


from handle_img import img_split, filter_noise, threshold_img
from learn import get_feature


class EventHandler:

    def need_write(self):
        pass
        
    def need_read(self):
        pass
    
    def fileno(self):
        return self._sock.fileno()
    
    def handle_receive(self):
        pass
        
    def handle_send(self):
        pass
        
        

class PublicData:
    
    index = 0

        
class TcpServer(EventHandler):
    
    def __init__(self, address, handlers, client_handler):
        self._sock = socket.socket()
        self._sock.bind(address)
        self._sock.listen()
        self.handlers = handlers
        self.client_handler = client_handler
        self.write = False
        self.read = True
    
    def handle_receive(self):
        s, a = self._sock.accept()
        print(a)
        self.handlers.append(self.client_handler(s, self.handlers))
        

class TcpClient(EventHandler):
    
    def __init__(self, sock, handlers):
        self._sock = sock
        self.msg = None
        self.read = True
        self.write = False
        self.handlers = handlers
    
    def handle_receive(self):
        try:
            data = self._sock.recv(1024)
        except Exception as e:
            print('handle_receive: ',e)
            self._sock.close()
            self.handlers.remove(self)
            print('delte')
        else:    
            self.read = False
            print(self._sock)
            pool.run(pool.discriminate_code, data, callback=lambda r: self.calc_success(r))
    
    def calc_error(self):
        self._sock.send(b'error')
        self.read = True
    
    def calc_success(self, res):
        if res == b'':
            pass
        else:
            self.read = True
            self._sock.send(res)
        
    def handle_send(self):
        msg = self.msg or b'None'
        try:
            self._sock.send(b'get: ' + msg)
        except Exception as e:
            print('handle_send:', e)
            self._sock.close()
            self.handlers.remove(self)
            print('delte')
            print(self.handlers)
            # self.read = True
            # self.write = True
        else:
            # print('success')
            self.read = True
            self.write = False
                
            
def event_loop(handlers):
    while 1:
        write_list = []
        read_list = []
        # print(len(handlers))
        for handler in handlers:
            if handler.write:
                write_list.append(handler)
            if handler.read:
                read_list.append(handler)        
        r, w, _ = select.select(read_list, write_list, [])
        for handler in r:
            handler.handle_receive()
            # print('handle_receive')
        for handler in w:
            handler.handle_send()
            # print('handle_send')
        

class ThreadPoolHandler(EventHandler):
    
    def __init__(self):
        self.pool = ThreadPoolExecutor(8)
        ser = socket.socket()
        ser.bind(('127.0.0.1', 12344))
        ser.listen(1)
        cli = socket.socket()
        cli.connect(('127.0.0.1', 12344))
        cli1, a = ser.accept()
        ser.close()
        self.done_sock = cli1
        self.signal_sock = cli
        self.write = False
        self.read = True
        self.future = []
    
    def fileno(self):
        return self.done_sock.fileno()
    
    def _compitle(self, callback, r):
        self.future.append([callback, r.result()])
        self.signal_sock.send(b'x')
    
    def run(self, func, *args, callback, **kw):
        r = self.pool.submit(func, *args, **kw)
        r.add_done_callback(lambda r: self._compitle(callback, r))
    
    def handle_receive(self):
        for callback, res in self.future:
            callback(res)
            self.done_sock.recv(1)
        self.future = []
        

class ImgThreadPoolHandler(ThreadPoolHandler):

    def __init__(self, pkl_path):
        super().__init__()
        self.knn = joblib.load(pkl_path)
        
    def discriminate_code(self, img_bin):   # 原始图片   
        # print('164 line', len(img_bin))
        if len(img_bin) == 0:
            return b''
        image = cv2.imdecode(np.asarray(bytearray(img_bin), dtype='uint8'), cv2.IMREAD_COLOR)   # 原始数据解码为np数组
        img_list = img_split(image)
        feature_list = []
        for _img in img_list:
            new_img = filter_noise(_img)    # 过滤线条干扰
            new_img_gray  = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            ret, new_img_th = cv2.threshold(new_img_gray, 250, 255, cv2.THRESH_BINARY)
            feature = get_feature(new_img_th)
            feature_list.append(feature)
        res_test = self.knn.predict(feature_list)
        print(''.join(res_test))
        return ''.join(res_test).encode('utf-8')
        
        
if __name__ == "__main__":
    pool = ImgThreadPoolHandler('code.pkl')
    handlers = []
    handlers.append(TcpServer(('127.0.0.1', 12345), handlers, TcpClient))
    handlers.append(pool)
    event_loop(handlers)
