import zmq
import numpy as np
import time
import cv2
from keypointserver.keypoint_helpers import *

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


if __name__=="__main__":

    config_file = "../models/racecar_pose.json"
    model_file = "../models/racecar_model.pth"
    
    model = KeypointDetector(config_file, model_file)
    context = zmq.Context()
    footage_socket = context.socket(zmq.REP)
    footage_socket.setsockopt(zmq.CONFLATE, 1)
    footage_socket.bind('tcp://*:5555')
   
    while True:
        try:
            start_time = time.time()
            msg = footage_socket.recv()
            buf = memoryview(msg)
            img_flat = np.frombuffer(buf, np.uint8)
            img = img_flat.reshape((720, 1280, 3))
            kps = model.detect_keypoints(img)
            send_array(footage_socket, kps)
            print("Total time: ")
            print(time.time() - start_time)
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            break
