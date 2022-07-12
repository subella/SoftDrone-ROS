import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

def detect_keypoints(cv2_image):
    ar = float(image.width) / float(image.height)
    quad = get_quad(0.0, (0, 0), 1.0, aspect_ratio=ar)
    image = transform_image(image, self.image_shape, quad)
    data = self.transform(image).cuda()[None, ...]
    cmap, paf = model(data)
    cmap, paf = cmap.cpu(), paf.cpu()
    object_counts, objects, peaks = self.parse_objects(cmap, paf)
    object_counts, objects, peaks = int(object_counts[0]), objects[0], peaks[0]
    
    for i in range(object_counts):
        kps = [0]*(33*3)
        cnt = 0
        for j in range(33):
            k = object[j]
            if k >= 0:
                peak = peaks[j][k]
                if ar > 1.0: # w > h w/h
                    x = peak[1]
                    y = (peak[0] - 0.5) * ar + 0.5
                else:
                    x = (peak[1] - 0.5) / ar + 0.5
                    y = peak[0]

                x = round(float(img['width'] * x))
                y = round(float(img['height'] * y))
                kps[j * 3 + 0] = x
                kps[j * 3 + 1] = y
                kps[j * 3 + 2] = 2
                # r = 2
                # drawing.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0,0))



def image_callback(msg):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        detect_keypoints(cv2_image)

def main():
    bridge = CvBridge()
    rospy.init_node('image_listener')
    image_topic = "rgb_img_in"
    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
