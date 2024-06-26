#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class TargetImagePublisher:
    def __init__(self):
        rospy.init_node('target_image_publisher', anonymous=True)
        self.pub = rospy.Publisher('target_image', Image, queue_size=10)
        self.bridge = CvBridge()
        self.image_path = rospy.get_param('~image_path', '')
        self.rate = rospy.Rate(0.1)  # 0.1 Hz = every 10 seconds

    def publish_image(self):
        if not self.image_path:
            rospy.logerr("No image path provided!")
            return

        img = cv2.imread(self.image_path)
        if img is None:
            rospy.logerr(f"Failed to load image from {self.image_path}")
            return

        msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        while not rospy.is_shutdown():
            self.pub.publish(msg)
            rospy.loginfo("Published target image")
            self.rate.sleep()

if __name__ == '__main__':
    try:
        publisher = TargetImagePublisher()
        publisher.publish_image()
    except rospy.ROSInterruptException:
        pass