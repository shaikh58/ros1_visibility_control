import numpy as np
import rospy
import array
import tf
import pickle
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from robot.controller.basic import BasicController
from robot.controller.mpc.drc_controller import DRCController
from robot.estimator.observer import AugmentedEKF
from env.config import *

class MyControllerNode:
    def __init__(self):
        rospy.init_node('my_controller_node', anonymous=True)

        # Subscribers
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.ref_vel_sub = rospy.Subscriber('/ref_vel', Twist, self.ref_vel_callback)

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.predicted_tgt_pub = rospy.Publisher('/pred_tgt_pose', PoseStamped, queue_size=10)
        self.fov_pub = rospy.Publisher('/fov', LaserScan, queue_size=10)

        # TF Listener
        self.tf_listener = tf.TransformListener()
        self.pose_timer = rospy.Timer(rospy.Duration(0.2), self.pose_callback)
        self.target_pose_timer = rospy.Timer(rospy.Duration(0.2), self.target_pose_callback)
        self.target_state_est_timer = rospy.Timer(rospy.Duration(0.01), self.target_state_est_callback)

        # Initialize variables
        self.map = None
        self.map_info = None
        self.scan = None
        self.target_pose = None
        self.prev_target_pose = None
        self.target_state = None
        self.prev_target_pose_ts = None
        self.target_velocity = np.array([0., 0.])
        self.pose = None
        self.ref_vel = None
        self.scan_header = None
        self.use_sampled_cbf_visibility = False
        self.use_raytracing = True
        self.fov_calc_ongoing = False
        self.fov_record = []
        self.solver = DRCController()
        self.initial_target_cov = np.diag([0, 0, 0, 0, 0])
        self.process_noise = np.diag([0, 0, 0, 1, 1])
        self.observation_noise = np.diag([0.05, 0.05, 0.05, 0.1, 0.1])

        self.estimator = AugmentedEKF(
            initial_covariance=self.initial_target_cov,
            process_noise_cov=self.process_noise,
            observation_noise_cov=self.observation_noise
        )
        self.solver_ongoing = False

    def map_callback(self, msg):
        if self.use_raytracing:
            width = msg.info.width
            height = msg.info.height
            map_data = np.array(msg.data).reshape((height, width))
            map_image = np.zeros((height, width), dtype=np.uint8)
            map_image[map_data == 0] = 255  # Free space
            map_image[map_data == -1] = 255  # Unknown space
            map_image[map_data > 0] = 0  # Occupied space
            self.map_info = {
                "height": msg.info.height,
                "resolution": msg.info.resolution,
                "origin": np.array((msg.info.origin.position.x, msg.info.origin.position.y, msg.info.origin.position.z)),
                "map": map_image[::-1]
            }

    def scan_callback(self, msg):
        self.scan = np.array(msg.ranges)
        self.scan_header = msg.header
        if not self.solver_ongoing:
            start = rospy.get_rostime()
            self.solver_ongoing = True
            pose = self.pose
            target_pose = self.target_pose
            target_velocity = self.target_velocity
            ref_vel = self.ref_vel
            scan = self.scan
            map_info = self.map_info
            if map_info is None and self.use_raytracing:
                rospy.loginfo('Raytracing mode enabled, but map not available. Cannot find optimal control.')
                self.solver_ongoing = False
                return

            # for debugging: check messages to see if any are None
            rospy.loginfo("scan" + str(type(scan)))
            rospy.loginfo("Target pose: " + str(type(target_pose)))
            rospy.loginfo(str(type(target_velocity)))
            rospy.loginfo("Agent pose: " + str(type(pose)))
            rospy.loginfo("Reference input: " + str(type(ref_vel)))
            rospy.loginfo("Map: " + str(type(map_info["map"])))
            self.control_loop(pose, target_pose, target_velocity, ref_vel, scan, map_info)
            end = rospy.get_rostime()
            rospy.loginfo(f'Controller processing time: {(end - start).to_sec() * 1000} ms')

    def target_pose_callback(self, event):
        try:
            trans = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(e)
            return

        ang = euler_from_quaternion(trans[1])
        observed_pose = np.array([trans[0][0], trans[0][1], ang[2]])
        observed_pose_ts = rospy.get_rostime()

        if self.prev_target_pose is not None:
            observed_velocity = self.compute_target_velocity(self.prev_target_pose, observed_pose, observed_pose_ts)
        else:
            observed_velocity = self.target_velocity

        observation = np.concatenate((observed_pose, observed_velocity))
        if self.target_pose is None:
            self.target_state = self.estimator.update(observation, observation)
        else:
            self.target_state = self.estimator.update(self.target_state, observation)

        self.target_pose = self.target_state[:3]
        self.target_velocity = self.target_state[3:]
        self.prev_target_pose_ts = observed_pose_ts
        self.prev_target_pose = self.target_state[:3]

    def pose_callback(self, event):
        try:
            trans = self.tf_listener.lookupTransform('map', 'map', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(e)
            return

        ang = euler_from_quaternion(trans[1])
        self.pose = np.array([trans[0][0], trans[0][1], ang[2]])

    def target_state_est_callback(self, event):
        if self.target_pose is None:
            return
        self.target_state = self.estimator.predict(self.target_state, 0.01)

        pred_tgt_pose_msg = PoseStamped()
        pred_tgt_pose_msg.header.frame_id = 'map'
        pred_tgt_pose_msg.header.stamp = rospy.get_rostime()
        pred_tgt_pose_msg.pose.position.x = self.target_state[0]
        pred_tgt_pose_msg.pose.position.y = self.target_state[1]
        self.predicted_tgt_pub.publish(pred_tgt_pose_msg)

    def ref_vel_callback(self, msg):
        self.ref_vel = np.array([msg.linear.x, msg.angular.z])

    def compute_target_velocity(self, prev_target_pose, target_pose, observed_pose_ts):
        target_pose_ts_nsec = observed_pose_ts.to_nsec()
        prev_target_pose_ts_nsec = self.prev_target_pose_ts.to_nsec()
        if np.abs(target_pose_ts_nsec - prev_target_pose_ts_nsec) < 1000:
            return self.target_velocity
        return (target_pose[:-1] - prev_target_pose[:-1]) / ((target_pose_ts_nsec - prev_target_pose_ts_nsec) / 1e9)

    def get_fov_from_lidar(self, scan, pose, stride=1):
        ix = np.arange(0, len(scan), stride)
        angles = angle_min + angle_inc * ix
        ranges = scan[ix]
        directions = np.stack([ranges * np.cos(angles), ranges * np.sin(angles)])
        theta = pose[2]
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        fov = R @ directions + pose[:2][:, None]
        fov = fov.T
        fov = np.append(fov, fov[0][None, :], axis=0)
        return fov

    def control_loop(self, pose, target_pose, target_velocity, ref_vel, scan, map_info=None):
        if scan is not None and target_pose is not None and target_velocity is not None and pose is not None and ref_vel is not None:
            scan[scan == np.inf] = radius
            fov = self.get_fov_from_lidar(scan, pose, stride=2)
            samples = self.solver.get_drc_samples(self.scan, n_samples=1)
            if self.use_sampled_cbf_visibility:
                visibility_samples = self.solver.get_visibility_cbf_samples(fov, pose, target_pose, target_velocity, map_info)
            else:
                visibility_samples = (None, None, None)

            u = self.solver.solve_drc(self.pose, self.target_pose, self.target_velocity,self.ref_vel, *samples, fov, map_info, self.use_sampled_cbf_visibility, *visibility_samples)
            self.fov_record.append(fov)
            #if len(self.fov_record) == 5:
             #   print(self.scan_header)
              #  fov_msg = LaserScan()
              #  fov_msg.header.frame_id = 'map'
              #  fov_msg.header.stamp = rospy.get_rostime()
              #  fov_msg.angle_min = self.scan_header.angle_min
              #  fov_msg.angle_max = self.scan_header.angle_max
              #  fov_msg.angle_increment = self.scan_header.angle_increment
              #  fov_msg.time_increment = self.scan_header.time_increment
              #  fov_msg.scan_time = self.scan_header.scan_time
              #  fov_msg.range_min = self.scan_header.range_min
              #  fov_msg.range_max = self.scan_header.range_max
              #  ranges = np.array(self.fov_record).reshape(-1)
              #  fov_msg.ranges = array.array('f', ranges)
              #  self.fov_pub.publish(fov_msg)
              #  self.fov_record = []

            if u is not None:
                vel_msg = Twist()
                vel_msg.linear.x = u[0]
                vel_msg.angular.z = u[1]
                self.cmd_vel_pub.publish(vel_msg)
        self.solver_ongoing = False

if __name__ == '__main__':
    try:
        node = MyControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

