import numpy as np
import time
from datetime import datetime
import rospy
import array
import tf
import pickle
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Point
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from robot.controller.mpc.drc_controller import DRCController
from robot.estimator.observer import AugmentedEKF
from env.config import *
from robot.controller.basic_cvx import BasicController

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
        self.fov_viz_pub = rospy.Publisher('/fov_viz', Marker, queue_size=10)

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
        self.raytraced_fov_record = []
        self.sdf_record = []
        self.pose_record = []
        self.ref_control_record = []
        self.target_velocity_record = []
        self.target_pose_record = []
        self.lidar_min_dist_record = []
        self.data_output_time_str = None
        self.data_pkl = None
        self.data_record_ts = None
        self.solver = BasicController() # DRCController()
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
            # rospy.loginfo("scan" + str(type(scan)))
            # rospy.loginfo("Target pose: " + str(type(target_pose)))
            # rospy.loginfo(str(type(target_velocity)))
            # rospy.loginfo("Agent pose: " + str(type(pose)))
            # rospy.loginfo("Reference input: " + str(type(ref_vel)))
            # rospy.loginfo("Map: " + str(type(map_info["map"])))
            self.control_loop(pose, target_pose, target_velocity, ref_vel, scan, map_info)
            end = rospy.get_rostime()
            rospy.loginfo(f'Controller processing time: {(end - start).to_sec() * 1000} ms')

    def target_pose_callback(self, event):
        try:
            trans = self.tf_listener.lookupTransform('map', 'tag_frame', rospy.Time(0))
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
            trans = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0))
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

    def publish_viz(self, scan):
        marker = Marker()
        marker.header.frame_id = "map"  # Adjust based on your coordinate frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "shape"
        marker.id = 0
        marker.type = Marker.LINE_STRIP  # Or use LINE_LIST for disconnected lines
        marker.action = Marker.ADD

        # Set the scale of the line (width in meters)
        marker.scale.x = 0.05

        # Set the color of the line (RGBA format)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully opaque

        # Add the points to the marker
        for pos in scan:
            point = Point()
            point.x = pos[0]
            point.y = pos[1]
            point.z = 0.0  # Z-axis set to 0 for a 2D shape
            marker.points.append(point)

        # Optionally, close the shape by connecting the last point to the first
        # If you want the shape to be closed like a polygon:
        point = Point()
        point.x = scan[0][0]
        point.y = scan[0][1]
        point.z = 0.0
        marker.points.append(point)

        # Publish the marker
        self.fov_viz_pub.publish(marker)


    def control_loop(self, pose, target_pose, target_velocity, ref_vel, scan, map_info=None):
        if (scan is not None and target_pose is not None and target_velocity is not None and
                pose is not None and ref_vel is not None):
            self.data_record_ts = time.time()
            print("Target pose: ", target_pose, "Target velocity: ", target_velocity, "Agent pose: ", pose,
                  "Planner reference control: ", ref_vel)

            if isinstance(self.solver, DRCController):
                scan[scan == np.inf] = radius
                fov = self.get_fov_from_lidar(scan, pose, stride=2)
                samples = self.solver.get_drc_samples(self.scan, n_samples=1)
                if self.use_sampled_cbf_visibility:
                    visibility_samples = self.solver.get_visibility_cbf_samples(fov, pose, target_pose, target_velocity, map_info)
                else:
                    visibility_samples = (None, None, None)

                u, raytraced_fov, target_sdf = self.solver.solve_drc(self.pose, self.target_pose, self.target_velocity, self.ref_vel, *samples,
                                          fov, map_info, self.use_sampled_cbf_visibility, *visibility_samples)

            elif isinstance(self.solver, BasicController):
                u, raytraced_fov, target_sdf = self.solver.solvecvx(pose, target_pose, target_velocity, np.array([0.2, 0]), scan, map_info)

            else:
                print("Invalid controller requested. Optimal control not found.")
                return

            ################################## save data to calculate metrics ##################################
            self.sdf_record.append(target_sdf)
            self.raytraced_fov_record.append(raytraced_fov)
            self.pose_record.append(pose)
            self.target_pose_record.append(target_pose)
            self.target_velocity_record.append(target_velocity)
            self.ref_control_record.append(ref_vel)
            self.lidar_min_dist_record.append(self.solver.lidar_min_dist)

            # prepare to pickle
            data_pkl = {
                "timestamp": self.data_record_ts,
                "sdf_record": self.sdf_record,
                "raytraced_fov_record": self.raytraced_fov_record,
                "robot_pose_record": self.pose_record,
                "target_pose_record": self.target_pose_record, 
                "target_velocity_record": self.target_velocity_record,
                "reference_control_record": self.ref_control_record,
                "lidar_min_dist_record": self.lidar_min_dist_record, # subtracts robot radius
                "rt_radius": self.solver.raytracing_radius,
                "raytracing_res": self.solver.raytracing_res,
                "rt_fov_range_angle": self.solver.raytracing_fov_range_angle
            }
            dt_object = datetime.fromtimestamp(self.data_record_ts)
            readable_time = dt_object.strftime('%Y-%m-%d_%H:%M:%S')
            self.data_pkl = data_pkl
            self.data_output_time_str = readable_time

            # publish the control
            if u is not None:
                # print(u)
                vel_msg = Twist()
                vel_msg.linear.x = float(u[0])
                vel_msg.angular.z = float(u[1])
                self.cmd_vel_pub.publish(vel_msg)
            self.publish_viz(raytraced_fov)
        self.solver_ongoing = False



if __name__ == '__main__':
    try:
        node = MyControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

