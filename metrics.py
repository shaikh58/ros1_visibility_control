import numpy as np 
import pickle
import argparse

parser = argparse.ArgumentParser(description="A script that takes in a filename.")
parser.add_argument("filename", type=str, help="The name of the file to process.")
args = parser.parse_args()
filename = args.filename

with open(f"/home/administrator/visibility_control/experiments/data/{filename}", 'rb') as file:
    data = pickle.load(file)

timestamp_record = np.array(data["timestamp_record"])
optimal_control_record = np.array(data["optimal_control_record"])
sdf_record = np.array(data["sdf_record"])
raytraced_fov_record = data["raytraced_fov_record"]
robot_pose_record = np.array(data["robot_pose_record"])
target_pose_record = np.array(data["target_pose_record"])
target_velocity_record = np.array(data["target_velocity_record"])
reference_control_record = np.array(data["reference_control_record"])
lidar_min_dist_record = np.array(data["lidar_min_dist_record"])
rt_radius = data["rt_radius"]
raytracing_res = data["raytracing_res"]
rt_fov_range_angle = data["rt_fov_range_angle"]

# % of time target is in sdf
frames_in_sdf = np.sum(sdf_record > 0)
pct_time_in_fov = frames_in_sdf / len(sdf_record)
print("% of time target in FoV: ", (pct_time_in_fov*100).round())
# relocalization time for each loss of visibility event
vis_loss_inds = np.where(sdf_record < 0)

# min. distance to obstacle
min_dist_obs = np.min(lidar_min_dist_record)
print("Minimum distance to obstacles along trajectory: ", min_dist_obs)

# average angular velocity
print("Average angular velocity: ", np.mean(np.abs(optimal_control_record[:,1])))