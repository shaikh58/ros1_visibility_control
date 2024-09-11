'''A script for evaluating the Car Model.
'''
import json
import threading
from collections import deque

import time
from functools import lru_cache

import cv2
import skimage.io
import numpy as np
import torch

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import control as oc

    from ompl import util as ou
except ImportError:
    raise ImportError("Container does not have OMPL installed")
from math import sin, cos, tan
from utils.utils_planner import map2world, world2map, SDF_RT, polygon_SDF, normalize_angle, boundaries, DDA
from robot.planner.transformer import Models
from os import path as osp
from robot.planner.eval_model import get_patch
from env.config import *
from env.map.map import Map

DEMO = False
if DEMO:
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_out = cv2.VideoWriter('../data/MPT-SST.mp4', fourcc, 5.0, (976, 780))

display_buffer = deque()
sim_map = Map().map
# sim_map = skimage.io.imread(map_file_path, as_gray=True).astype(np.uint8)
robot_radius = 0.2
carLength = 0.3
cam_alpha = np.pi / 3
cam_radius = 100

map_origin = np.array([[490], [488], [np.pi / 2]])
map_ratio = 2

# Planning parameters
space = ob.SE2StateSpace()
# Set the bounds
bounds = ob.RealVectorBounds(2)
bounds.setLow(-200)
bounds.setHigh(200)
space.setBounds(bounds)

cspace = oc.RealVectorControlSpace(space, 2)
cbounds = ob.RealVectorBounds(2)
cbounds.setLow(0, v_lb)
cbounds.setHigh(0, v_ub)
cbounds.setLow(1, omega_lb)
cbounds.setHigh(1, omega_ub)
cspace.setBounds(cbounds)
ss = oc.SimpleSetup(cspace)
si = ob.SpaceInformation(space)

ou.setLogLevel(ou.LOG_WARN)
seed = 42


def kinematicCarODE(q, u, qdot):
    theta = q[2]

    qdot[0] = u[0] * cos(theta)
    qdot[1] = u[0] * sin(theta)
    qdot[2] = u[0] * tan(u[1]) / carLength


def kinematicUnicycleODE(x, u, xdot):
    xdot[0] = u[0] * cos(x[2])
    xdot[1] = u[0] * sin(x[2])
    xdot[2] = u[1]


# class CustomGoal(ob.Goal):
#     def __init__(self, space, target_condition, goal_state, sim_map):
#         super().__init__(space)
#         self.target_condition = target_condition
#         self.goal_state = goal_state.get()
#         self.sim_map = sim_map
#
#     def isSatisfied(self, state):
#         # Check if the goal condition is met
#         satisfied = self.target_condition(state, self.goal_state, self.sim_map)
#         return satisfied


class CustomGoal(ob.Goal):
    def __init__(self, space, target_condition, goal_state, sim_map):
        super().__init__(space)
        self.target_condition = target_condition
        self.goal_state = goal_state.get()
        self.sim_map = sim_map
        target_w = np.array([self.goal_state.getX(), self.goal_state.getY(), self.goal_state.getYaw()]).reshape((3, 1))
        target_d = world2map(map_origin, map_ratio, target_w.reshape((3, 1))).squeeze()
        self.polygon = SDF_RT_circular(target_d, 100, 200, sim_map)
        self.polygon = (map2world(map_origin, map_ratio, self.polygon.T)[0:2, :]).T

    def isSatisfied(self, state):
        # Check if the goal condition is met
        satisfied = self.isGoalConditionMet_TgtPer(state, self.goal_state, self.sim_map)
        return satisfied

    def isGoalConditionMet_TgtPer(self, state, target, sim_map):
        same_pos = bool(np.sqrt((state.getX() - target.getX()) ** 2 + (state.getY() - target.getY()) ** 2) < 80)
        in_fov, face_to_target = False, False
        if same_pos:
            robot_w = np.array([state.getX(), state.getY(), state.getYaw()]).reshape((3, 1))
            in_fov = polygon_SDF(self.polygon, robot_w[0:2].T) < -5
            face_to_target = np.abs(np.arctan2(target.getY() - state.getY(),
                                               target.getX() - state.getX()) - state.getYaw()) < 0.5 * np.pi / 3

            # if in_fov and face_to_target: print(getSDF(state, target, sim_map))
        return bool(in_fov and face_to_target)


def SDF_RT_circular(robot_pose, radius, RT_res, grid):
    global global_map
    global_map = grid
    pts = raytracing_circular(robot_pose, 2 * np.pi, radius, RT_res)
    return np.array(pts)


def raytracing_circular(robot_pose, fov, radius, RT_res):
    x0, y0, theta = robot_pose
    out_1, out_2 = boundaries(robot_pose, fov, radius)
    y_mid = [y0 + radius * np.sin(theta - 0.5 * fov + i * fov / RT_res) for i in range(RT_res + 1)]
    x_mid = [x0 + radius * np.cos(theta - 0.5 * fov + i * fov / RT_res) for i in range(RT_res + 1)]
    # y_mid = np.linspace(out_1[1], out_2[1], RT_res)
    # x_mid = np.linspace(out_1[0], out_2[0], RT_res)
    pts = []
    for i in range(len(x_mid)):
        xx, yy = DDA(int(x0), int(y0), int(x_mid[i]), int(y_mid[i]))
        if not pts or (yy != pts[-1][1] or xx != pts[-1][0]):
            pts.append([xx, yy])
    return pts


@lru_cache(None)
def DDA(x0, y0, x1, y1):
    # find absolute differences
    dx = x1 - x0
    dy = y1 - y0

    # find maximum difference
    steps = int(max(abs(dx), abs(dy)))

    # calculate the increment in x and y
    x_inc = dx / steps
    y_inc = dy / steps

    # start with 1st point
    x = float(x0)
    y = float(y0)

    for i in range(steps):
        if 0 < int(x) < len(global_map) and 0 < int(y) < len(global_map[0]):
            if global_map[int(x), int(y)] == 0:
                break
        x = x + x_inc
        y = y + y_inc
    return int(x) + 1, int(y) + 1


class MotionCost(ob.OptimizationObjective):
    def __init__(self, si):
        super(MotionCost, self).__init__(si)

    def motionCost(self, s1, s2):
        # This method can be used to define the cost of a motion
        # For simplicity, let's just use Euclidean distance as an example
        return ob.Cost(np.sqrt((s1.getX() - s2.getX()) ** 2 + (s1.getY() - s2.getY()) ** 2))


class StateCost(ob.StateCostIntegralObjective):
    def __init__(self, si, goal):
        super(StateCost, self).__init__(si)
        self.goal = goal()

    def stateCost(self, state):
        return ob.Cost(self.CombinedCost(state, self.goal))
        # return ob.Cost(0)

    def DistanceCost(self, state, goal):
        return (state.getX() - goal.getX()) ** 2 + (state.getY() - goal.getY()) ** 2

    def SDFCost(self, state, goal):
        return getSDF(state, goal, sim_map)

    def CombinedCost(self, state, goal, th=50):
        distance = np.sqrt(self.DistanceCost(state, goal))
        if distance < th:
            return self.SDFCost(state, goal)
        return distance

    def SigmoidCost(self, state, goal, th=50, sigma=1):
        distance = np.sqrt(self.DistanceCost(state, goal))
        SDF = self.SDFCost(state, goal)
        w = 1 / (1 + np.exp(-sigma * (distance - th)))
        return w * distance + (1 - w) * SDF


def getSDF(s1, s2, sim_map):
    robot_w = np.array([s1.getX(), s1.getY(), s1.getYaw()]).reshape((3, 1))
    target_w = np.array([s2.getX(), s2.getY(), s2.getYaw()]).reshape((3, 1))
    robot_m = world2map(map_origin, map_ratio, robot_w)
    visibility_m = SDF_RT(robot_m, cam_alpha, cam_radius, 50, sim_map, 0)
    visibility_w = (map2world(map_origin, map_ratio, visibility_m.T)[0:2, :]).T
    signed_distance = polygon_SDF(visibility_w, target_w[0:2].T)
    return signed_distance


def isGoalConditionMet(state, target, sim_map):
    same_pos = np.sqrt((state.getX() - target.getX()) ** 2 + (state.getY() - target.getY()) ** 2) < cam_radius
    same_yaw = abs(state.getYaw() - target.getYaw()) < 0.05
    in_fov = False
    if same_pos and True:
        in_fov = getSDF(state, target, sim_map) < -5
    return in_fov


class ValidityChecker(ob.StateValidityChecker):
    '''A class to check if an obstacle is in collision or not.
    '''

    def __init__(self, si, CurMap, mpt_mask, res=0.05, robot_radius=robot_radius, ):
        '''
        Initialize the class object, with the current map and mask generated
        from the transformer model.
        :param si: an object of type ompl.base.SpaceInformation
        :param CurMap: A np.array with the current map.
        :param MapMask: Areas of the map to be masked.
        '''
        super().__init__(si)
        self.size = CurMap.shape
        self.map = CurMap
        self.map_origin = np.array([[len(self.map) // 2], [len(self.map[0]) // 2], [np.pi / 2]])
        self.patch_map = mpt_mask

    def isValid(self, state):
        '''
        Check if the given state is valid.
        :param state: An ob.State object to be checked.
        :returns bool: True if the state is valid.
        '''
        x, y = state.getX(), state.getY()
        pix_dim = world2map(self.map_origin, 2, np.array([[x], [y], [0]])).astype(np.int16)
        if pix_dim[0] < 0 or pix_dim[0] >= self.size[0] or pix_dim[1] < 0 or pix_dim[1] >= self.size[1]:
            return False
        return bool(self.map[pix_dim[0], pix_dim[1]][0]) and bool(self.patch_map[pix_dim[0], pix_dim[1]][0])


def getBalancedObjective(si, goal):
    cost_1 = MotionCost(si)
    cost_2 = StateCost(si, goal)

    opt = ob.MultiOptimizationObjective(si)
    opt.addObjective(cost_1, 1.0)
    opt.addObjective(cost_2, 1.0)

    return opt


def get_path(start, goal, sim_map, mpt_mask, step_time=0.1, max_time=300.):
    '''
    Plan a path given the start, goal and patch_map.
    :param start: The SE(2) co-ordinate of start position
    :param goal: The SE(2) co-ordinate of goal position
    :param step_time: The time step for the planner.
    :param max_time: The maximum time to plan
    :param exp: If true, the planner switches between exploration and exploitation.
    returns tuple: Returns path array, time of solution, number of vertices, success.
    '''
    # Tried importance sampling, but seems like it makes not much improvement
    # over rejection sampling.

    StartState = ob.State(space)
    StartState().setX(start[0])
    StartState().setY(start[1])
    StartState().setYaw(start[2])

    GoalState = ob.State(space)
    GoalState().setX(goal[0])
    GoalState().setY(goal[1])
    GoalState().setYaw(goal[2])

    # setup validity checker
    ValidityCheckerObj = ValidityChecker(si, sim_map, mpt_mask)

    ss.setStateValidityChecker(ValidityCheckerObj)

    # ss.setOptimizationObjective(getBalancedObjective(si, GoalState))
    ss.setOptimizationObjective(MotionCost(si))
    ss.clear()
    ss.clearStartStates()
    ss.addStartState(StartState)
    ss.setGoal(CustomGoal(si, isGoalConditionMet, GoalState, sim_map))

    ode = oc.ODE(kinematicUnicycleODE)
    odeSolver = oc.ODEBasicSolver(ss.getSpaceInformation(), ode)
    propagator = oc.ODESolver.getStatePropagator(odeSolver)
    ss.setStatePropagator(propagator)
    ss.getSpaceInformation().setPropagationStepSize(0.2)
    ss.getSpaceInformation().setMinMaxControlDuration(5, 5)

    # planner = oc.SST(ss.getSpaceInformation())
    planner = oc.RRT(ss.getSpaceInformation())

    ss.setPlanner(planner)
    time = step_time
    ss.solve(step_time)
    solved = False

    path = []
    controls = []
    while time < max_time:
        ss.solve(step_time)
        time += step_time
        if ss.haveExactSolutionPath():
            print("Found Solution")
            solved = True
            path = np.array([[ss.getSolutionPath().getState(i).getX(),
                              ss.getSolutionPath().getState(i).getY(),
                              ss.getSolutionPath().getState(i).getYaw()]
                             for i in range(ss.getSolutionPath().getStateCount())])
            controls = np.array([[ss.getSolutionPath().getControl(i)[0],
                                  ss.getSolutionPath().getControl(i)[1]]
                                 for i in range(ss.getSolutionPath().getControlCount())])
            # cost = ss.getSolutionPath().asGeometric().cost(getBalancedObjective(si, GoalState)).value()
            # print("Cost: ", cost)
            break
    plannerData = ob.PlannerData(si)
    planner.getPlannerData(plannerData)
    numVertices = plannerData.numVertices()

    ss.clear()
    ss.clearStartStates()
    return controls, path, time, numVertices, solved


def cv_render(rbt, tgt, input_map=sim_map, traj=None, patch=None):
    map_display = input_map.astype(np.uint8)
    # map_display = (sim_map * 255 / np.max(sim_map)).astype(np.uint8)
    tgt_d = world2map(map_origin, map_ratio, tgt.reshape((3, 1))).squeeze()
    rbt_d = world2map(map_origin, map_ratio, rbt.reshape((3, 1))).squeeze()
    rt_visible = SDF_RT(rbt_d, cam_alpha, cam_radius, 50, map_display)
    # rt_visible = SDF_RT_circular(rbt_d, cam_radius, 300, map_display)
    visible_region = (map2world(map_origin, map_ratio, rt_visible.T)[0:2, :]).T
    SDF_center = polygon_SDF(visible_region, tgt[0:2])
    if np.ndim(map_display) == 2:
        visible_map = cv2.cvtColor(map_display, cv2.COLOR_GRAY2BGR)
    else:
        visible_map = map_display
    rt_visible = np.flip(rt_visible.astype(np.int32))
    visible_map = cv2.polylines(visible_map, [rt_visible.reshape(-1, 1, 2)], True,
                                (50, 255 if SDF_center < 0 else 128, 0), 2)
    visible_map[int(tgt_d[0]) - 2:int(tgt_d[0]) + 2,
    int(tgt_d[1]) - 2:int(tgt_d[1]) + 2] = np.array([0, 0, 255])
    visible_map[int(rbt_d[0]) - 2:int(rbt_d[0]) + 2,
    int(rbt_d[1]) - 2:int(rbt_d[1]) + 2] = np.array([255, 0, 0])
    if traj is not None:
        traj_cv = traj[:, ::-1].reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(visible_map, [traj_cv], False, (240, 100, 0), 1)
        visible_map[traj[:, 0], traj[:, 1]] = np.array([0, 180, 200])

        l = np.linalg.norm(traj[-1] - traj[-2]) + 1
        arrow_head = np.array([20 * (traj[-1, 0] - traj[-2, 0]) / l + traj[-1, 0],
                               20 * (traj[-1, 1] - traj[-2, 1]) / l + traj[-1, 1]])
        cv2.arrowedLine(visible_map, traj[-1, ::-1], arrow_head.astype(np.int32)[::-1], (240, 100, 0), 1, tipLength=0.2)
    # visible_map = visible_map[100:-100, :, :]
    if DEMO: video_out.write(visible_map)
    display_buffer.append(visible_map)
    cv2.imshow('debugging', visible_map)
    # if SHOW_FPS: print(1 / (fps_timer - time.time()))
    # fps_timer = time.time()
    key = cv2.waitKey(1) & 0xFF
    return visible_map


def load_model(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelFolder = model_path
    modelFile = osp.join(modelFolder, f'model_params.json')
    model_param = json.load(open(modelFile))
    transformer = Models.Transformer(**model_param)
    transformer.to(device)
    # checkpoint = torch.load(osp.join(modelFolder, f'model_weights.pkl'), map_location=torch.device('cpu'))
    checkpoint = torch.load(osp.join(modelFolder, f'model_weights.pkl'))
    transformer.load_state_dict(checkpoint['state_dict'])
    transformer.eval()
    return transformer


def planner_single(start, goal, if_mpt=False, model=None):
    # sim_map = Map().map
    # sim_map = skimage.io.imread(map_file_path, as_gray=True).astype(np.uint8)
    # sim_map = np.ones_like(sim_map)
    cols, rows = sim_map.shape
    small_map = cv2.resize(sim_map, (cols // 2, rows // 2))
    small_map = (small_map / np.max(small_map)).astype(np.uint8)
    start = start.reshape((3,))
    goal = goal.reshape((3,))
    start_m = world2map(map_origin, map_ratio, start).squeeze()[0:2]
    goal_m = world2map(map_origin, map_ratio, goal).squeeze()[0:2]
    time_inference = time.time()
    if if_mpt:
        patch_map, _ = get_patch(model,
                                 (0.5 * start_m).astype(np.int16)[::-1],
                                 (0.5 * goal_m).astype(np.int16)[::-1],
                                 small_map)
        fs_patch_map = cv2.resize(patch_map, sim_map.shape[::-1]) * 255
    else:
        fs_patch_map = np.ones_like(sim_map)
    # print('MPT takes: '+str(round(time.time()-time_inference, 5))+'s for referencing')
    actions, trajectory, planner_time, numVer, success = get_path(start, goal, sim_map, fs_patch_map, max_time=planner_max_time)
    if len(actions) == 0:
        actions = None
    traj_m = None
    if success:
        traj_m = world2map(map_origin, map_ratio, trajectory.T)[0:2].T.astype(np.int16)
        # key = cv_render(start, goal, sim_map, traj_m)
        start = trajectory[1]
        start[2] = normalize_angle(start[2])
    else:
        trajectory = start.reshape((1, 3))
        # key = cv_render(start, goal, sim_map, traj_m)
    return start, trajectory, success, actions


def planner_main():
    sim_map = skimage.io.imread('./map_solid.png', as_gray=True).astype(np.uint8)
    small_map = skimage.io.imread('map_solid_ds.png', as_gray=True)
    goal_traj = np.load('Lsj.npz')['arr_0']
    start = np.array([-100., -100., 0.])

    ref_input = np.array([10, 0])

    log = []
    model = load_model('../models/final_models/car_robot')
    for i in range(200, 6000, 20):
        goal = goal_traj[i, :3].reshape((3,))  # todo: replace i with i+k, k is param from mpc

        patch_map, predProb = get_patch(model,
                                        (0.5 * world2map(map_origin, map_ratio, start).squeeze()[0:2]).astype(np.int16),
                                        (0.5 * world2map(map_origin, map_ratio, goal).squeeze()[0:2]).astype(np.int16),
                                        small_map)
        fs_patch_map = cv2.resize(patch_map, sim_map.shape) * 255
        # fs_patch_map = np.ones_like(sim_map)
        # cv2.imwrite('./results/sampling_mask.png', fs_patch_map)

        robot_actions, trajectory, time, numVer, success = get_path(start, goal, sim_map, fs_patch_map, max_time=3)
        traj_m = None
        if success:
            log.append((np.vstack((start, goal)), robot_actions, trajectory))
            traj_m = world2map(map_origin, map_ratio, trajectory.T)[0:2].T.astype(np.int16)
            key = cv_render(start, goal, sim_map, traj_m)
            start = trajectory[1]
        else:
            log.append((np.vstack((start, goal)), [], []))
            key = cv_render(start, goal, sim_map, traj_m)
            # start = start + ref_input[0] * np.array([np.cos(start[2]), np.sin(start[2]), 0])
        start[2] = normalize_angle(start[2])
        if key == 27:
            break
    # import pickle
    # with open(r"logs.pickle", "wb") as output_file:
    #     pickle.dump(log, output_file)
    if DEMO: video_out.release()


def load_map(path, map_ratio):
    m = skimage.io.imread(path, as_gray=True).astype(np.uint8)
    map_origin = np.array([[len(m) // map_ratio, len(m[0]) // map_ratio, np.pi / 2]])
    return m, map_origin, map_ratio


def display_sim(buffer):
    while True:
        if len(buffer) > 0:
            cv2.imshow('sim', buffer.popleft())
        if cv2.waitKey(100) & 0xFF == 27:
            break
    return 0


class Planner:
    def __init__(self):
        self.mpt_model = load_model('../robot/planner/models/final_models/point_robot')
        self.start = np.array([-100., -100., 0.1])
        self.goal_traj = np.load('../robot/planner/Lsj.npz')['arr_0']
        self.k = 200
        self.step = 5
        self.last_success = [[],[]]
        self.last_success_index = 1

    def get_next_state(self, start):
        goal = self.goal_traj[self.k, :3].reshape((3,))
        display_goal = self.goal_traj[self.k + 50, :3].reshape((3,))
        self.k += self.step
        self.start = start
        self.start, trajectory, success, actions = planner_single(self.start, goal, if_mpt=True, model=self.mpt_model)
        if success:
            self.last_success = [trajectory, actions]
            self.last_success_index = 1
        elif self.last_success_index < len(self.last_success[0]):
            self.start = self.last_success[0][self.last_success_index]
            self.start[2] = normalize_angle(self.start[2])
            actions = self.last_success[1][self.last_success_index:]
            self.last_success_index += 1
        return self.start, display_goal, trajectory, actions

    def get_next_state_cbf(self, start, goal):
        goal = goal.reshape((3,))
        # display_goal = self.goal_traj[self.k + 50, :3].reshape((3,))
        self.k += self.step
        self.start = start
        self.start, trajectory, success, actions = planner_single(self.start, goal, if_mpt=True, model=self.mpt_model)
        if success:
            self.last_success = [trajectory, actions]
            self.last_success_index = 1
        elif self.last_success_index < len(self.last_success[0]):
            self.start = self.last_success[0][self.last_success_index]
            self.start[2] = normalize_angle(self.start[2])
            actions = self.last_success[1][self.last_success_index:]
            self.last_success_index += 1
        return trajectory, actions


def main():
    mpt_model = load_model('models/final_models/point_robot')
    display_thread = threading.Thread(target=display_sim, args=(display_buffer,), daemon=True)
    # display_thread.start()

    start = np.array([-100., -100., 0.1])
    i = 200
    goal_traj = np.load('Lsj.npz')['arr_0']
    time_rec = []
    success_rec = []
    traj_rec = np.array([[-100., -100., 0.]])
    display_image = skimage.io.imread('../../env/map/carla_map.png', as_gray=True).astype(np.uint8)[100:-100, :] * 255
    # display_image = (display_image * 255 / np.max(display_image)).astype(np.uint8)
    for k in range(i, i + 1000, 5):
        goal = goal_traj[k, :3].reshape((3,))
        start, trajectory, success, actions = planner_single(start, goal, if_mpt=True, model=mpt_model)
        # time_rec.append(time)
        # success_rec.append(success)
        # if success:
        #     traj_m = world2map(map_origin, map_ratio, trajectory.T)[0:2].T.astype(np.int16)
        # cv_render(start, goal, display_image, traj_m)
    # display_buffer.append(display_image)
    # print(time_rec)
    # print(np.sum(success_rec))
    while True:
        pass


if __name__ == "__main__":
    main()
