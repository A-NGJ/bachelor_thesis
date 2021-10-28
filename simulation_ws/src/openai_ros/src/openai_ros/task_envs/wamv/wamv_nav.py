import os

import numpy

# pylint: disable=import-error
import rospy
from gym import spaces

from openai_ros.openai_ros_common import ROSLauncher
from openai_ros.robot_envs import wamv_env
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.task_envs.wamv.utils import Actions
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from tf.transformations import euler_from_quaternion


class WamvNavTwoSetsBuoysEnv(wamv_env.WamvEnv):
    def __init__(self):
        self.ros_ws_abspath = None
        self.rospackage_name = None
        self.launch_file_name = None

        self.cumulated_reward = None
        self.previous_distance_from_des_point = None

        self._load_config()

        if not self.ros_ws_abspath:
            raise ValueError('ros_abspath in the yaml config file not set.')

        if not os.path.exists(self.ros_ws_abspath):
            raise FileNotFoundError('The Simulation ROS Workspace path'
                                   f'{self.ros_ws_abspath} does not exist')

        ROSLauncher(rospackage_name=self.rospackage_name,
                    launch_file_name=self.launch_file_name,
                    ros_ws_abspath=self.ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/wamv/config",
                               yaml_file_name="wamv_nav.yaml")

        super().__init__()

        rospy.logdebug(f'Start {type(self).__name__} INIT...')
        number_actions = rospy.get_param('/wamv/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        # Actions and Observations
        self.propeller_high_speed = rospy.get_param('/wamv/propeller_high_speed')
        self.propeller_low_speed = rospy.get_param('/wamv/propeller_low_speed')
        self.max_angular_speed = rospy.get_param('/wamv/max_angular_speed')
        self.max_distance_from_des_point = rospy.get_param('/wamv/max_distance_from_des_point')

        # Get Desired Point to Get
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/wamv/desired_point/x")
        self.desired_point.y = rospy.get_param("/wamv/desired_point/y")
        self.desired_point.z = rospy.get_param("/wamv/desired_point/z")
        self.desired_point_epsilon = rospy.get_param("/wamv/desired_point_epsilon")

        self.work_space_x_max = rospy.get_param("/wamv/work_space/x_max")
        self.work_space_x_min = rospy.get_param("/wamv/work_space/x_min")
        self.work_space_y_max = rospy.get_param("/wamv/work_space/y_max")
        self.work_space_y_min = rospy.get_param("/wamv/work_space/y_min")

        self.dec_obs = rospy.get_param("/wamv/number_decimals_precision_obs")

        high = numpy.array([
            self.work_space_x_max,
            self.work_space_y_max,
            1.57,
            1.57,
            3.14,
            self.propeller_high_speed,
            self.propeller_high_speed,
            self.max_angular_speed,
            self.max_distance_from_des_point
        ])

        low = numpy.array([
            self.work_space_x_min,
            self.work_space_y_min,
            -1*1.57,
            -1*1.57,
            -1*3.14,
            -1*self.propeller_high_speed,
            -1*self.propeller_high_speed,
            -1*self.max_angular_speed,
            0.0
        ])

        self.observation_space = spaces.Box(low, high)

        rospy.logdebug(f'Action spaces type: {self.action_space}')
        rospy.logdebug(f'Observation spaces type: {self.observation_space}')

        self.done_reward =rospy.get_param("/wamv/done_reward")
        self.closer_to_point_reward = rospy.get_param("/wamv/closer_to_point_reward")

        self.cumulated_steps = 0.0

        rospy.logdebug(f'END {type(self).__name__} INIT...')


    def _load_config(self):
        self.ros_ws_abspath = rospy.get_param('/wamv/ros_ws_abspath', None)
        self.rospackage_name = rospy.get_param('/wamv/rospackage_name', None)
        self.launch_file_name = rospy.get_param('/wamv/launch_file_name', None)


    def _set_init_pose(self):
        right_propeller_speed = 0.0
        left_propeller_speed = 0.0
        self.set_propellers_speed(
            right_propeller_speed,
            left_propeller_speed,
            time_sleep=1.0
        )

        return True


    def _init_env_variables(self):
        self.cumulated_reward = 0.0
        # We get the initial pose to mesure the distance from the desired point.
        odom = self.odom()
        current_position = Vector3()
        current_position.x = odom.pose.pose.position.x
        current_position.y = odom.pose.pose.position.y
        self.previous_distance_from_des_point =\
            self.get_distance_from_desired_point(current_position)


    def _set_action(self, action):
        """
        It sets the joints of wamv based on the action integer given
        based on the action number given.
        :param action: The action integer that sets what movement to do next.
        """

        rospy.logdebug(f'Start Set Action {action}')

        right_propeller_speed = 0.0
        left_propeller_speed = 0.0

        if action == Actions.FORWARD.value: # Go Forwards
            right_propeller_speed = self.propeller_high_speed
            left_propeller_speed = self.propeller_high_speed
        elif action == Actions.BACKWARD.value: # Go BackWards
            right_propeller_speed = -1*self.propeller_high_speed
            left_propeller_speed = -1*self.propeller_high_speed
        elif action == Actions.LEFT.value: # Turn Left
            right_propeller_speed = self.propeller_high_speed
            left_propeller_speed = -1*self.propeller_high_speed
        elif action == Actions.RIGHT.value: # Turn Right
            right_propeller_speed = -1*self.propeller_high_speed
            left_propeller_speed = self.propeller_high_speed
        else:
            raise ValueError(f'Invalid action: {action}')

        self.set_propellers_speed(
            right_propeller_speed,
            left_propeller_speed,
            time_sleep=1.0
        )

        rospy.logdebug(f'END Set Action {action}')

    def _get_obs(self):
        rospy.logdebug('Start Get Observation')

        odom = self.odom()
        image_right = self.image_right()
        image_left = self.image_left()
        image_front = self.image_front()
        base_position = odom.pose.pose.position
        base_orientation_quat = odom.pose.pose.orientation
        base_roll, base_pitch, base_yaw = self.get_orientation_euler(base_orientation_quat)
        base_speed_linear = odom.twist.twist.linear
        base_speed_angular_yaw = odom.twist.twist.angular.z

        distance_from_desired_point = self.get_distance_from_desired_point(base_position)

        observation = []
        observation.append(round(base_position.x, self.dec_obs))
        observation.append(round(base_position.y, self.dec_obs))

        observation.append(round(base_roll, self.dec_obs))
        observation.append(round(base_pitch, self.dec_obs))
        observation.append(round(base_yaw, self.dec_obs))

        observation.append(round(base_speed_linear.x, self.dec_obs))
        observation.append(round(base_speed_linear.y, self.dec_obs))

        observation.append(round(base_speed_angular_yaw, self.dec_obs))

        observation.append(round(distance_from_desired_point, self.dec_obs))
        observation.append(image_right)
        observation.append(image_left)
        observation.append(image_front)

        return observation


    def _is_done(self, observations):

        current_position = Vector3()
        current_position.x = observations[0]
        current_position.y = observations[1]

        is_inside_corridor = self.is_inside_workspace(current_position)
        has_reached_des_point = self.is_in_desired_position(
                                        current_position,
                                        self.desired_point_epsilon
                                    )

        done = not(is_inside_corridor) or has_reached_des_point

        return done

    def _compute_reward(self, observations, done):
        current_position = Point()
        current_position.x = observations[0]
        current_position.y = observations[1]

        distance_from_des_point = self.get_distance_from_desired_point(current_position)
        distance_difference = distance_from_des_point - self.previous_distance_from_des_point

        if not done:
            # If there has been a decrease in the distance to the desired point, we reward it
            if distance_difference < 0.0:
                reward = self.closer_to_point_reward
            else:
                reward = -1*self.closer_to_point_reward
        else:
            if self.is_in_desired_position(current_position, self.desired_point_epsilon):
                reward = self.done_reward
            else:
                reward = -1*self.done_reward

        self.previous_distance_from_des_point = distance_from_des_point

        rospy.logdebug('reward={reward}')
        self.cumulated_reward += reward
        rospy.logdebug(f'Cumulated_reward={self.cumulated_reward}')
        self.cumulated_steps += 1
        rospy.logdebug(f'Cumulated_steps={self.cumulated_steps}')

        return reward


    def is_in_desired_position(self, current_position, epsilon=0.05):
        is_in_desired_pos = False

        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = x_pos_minus < x_current <= x_pos_plus
        y_pos_are_close = y_pos_minus < y_current <= y_pos_plus
        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        rospy.logdebug('#'*20)
        rospy.logdebug(f'current_position:\n{current_position}')
        rospy.logdebug(f'x_pos_plus: {x_pos_plus}, x_pos_minus: {x_pos_minus}')
        rospy.logdebug(f'y_pos_plus: {y_pos_plus}, y_pos_minus: {y_pos_minus}')
        rospy.logdebug(f'x_pos_are_close: {x_pos_are_close}')
        rospy.logdebug(f'x_pos_are_close: {y_pos_are_close}')
        rospy.logdebug(f'is_in_desired_pos {is_in_desired_pos}')
        rospy.logdebug('#'*20)

        return is_in_desired_pos

    def get_distance_from_desired_point(self, current_position):
        distance = self.get_distance_from_point(
            current_position,
            self.desired_point
        )

        return distance

    def get_distance_from_point(self, pstart, p_end):
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw

    def is_inside_workspace(self,current_position):
        is_inside = False

        rospy.loginfo('#'*20)
        rospy.loginfo(f'Current position: {current_position}')
        rospy.loginfo(f'work_space_x_max: {self.work_space_x_max}, '
                      f'work_space_x_min: {self.work_space_x_min}')
        rospy.loginfo(f'work_space_y_max: {self.work_space_y_max}, '
                      f'work_space_y_min: {self.work_space_y_min}')
        rospy.loginfo('#'*20)

        if current_position.x > self.work_space_x_min and\
            current_position.x <= self.work_space_x_max:
            if current_position.y > self.work_space_y_min and\
                current_position.y <= self.work_space_y_max:
                is_inside = True

        return is_inside
