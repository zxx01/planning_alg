import numpy as np
import matplotlib.pyplot as plt
import math
import sys


class RobotConfig:
    """机器人和DWA算法的配置参数类"""

    def __init__(self):
        # 机器人速度参数
        self.v_max = 1.0  # 最大线速度 [m/s]
        self.v_min = -0.5  # 最小线速度 [m/s]
        self.w_max = 40.0 * math.pi / 180.0  # 最大角速度 [rad/s]
        self.w_min = -40.0 * math.pi / 180.0  # 最小角速度 [rad/s]
        
        # 加速度限制
        self.a_vmax = 0.2  # 最大线加速度 [m/ss]
        self.a_wmax = 40.0 * math.pi / 180.0  # 最大角加速度 [rad/ss]
        
        # 采样分辨率
        self.v_sample = 0.01  # 线速度采样分辨率 [m/s]
        self.w_sample = 0.1 * math.pi / 180.0  # 角速度采样分辨率 [rad/s]
        
        # 仿真参数
        self.dt = 0.1  # 离散时间步长 [s]
        self.predict_time = 3.0  # 轨迹推算时间长度 [s]
        
        # 轨迹评价函数权重
        self.to_goal_cost_gain = 1.0  # 目标方向评价权重
        self.obstacle_cost_gain = 1.0  # 障碍物距离评价权重
        self.speed_cost_gain = 1.0  # 速度评价权重
        
        # 碰撞检测参数
        self.robot_radius = 1.0  # 机器人半径 [m]
        self.safe_distance = self.robot_radius + 0.2  # 安全距离阈值
        self.judge_distance = 10  # 无障碍物威胁时的默认距离评价值
        
    def set_scenario(self, obstacles=None, target=None):
        """设置场景中的障碍物和目标点"""
        # 默认障碍物位置 [x(m) y(m), ....]
        self.obstacles = np.array([
            [-1, -1],
            [0, 2],
            [4.0, 2.0],
            [5.0, 4.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [5.0, 9.0],
            [8.0, 9.0],
            [7.0, 9.0],
            [8.0, 10.0],
            [9.0, 11.0],
            [12.0, 13.0],
            [12.0, 12.0],
            [15.0, 15.0],
            [13.0, 13.0],
        ]) if obstacles is None else obstacles
        
        # 默认目标点位置
        self.target = np.array([10, 10]) if target is None else target


class DWA:
    """动态窗口法路径规划算法实现"""
    
    def __init__(self, config):
        """初始化DWA算法
        
        Args:
            config: 包含算法参数的配置类
        """
        # 从配置中获取参数
        self.dt = config.dt
        self.v_min = config.v_min
        self.w_min = config.w_min
        self.v_max = config.v_max
        self.w_max = config.w_max
        self.predict_time = config.predict_time
        self.a_vmax = config.a_vmax
        self.a_wmax = config.a_wmax
        self.v_sample = config.v_sample
        self.w_sample = config.w_sample
        
        # 评价函数权重
        self.to_goal_cost_gain = config.to_goal_cost_gain
        self.obstacle_cost_gain = config.obstacle_cost_gain
        self.speed_cost_gain = config.speed_cost_gain
        
        # 碰撞检测参数
        self.radius = config.robot_radius
        self.safe_distance = config.safe_distance
        self.judge_distance = config.judge_distance

    def plan(self, state, goal, obstacle):
        """滚动窗口算法入口

        Args:
            state (_type_): 机器人当前状态--[x,y,yaw,v,w]
            goal (_type_): 目标点位置，[x,y]

            obstacle (_type_): 障碍物位置，dim:[num_ob,2]

        Returns:
            _type_: 控制量、轨迹（便于绘画）
        """
        control, trajectory = self._evaluate_trajectories(state, goal, obstacle)
        return control, trajectory

    def _calculate_dynamic_window(self, state, obstacles):
        """计算动态窗口（考虑速度限制、加速度限制和障碍物限制）
        
        Args:
            state (np.ndarray): 机器人当前状态
            obstacles (np.ndarray): 障碍物位置数组
            
        Returns:
            np.ndarray: 动态窗口 [v_min, v_max, w_min, w_max]
        """
        # 计算各种限制下的速度窗口
        v_limits = self._calculate_velocity_limits()
        a_limits = self._calculate_acceleration_limits(state[3], state[4])
        o_limits = self._calculate_obstacle_limits(state, obstacles)
        
        # 合并所有限制，取交集
        v_min = max([v_limits[0], a_limits[0], o_limits[0]])
        v_max = min([v_limits[1], a_limits[1], o_limits[1]])
        w_min = max([v_limits[2], a_limits[2], o_limits[2]])
        w_max = min([v_limits[3], a_limits[3], o_limits[3]])
        
        return np.array([v_min, v_max, w_min, w_max])

    def _calculate_velocity_limits(self):
        """计算速度边界限制
        
        Returns:
            np.ndarray: 速度限制窗口 [v_min, v_max, w_min, w_max]
        """
        return np.array([self.v_min, self.v_max, self.w_min, self.w_max])

    def _calculate_acceleration_limits(self, current_v, current_w):
        """计算基于加速度的速度限制
        
        Args:
            current_v (float): 当前线速度
            current_w (float): 当前角速度
            
        Returns:
            np.ndarray: 考虑加速度的速度窗口 [v_min, v_max, w_min, w_max]
        """
        v_min = current_v - self.a_vmax * self.dt
        v_max = current_v + self.a_vmax * self.dt
        w_min = current_w - self.a_wmax * self.dt
        w_max = current_w + self.a_wmax * self.dt
        
        return np.array([v_min, v_max, w_min, w_max])

    def _calculate_obstacle_limits(self, state, obstacles):
        """计算考虑障碍物的速度限制
        
        Args:
            state (np.ndarray): 机器人当前状态
            obstacles (np.ndarray): 障碍物位置数组
            
        Returns:
            np.ndarray: 考虑障碍物的速度窗口 [v_min, v_max, w_min, w_max]
        """
        min_dist = self._calculate_min_distance(state, obstacles)
        
        # 基于到障碍物的距离计算安全速度上限
        v_max = np.sqrt(2 * min_dist * self.a_vmax) if min_dist > 0 else 0
        w_max = np.sqrt(2 * min_dist * self.a_wmax) if min_dist > 0 else 0
        
        return np.array([self.v_min, v_max, self.w_min, w_max])

    def _predict_trajectory(self, state, v, w):
        """预测给定控制输入下的轨迹
        
        Args:
            state (np.ndarray): 初始状态
            v (float): 线速度
            w (float): 角速度
            
        Returns:
            np.ndarray: 预测的轨迹，形状为 [n_steps, 5]
        """
        trajectory = [state.copy()]
        s = state.copy()
        time = 0
        
        # 在预测时间内模拟轨迹
        while time <= self.predict_time:
            s = update_state(s, [v, w], self.dt)
            trajectory.append(s.copy())
            time += self.dt

        return np.array(trajectory)

    def _evaluate_trajectories(self, state, goal, obstacles):
        """评估所有可能的轨迹并选择最优的
        
        Args:
            state (np.ndarray): 当前状态
            goal (np.ndarray): 目标位置
            obstacles (np.ndarray): 障碍物位置
            
        Returns:
            tuple: (最优控制 [v, w], 最优轨迹)
        """
        # 计算动态窗口
        dw = self._calculate_dynamic_window(state, obstacles)
        
        # 初始化最优评价及相关变量
        best_score = -float('inf')
        best_control = [0.0, 0.0]
        best_trajectory = np.array([state])
        
        # 缓存所有评价的和，用于归一化
        heading_costs = []
        distance_costs = []
        velocity_costs = []
        
        # 第一遍扫描，收集评价值
        for v in np.arange(dw[0], dw[1], self.v_sample):
            for w in np.arange(dw[2], dw[3], self.w_sample):
                trajectory = self._predict_trajectory(state, v, w)
                
                heading_cost = self._calculate_heading_cost(trajectory, goal)
                distance_cost = self._calculate_obstacle_cost(trajectory, obstacles)
                velocity_cost = self._calculate_velocity_cost(trajectory)
                
                heading_costs.append(heading_cost)
                distance_costs.append(distance_cost)
                velocity_costs.append(velocity_cost)
        
        # 计算归一化因子（避免除以0）
        sum_heading = sum(heading_costs) if heading_costs else 1
        sum_distance = sum(distance_costs) if distance_costs else 1
        sum_velocity = sum(velocity_costs) if velocity_costs else 1
        
        # 第二遍扫描，评估归一化后的总得分
        for v in np.arange(dw[0], dw[1], self.v_sample):
            for w in np.arange(dw[2], dw[3], self.w_sample):
                trajectory = self._predict_trajectory(state, v, w)
                
                heading_cost = self._calculate_heading_cost(trajectory, goal) / sum_heading
                distance_cost = self._calculate_obstacle_cost(trajectory, obstacles) / sum_distance
                velocity_cost = self._calculate_velocity_cost(trajectory) / sum_velocity
                
                # 计算加权总分
                score = (self.to_goal_cost_gain * heading_cost + 
                         self.obstacle_cost_gain * distance_cost + 
                         self.speed_cost_gain * velocity_cost)
                
                # 更新最优解
                if score > best_score:
                    best_score = score
                    best_control = [v, w]
                    best_trajectory = trajectory
        
        return best_control, best_trajectory

    def _calculate_min_distance(self, state, obstacles):
        """计算当前位置到最近障碍物的距离
        
        Args:
            state (np.ndarray): 当前状态
            obstacles (np.ndarray): 障碍物位置
            
        Returns:
            float: 到最近障碍物的距离
        """
        if obstacles.size == 0:
            return float('inf')
            
        # 计算到所有障碍物的距离
        dx = state[0] - obstacles[:, 0]
        dy = state[1] - obstacles[:, 1]
        distances = np.hypot(dx, dy)
        
        return np.min(distances)

    def _calculate_obstacle_cost(self, trajectory, obstacle):
        """计算轨迹与障碍物的最小距离评价
        
        Args:
            trajectory (np.ndarray): 预测的轨迹
            obstacles (np.ndarray): 障碍物位置
            
        Returns:
            float: 障碍物距离评价值（越大越好）
        """
        ox = obstacle[:, 0]
        oy = obstacle[:, 1]
        
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r = np.hypot(dx, dy)
        
        min_dist = np.min(r)
        
        return min_dist if min_dist < self.safe_distance else self.judge_distance
        

    def _calculate_heading_cost(self, trajectory, goal):
        """计算轨迹终点朝向与目标方向的一致性评价
        
        Args:
            trajectory (np.ndarray): 预测的轨迹
            goal (np.ndarray): 目标位置
            
        Returns:
            float: 方向评价值（越大越好）
        """
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = math.pi - abs(cost_angle)

        return cost

    def _calculate_velocity_cost(self, trajectory):
        """计算速度评价（鼓励更快的前进速度）
        
        Args:
            trajectory (np.ndarray): 预测的轨迹
            
        Returns:
            float: 速度评价值（越大越好）
        """
        return trajectory[-1, 3]


def update_state(state, control, dt):
    """更新机器人状态的运动学模型
    
    Args:
        state (np.ndarray): 当前状态 [x, y, yaw, v, w]
        control (list): 控制量 [v, w]
        dt (float): 时间步长
        
    Returns:
        np.ndarray: 更新后的状态
    """
    new_state = state.copy()
    v, w = control
    
    # 更新位置和朝向
    new_state[0] += v * math.cos(state[2]) * dt  # x
    new_state[1] += v * math.sin(state[2]) * dt  # y
    new_state[2] += w * dt                       # yaw
    
    # 更新速度
    new_state[3] = v  # 线速度
    new_state[4] = w  # 角速度
    
    return new_state


def plot_arrow(x, y, yaw, length=0.5, width=0.1):
    """绘制表示机器人朝向的箭头"""
    plt.arrow(
        x, y,
        length * math.cos(yaw), length * math.sin(yaw),
        head_length=width, head_width=width
    )
    plt.plot(x, y)


def plot_robot(x, y, yaw, radius):
    """绘制机器人形状（圆形）及其朝向"""
    # 绘制机器人主体（圆形）
    circle = plt.Circle((x, y), radius, color="b" , fill=False)
    plt.gcf().gca().add_artist(circle)
    
    # 绘制朝向线
    out_x, out_y = (
        np.array([x, y]) + np.array([np.cos(yaw), np.sin(yaw)]) * radius
    )
    plt.plot([x, out_x], [y, out_y], "-k")
    
def setup_figure():
    """设置图形窗口并添加键盘事件处理"""
    fig = plt.figure(figsize=(8, 6))
    # 添加ESC键退出功能
    plt.gcf().canvas.mpl_connect(
        "key_release_event",
        lambda event: [sys.exit() if event.key == "escape" else None]
    )
    return fig
  
def update_plot(robot_state, goal, obstacles, config, predicted_trajectory=None, path_history=None):
    """更新绘图，显示当前状态、轨迹和障碍物"""
    plt.cla()  # 清除当前坐标轴
    
    # 绘制预测轨迹
    if predicted_trajectory is not None:
        plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g", label="Predicted Path")
    
    # 绘制历史轨迹
    if path_history is not None:
        plt.plot(path_history[:, 0], path_history[:, 1], "-r", label="Robot Path")
    
    # 绘制机器人、目标和障碍物
    plt.plot(robot_state[0], robot_state[1], "xr", label="Robot")
    plt.plot(goal[0], goal[1], "xb", label="Goal")
    plt.plot(obstacles[:, 0], obstacles[:, 1], "ok", label="Obstacles")
    
    plot_robot(robot_state[0], robot_state[1], robot_state[2], config.robot_radius)
    plot_arrow(robot_state[0], robot_state[1], robot_state[2])
    
    # 设置图形属性
    plt.axis("equal")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("DWA Path Planning Simulation")
    
    plt.pause(0.001)  # 短暂暂停以更新图形

def run_simulation():
    """运行DWA路径规划仿真"""
    # 加载配置
    config = RobotConfig()
    config.set_scenario()
    
    # 初始化DWA规划器
    planner = DWA(config)
    
    # 初始化机器人状态 [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    robot_state = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    goal = config.target
    obstacles = config.obstacles
    
    # 设置图形界面
    fig = setup_figure()
    
    # 保存轨迹历史
    path_history = np.array([robot_state])
    
    # 开始仿真循环
    while True:
        # 计算控制量和预测轨迹
        control, predicted_trajectory = planner.plan(robot_state, goal, obstacles)
        
        # 更新机器人状态
        robot_state = update_state(robot_state, control, config.dt)
        path_history = np.vstack((path_history, robot_state))
        
        # 更新图形
        update_plot(robot_state, goal, obstacles, config, predicted_trajectory, path_history)
        
        # 检查是否到达目标
        dist_to_goal = math.hypot(robot_state[0] - goal[0], robot_state[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            print("目标达成！")
            break
    
    # 显示最终路径
    plt.title("DWA Path Planning Simulation")
    plt.plot(path_history[:, 0], path_history[:, 1], "-r", linewidth=2, label="Final Path")
    plt.legend()
    plt.pause(0.0005)
    plt.show()


if __name__ == "__main__":
    run_simulation()
