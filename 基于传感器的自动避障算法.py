import cv2
import numpy as np
from robomaster import robot, vision
import time
import os
from datetime import datetime

# ==================== 全局参数配置 ====================
TEAM_NUMBER = 17

# 巡线参数
BLUE_HSV = (95, 80, 50), (130, 255, 255)  # 蓝线HSV范围
PID_PARAM = (0.4, 0.001, 0.05)            # PID参数 (Kp, Ki, Kd)
OUT_LIMIT = 80                             # PID输出限幅
SPD_BASE = 0.3                             # 基础前进速度
SPD_SLOW = 0.1                             # 最大减速量
SPD_SCAN = 0.01                            # 扫描时前进速度
HIST_LEN = 8                               # 平滑历史长度
LOST_MAX = 10                              # 丢线最大帧数
CTRL_T = 0.2                               # 底盘指令周期
SCAN_ANGLE = 120                          # 扫描角度范围
SCAN_SPEED = 50                            # 扫描速度（度/秒）
SCAN_D_ANGLE = 5                         # 每次扫描步进角度
SCAN_PITCH = -15                          # 扫描时的俯仰角度



# 红绿灯识别参数
RED_LOWER1 = (0, 150, 150)                 # 调整红色范围，避免与红色Marker冲突
RED_UPPER1 = (10, 255, 255)
RED_LOWER2 = (170, 150, 150)               # 调整红色范围，避免与红色Marker冲突
RED_UPPER2 = (180, 255, 255)
GREEN_LOWER = (60, 80, 50)
GREEN_UPPER = (80, 255, 255)
RED_THRESHOLD = 1200                       # 提高红灯检测阈值
GREEN_THRESHOLD = 800                     # 提高绿灯检测阈值
LIGHT_CHECK_INTERVAL = 0.5                 # 红绿灯检查间隔(秒)
MIN_CIRCULARITY = 0.8                      # 最小圆形度阈值
MIN_ASPECT_RATIO = 0.8                     # 最小宽高比阈值
MAX_ASPECT_RATIO = 1.2                     # 最大宽高比阈值
MIN_LIGHT_RADIUS = 10                      # 最小灯光半径
MAX_LIGHT_RADIUS = 50                      # 最大灯光半径

# 交通堵塞检测参数
DIST_THR = 400                             # 距离阈值(mm)
BOX_COLOR = (0, 255, 0)                    # 框颜色(绿色)
JAM_CHECK_INTERVAL = 0.3                   # 交通堵塞检查间隔(秒)

# 文件保存路径
SAVE_PATH = r"D:\PythonProject"            # 图片保存路径

# ==================== 类定义 ====================
class SmoothPID:
    """带积分限幅的PID控制器"""
    def __init__(self, kp, ki, kd, limit):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self.intg = 0.0
        self.last_err = 0.0
        self.last_t = time.time()

    def update(self, target, current):
        dt = time.time() - self.last_t
        if dt <= 0: dt = 0.01
        err = target - current

        # 比例项
        P = self.kp * err
        # 积分项（限幅）
        self.intg += err * dt
        self.intg = np.clip(self.intg, -self.limit, self.limit)
        I = self.ki * self.intg
        # 微分项
        D = self.kd * (err - self.last_err) / dt
        self.last_err = err
        self.last_t = time.time()

        out = P + I + D
        return np.clip(out, -self.limit, self.limit)



# ==================== 全局变量 ====================
distance = [2000] * 4           # 距离传感器数据
current_state = "LINE_FOLLOWING"  # 当前状态
traffic_light_state = "GREEN"   # 红绿灯状态
last_save_time = 0              # 上次保存时间
snapshot_count = 0              # 快照计数
lost_cnt = 0                    # 丢线计数
scan_direction = 1              # 扫描方向
current_yaw = 0                 # 当前yaw角度
scan_start_time = 0             # 扫描开始时间
last_light_check = 0            # 上次红绿灯检查时间
last_jam_check = 0              # 上次交通堵塞检查时间

# ==================== 功能函数 ====================
def ensure_save_directory():
    """确保保存目录存在"""
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        print(f"Created image save directory: {SAVE_PATH}")
    else:
        print(f"Images will be saved to: {SAVE_PATH}")



def save_snapshot(frame, text, object_type="marker", center_x=None, center_y=None):
    """保存包含检测信息的快照"""
    global snapshot_count, last_save_time
    
    # 添加文本信息
    cv2.putText(frame, text, (50, 50 if center_y is None else center_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_file = os.path.join(SAVE_PATH, f"{object_type}_{snapshot_count}_{timestamp}.jpg")
    cv2.imwrite(save_file, frame)
    print(f"Snapshot saved: {save_file}")
    
    snapshot_count += 1
    last_save_time = time.time()
    return save_file



def detect_line_in_frame(frame, hsv_range):
    """在一帧图像中检测线条位置，返回线条中心x坐标或None"""
    if frame is None:
        return None
        
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, *hsv_range)

    # 去噪：闭→开→高斯
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # 多行ROI检测
    roi_rows = [h - 300, h - 280, h - 260]
    line_pts = []
    for y in roi_rows:
        row = mask[y, :]
        whites = np.where(row == 255)[0]
        if len(whites) > 10:
            line_pts.append(int(np.mean(whites)))

    if line_pts:
        return int(np.mean(line_pts))
    else:
        return None

def is_circular_contour(cnt):
    """检查轮廓是否为圆形"""
    # 计算轮廓面积
    area = cv2.contourArea(cnt)
    if area < 100:  # 面积太小，忽略
        return False
    
    # 计算轮廓周长
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return False
    
    # 计算圆形度
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # 计算外接矩形
    _, _, w, h = cv2.boundingRect(cnt)
    if w == 0 or h == 0:
        return False
    
    # 计算宽高比
    aspect_ratio = float(w) / h
    
    # 计算等效半径
    equivalent_radius = np.sqrt(area / np.pi)
    
    # 检查是否为圆形
    return (circularity > MIN_CIRCULARITY and 
            MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO and
            MIN_LIGHT_RADIUS < equivalent_radius < MAX_LIGHT_RADIUS)

def sub_distance_handler(sub_info):
    """距离传感器数据回调函数"""
    global distance
    distance = sub_info

def detect_traffic_jam(ep_camera):
    """检测交通堵塞并返回结果和标注后的图像"""
    global last_save_time, snapshot_count
    
    # 读取图像
    img = ep_camera.read_cv2_image(strategy="newest", timeout=3)
    if img is None:
        return False, None
        
    h, w = img.shape[:2]
    
    # 简单检测：如果前方有障碍物，假设是车辆
    if distance[0] < DIST_THR:
        text = f"Team {TEAM_NUMBER} stopped before a robot"
        cv2.putText(img, text, (50, h-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # 在图像中心画一个矩形表示检测到的障碍物
        rect_size = 100
        x1 = w//2 - rect_size//2
        y1 = h//2 - rect_size//2
        x2 = w//2 + rect_size//2
        y2 = h//2 + rect_size//2
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)
        
        # 保存图像
        current_time = time.time()
        if current_time - last_save_time > 2.0:
            save_snapshot(img.copy(), text, "traffic_jam")
        return True, img
    return False, img

# ==================== 主函数 ====================
def main():
    global current_state, lost_cnt, scan_direction
    global current_yaw, scan_start_time, last_jam_check
    
    # 确保保存目录存在
    ensure_save_directory()
    
    # 初始化机器人
    print("Initializing robot...")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    print("Robot AP connection initialized successfully")
    
    # 获取各个模块
    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal
    ep_sensor = ep_robot.sensor
    
    # 设置摄像头角度
    ep_gimbal.recenter().wait_for_completed()
    ep_gimbal.moveto(pitch=SCAN_PITCH, yaw=0, pitch_speed=50, yaw_speed=50).wait_for_completed()
    
    # 启动视频流
    ep_camera.start_video_stream(display=False)
    
    # 订阅距离传感器
    ep_sensor.sub_distance(freq=10, callback=sub_distance_handler)
    time.sleep(1)  # 等待传感器数据稳定
    
    # 初始化PID与平滑缓存
    pid = SmoothPID(*PID_PARAM, OUT_LIMIT)
    hist = []
    
    # 状态机状态
    STATE_FOLLOW = 0      # 正常巡线
    STATE_SEARCH = 1      # 丢线后缓慢前移并扫描
    line_following_state = STATE_FOLLOW
    
    # 扫描相关变量
    scan_period = 0.3  # 每0.5秒扫描一次步进
    
    # 任务执行标志
    task_in_progress = False
    
    print("System initialized. Starting line following...")
    
    try:
        while True:
            # 从摄像头读取图像
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if frame is None:
                continue
                
            h, w = frame.shape[:2]
            center_x = w // 2
            center_y = h // 2
            
            # 初始化mask变量
            mask = None
            
            # 显示图像（避免画面卡住）
            display_frame = frame.copy()
            
            # ========== 1. 检查交通堵塞 ==========
            current_time = time.time()
            if current_time - last_jam_check > JAM_CHECK_INTERVAL and not task_in_progress:
                last_jam_check = current_time
                
                front_distance = distance[0]  # 前方距离
                if front_distance < DIST_THR:
                    print(f"Traffic jam detected! Distance: {front_distance} mm")
                    task_in_progress = True
                    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
                    ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)  # 停止云台
                    
                    # 检测并保存交通堵塞图像
                    jam_detected, jam_frame = detect_traffic_jam(ep_camera)
                    
                    # 等待交通堵塞解除
                    while distance[0] < DIST_THR:
                        time.sleep(0.1)
                    
                    print("Traffic jam cleared, resuming...")
                    task_in_progress = False
                    # 重置丢线计数，避免交通堵塞被误识别为断线
                    lost_cnt = 0
                    continue
            
            # ========== 2. 巡线逻辑 ==========
            if current_state == "LINE_FOLLOWING" and not task_in_progress:
                # HSV提取蓝线
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, *BLUE_HSV)
                
                # 去噪：闭→开→高斯
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.GaussianBlur(mask, (5, 5), 0)

                # 多行ROI检测
                roi_rows = [h - 100, h - 70, h - 40]
                line_pts = []
                for y in roi_rows:
                    row = mask[y, :]
                    whites = np.where(row == 255)[0]
                    if len(whites) > 10:
                        line_pts.append(int(np.mean(whites)))
                        cv2.line(display_frame, (0, y), (w, y), (0, 255, 0), 1)
                        cv2.circle(display_frame, (line_pts[-1], y), 5, (0, 0, 255), -1)

                if line_pts:
                    # 正常巡线状态
                    line_following_state = STATE_FOLLOW
                    
                    avg_pt = int(np.mean(line_pts))
                    hist.append(avg_pt)
                    if len(hist) > HIST_LEN: hist.pop(0)
                    smooth_pt = int(np.mean(hist))

                    error = center_x - smooth_pt
                    z_spd = pid.update(0, error)  # 目标误差=0

                    # 速度随偏差衰减
                    spd_drop = min(abs(error) / 100.0, SPD_SLOW)
                    x_spd = SPD_BASE - spd_drop

                    # 控制底盘+云台同步旋转
                    ep_chassis.drive_speed(x=x_spd, y=0, z=z_spd, timeout=CTRL_T)
                    ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=z_spd)

                    cv2.circle(display_frame, (smooth_pt, h - 50), 8, (255, 0, 0), -1)
                    cv2.circle(display_frame, (center_x, h - 50), 8, (0, 255, 255), -1)
                    cv2.putText(display_frame, f"Err:{error}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    lost_cnt = 0
                else:
                    # 丢线处理 - 只有在没有交通堵塞时才进行
                    if distance[0] > DIST_THR * 1.5:  # 确保前方没有障碍物
                        lost_cnt += 1
                        if lost_cnt > LOST_MAX and line_following_state == STATE_FOLLOW:
                            # 切换到搜索状态
                            line_following_state = STATE_SEARCH
                            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=CTRL_T)
                            ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                            hist.clear()
                            scan_direction = 1
                            current_yaw = 0
                            scan_start_time = time.time()
                            print("Line lost, entering scan mode...")
                        
                        if line_following_state == STATE_SEARCH:
                            # 缓慢前移
                            ep_chassis.drive_speed(x=SPD_SCAN, y=0, z=10, timeout=CTRL_T)

                            # 定时扫描
                            if time.time() - scan_start_time > scan_period:
                                scan_start_time = time.time()
                                current_yaw += scan_direction * SCAN_D_ANGLE
                                if abs(current_yaw) > SCAN_ANGLE:
                                    scan_direction *= -1
                                    current_yaw = np.clip(current_yaw, -SCAN_ANGLE, SCAN_ANGLE)

                                # 只旋转云台，不旋转底盘
                                ep_gimbal.moveto(pitch=SCAN_PITCH, yaw=current_yaw, pitch_speed=100, yaw_speed=SCAN_SPEED).wait_for_completed()

                            # 在扫描过程中持续检测
                            scan_frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
                            scan_line_x = detect_line_in_frame(scan_frame, BLUE_HSV)
                            if scan_line_x is not None:
                                # 找到线条，切换回跟随状态
                                print("Line found, resuming following...")
                                line_following_state = STATE_FOLLOW
                                hist = [scan_line_x]
                                pid.intg = 0.0
                                pid.last_err = 0.0
                                ep_gimbal.moveto(pitch=SCAN_PITCH, yaw=0, pitch_speed=100, yaw_speed=SCAN_SPEED).wait_for_completed()
                                continue

                            # 显示扫描状态
                            cv2.putText(display_frame, "LINE LOST - SCANNING", (w//2 - 150, h//2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        # 前方有障碍物，重置丢线计数
                        lost_cnt = 0

                # 显示检测状态
                status_text = f"State: {current_state}"
                cv2.putText(display_frame, status_text, (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            

            cv2.imshow("Line Following", display_frame)
            if mask is not None:
                cv2.imshow("Mask", mask)
            

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User pressed quit key, exiting...")
                break

    except Exception as e:
        print(f"Error during operation: {e}")
        import traceback
        traceback.print_exc()
    finally:

        try:
            ep_sensor.unsub_distance()
            ep_camera.stop_video_stream()
            ep_robot.close()
        except:
            pass
        cv2.destroyAllWindows()
        print("Program exited normally")

if __name__ == '__main__':
    main()