#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import robomaster
from robomaster import robot, vision
from datetime import datetime
import os
from ultralytics import YOLO

# 全局参数配置
# 蓝线HSV颜色范围
BLUE_HSV = (95, 80, 50), (130, 255, 255)



# 巡线PID参数
PID_PARAM = (0.4, 0.001, 0.05)
OUT_LIMIT = 80
SPD_BASE = 0.4
SPD_SLOW = 0.1
SPD_SCAN = 0.01
HIST_LEN = 8
LOST_MAX = 10
CTRL_T = 0.2

# 扫描参数
SCAN_ANGLE = 100  # 扩大扫描角度到100度
SCAN_SPEED = 50
SCAN_D_ANGLE = 10

# 避障参数
DIST_THR = 300  # 毫米
CAR_LABELS = {0, 1, 2}
BOX_COLOR = (0, 255, 0)  # 框
TEXT_EN = "team 17 stopped before a robot"



# 云台初始角度设置（统一参数）
GIMBAL_PITCH = -20  # 俯仰角度，稍微向下以便看到地面和前方
GIMBAL_YAW = 0  # 偏航角度，正前方

# 初始化YOLO模型（用于障碍物检测）
try:
    model = YOLO(r"E:\anaconda\envs\py3810\yolov5nu.pt")
    print("YOLO模型加载成功")
except Exception as e:
    print(f"YOLO模型加载失败: {e}")
    model = None

# 文件保存路径
SAVE_PATH = r"C:\Users\Suletta Mercury\Desktop\robomaster photo"  # 图片保存路径

# 全局变量
distance = [2000] * 4  # 距离传感器数据
current_state = "LINE_FOLLOWING"  # 当前状态: LINE_FOLLOWING, OBSTACLE


# 类定义
class SmoothPID:


    def __init__(self, kp, ki, kd, limit):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self.intg = 0.0
        self.last_err = 0.0
        self.last_t = time.time()

    def update(self, target, current):
        dt = time.time() - self.last_t
        if dt <= 0:
            dt = 0.01
        err = target - current

        # 比例项
        P = self.kp * err
        # 积分项（带限幅）
        self.intg += err * dt
        self.intg = np.clip(self.intg, -self.limit, self.limit)
        I = self.ki * self.intg
        # 微分项
        D = self.kd * (err - self.last_err) / dt

        self.last_err = err
        self.last_t = time.time()

        out = P + I + D
        return np.clip(out, -self.limit, self.limit)





#  功能函数
def sub_data_handler(sub_info):
    global distance
    distance = sub_info





def detect_line_in_frame(frame, hsv_range):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, *hsv_range)

    # 形态学操作去噪
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





def save_snapshot(frame, text):
    # 添加检测文字
    cv2.putText(frame, f"Team 17 detects {text}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 保存图片
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_file = os.path.join(SAVE_PATH, f"{text}_{timestamp}.jpg")

    cv2.imwrite(save_file, frame)
    print(f"快照已保存: {save_file}")
    return save_file


def snap_and_annotate(ep_camera):
    # 拍照
    img = ep_camera.read_cv2_image(strategy="newest", timeout=3)
    if img is None:
        return None

    h, w = img.shape[:2]

    # 识别障碍物
    if model is not None:
        results = model(img, conf=0.25, iou=0.45)
        det = results[0].boxes  # xyxy, cls, conf

        # 画框（只画 car/bus/truck）
        for box in det:
            cls = int(box.cls.item())
            if cls in CAR_LABELS:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                # 画正方形
                side = max(x2 - x1, y2 - y1)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                x1 = cx - side // 2
                y1 = cy - side // 2
                x2 = cx + side // 2
                y2 = cy + side // 2
                cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)

    # 添加文字说明
    cv2.putText(img, TEXT_EN, (50, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # 保存
    fname = os.path.join(SAVE_PATH, f"stopped_{datetime.now():%Y%m%d_%H%M%S}.jpg")
    cv2.imwrite(fname, img)
    print(f"已保存标注图: {fname}")
    return fname








# ---------------- 主函数 ----------------
def main():
    global current_state, distance

    # 初始化机器人
    print("初始化机器人...")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    print("机器人AP连接初始化成功")

    # 获取各个模块
    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal
    ep_sensor = ep_robot.sensor

    # 启动视频流和传感器
    ep_camera.start_video_stream(display=False)
    ep_sensor.sub_distance(freq=10, callback=sub_data_handler)

    # 初始化PID与平滑缓存
    pid = SmoothPID(*PID_PARAM, OUT_LIMIT)
    hist = []
    lost_cnt = 0

    # 设置云台初始角度
    ep_gimbal.recenter().wait_for_completed()
    ep_gimbal.moveto(pitch=GIMBAL_PITCH, yaw=GIMBAL_YAW,
                     pitch_speed=100, yaw_speed=100).wait_for_completed()

    # 状态机变量
    STATE_FOLLOW = 0  # 正常巡线
    STATE_SEARCH = 1  # 丢线后缓慢前移并扫描
    line_follow_state = STATE_FOLLOW

    # 扫描相关变量
    scan_direction = 1  # 1表示向右扫描，-1表示向左扫描
    current_yaw = 0
    scan_start_time = 0
    scan_period = 0.3  # 每0.3秒扫描一次步进

    print("系统初始化完成，开始巡线...")

    try:
        while True:
            # 读取图像和传感器数据
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            center_x = w // 2

            # 检查前方障碍物（最高优先级）
            front_dist = distance[0]
            if front_dist < DIST_THR and current_state != "OBSTACLE":
                print(f"前方有障碍物 ({front_dist} mm)，停车并拍照...")
                current_state = "OBSTACLE"
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
                snap_and_annotate(ep_camera)

            # 障碍物状态下等待障碍移除
            if current_state == "OBSTACLE":
                if distance[0] >= DIST_THR:
                    print("障碍物已移除，继续行驶")
                    current_state = "LINE_FOLLOWING"
                else:
                    time.sleep(0.1)
                    continue

            # 巡线状态处理
            if current_state == "LINE_FOLLOWING":
                # 检测线条
                line_x = detect_line_in_frame(frame, BLUE_HSV)

                if line_follow_state == STATE_FOLLOW:
                    if line_x is not None:
                        # 正常巡线
                        hist.append(line_x)
                        if len(hist) > HIST_LEN:
                            hist.pop(0)
                        smooth_pt = int(np.mean(hist))

                        error = center_x - smooth_pt
                        z_spd = pid.update(0, error)

                        # 速度随偏差衰减
                        spd_drop = min(abs(error) / 100.0, SPD_SLOW)
                        x_spd = SPD_BASE - spd_drop

                        ep_chassis.drive_speed(x=x_spd, y=0, z=z_spd, timeout=CTRL_T)
                        ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=z_spd)

                        # 可视化
                        cv2.circle(frame, (smooth_pt, h - 50), 8, (255, 0, 0), -1)
                        cv2.circle(frame, (center_x, h - 50), 8, (0, 255, 255), -1)
                        cv2.putText(frame, f"Err:{error}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        lost_cnt = 0
                    else:
                        lost_cnt += 1
                        if lost_cnt > LOST_MAX:
                            # 切换到搜索状态
                            line_follow_state = STATE_SEARCH
                            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=CTRL_T)
                            ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                            hist.clear()
                            scan_direction = 1  # 优先向右扫描
                            current_yaw = 0
                            scan_start_time = time.time()
                            print("线条丢失，进入扫描搜索模式...")

                elif line_follow_state == STATE_SEARCH:
                    # 不移动底盘，只转动云台

                    # 定时扫描
                    if time.time() - scan_start_time > scan_period:
                        scan_start_time = time.time()
                        current_yaw += scan_direction * SCAN_D_ANGLE

                        # 限制扫描范围在-100°到+100°之间
                        if current_yaw > SCAN_ANGLE:
                            scan_direction = -1  # 向右扫描到头后向左扫描
                            current_yaw = SCAN_ANGLE
                        elif current_yaw < -SCAN_ANGLE:
                            scan_direction = 1  # 向左扫描到头后向右扫描
                            current_yaw = -SCAN_ANGLE

                        # 只转动云台，不移动底盘
                        ep_gimbal.moveto(pitch=GIMBAL_PITCH, yaw=current_yaw,
                                         pitch_speed=100, yaw_speed=100).wait_for_completed()

                    # 在扫描过程中持续检测
                    scan_frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
                    scan_line_x = detect_line_in_frame(scan_frame, BLUE_HSV)
                    if scan_line_x is not None:
                        # 找到线条，切换回跟随状态
                        print("重新找到线条，恢复巡线...")
                        line_follow_state = STATE_FOLLOW
                        hist = [scan_line_x]
                        pid.intg = 0.0
                        pid.last_err = 0.0
                        ep_gimbal.moveto(pitch=GIMBAL_PITCH, yaw=GIMBAL_YAW,
                                         pitch_speed=100, yaw_speed=100).wait_for_completed()
                        continue

                    # 显示扫描状态
                    cv2.putText(frame, "LINE LOST - SCANNING", (w // 2 - 150, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 显示状态信息
            status_text = f"State: {current_state}"
            cv2.putText(frame, status_text, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 显示图像
            cv2.imshow("Main View", frame)

            # 显示蓝线掩模（调试用）
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv, *BLUE_HSV)
            kernel = np.ones((5, 5), np.uint8)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
            blue_mask = cv2.GaussianBlur(blue_mask, (5, 5), 0)
            blue_mask_resized = cv2.resize(blue_mask, (320, 240))
            cv2.imshow("Blue Mask", blue_mask_resized)

            # 退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户按下退出键，程序结束...")
                break

    except Exception as e:
        print(f"运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 释放资源
        try:
            ep_camera.stop_video_stream()
            ep_sensor.unsub_distance()
            ep_robot.close()
        except Exception as e:
            print(f"资源释放错误: {e}")
        cv2.destroyAllWindows()
        print("程序正常退出")


if __name__ == '__main__':
    main()