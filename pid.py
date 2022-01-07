from PyQt5.QtCore import *
import time
import numpy as np
import imgProcess
import main
import itertools  # gat all permutations
import send2Serial
import cv2
import kal


# --------PID CONSTANT BEGIN------- #
# --------添加PID相关常量定义------ #
Mode = 3
Kp = 0.35
Ki = 0.15
Kd = 0.01
MAX_CNT = 5
INF = 100000000.0
SIGNAL = [1300.0, 1500.0]  # 复位位置
# --------PID CONSTANT END--------- #


# 子线程类，选手需完成其中PID运算部分
class WorkThread(QThread):
    def __init__(self, work_widget=main.WorkWidget, parent=None):
        super().__init__(parent)
        self.work_widget = work_widget

    def run(self):
        # --------PID VARIABLE BEGIN------- #
        control_signal = SIGNAL
        K = 1.0  # urr前的常数
        targets = []
        point = ()
        flag1 = False  # 记录初始targets是否连续两帧完成五个圆的识别
        flag2 = False  # 记录是否连续两帧完成三个圆的识别
        flag3 = False  # 是否正在复位过程中
        flag4 = False  # 受否需要减小调整量
        tri_tar = [(320.0, 240.0), (320.0, 240.0), (320.0, 240.0), (320.0, 240.0), (320.0, 240.0), (320.0, 240.0)]
        # 记录五个圆的坐标

        order = [1, 2, 3, 4, 5]   # 生成全排列所用变量

        seq = [1, 2, 3, 4, 5]  # 最短路径序列

        cur_seq = [0, 1, 2, 3, 4]  # cur_seq[i]代表编号为i的圆在targets的位置

        prem = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]  # 012的全排列

        dist = [[], [], [], [], [], [], []]  # 五个圆两两之间的距离

        cnt = 1  # 当前正在找的圆的序号

        cnt_nxt = 0  # 连续几帧减少且数目保持相同

        cnt_kalman = 0  # 缓冲前几帧的卡尔曼滤波

        predicted = [[], [], [], []]  # 对下一帧的卡尔曼滤波预测

        pre_predicted = [[], [], [], []]  # 对当前帧的卡尔曼滤波预测

        sign = []  # 标准值

        cur = [320.0, 240.0]  # 当前值

        num_circle = [0]  # 识别出的圆的个数

        pt_num_circle = 0  # num_circle的位置指针

        err = [0.0, 0.0]  # 误差值 e(t)

        last_err = [0.0, 0.0]  # e(t-1)

        pre_err = [0.0, 0.0]  # 误差累加值 e(t-2)

        urr = [0.0, 0.0]  # 通过PID控制的新输出值

        t1 = 0.0  # 激光点鲁棒性

        t2 = 0.0  # 激光点鲁棒性

        # --------PID VARIABLE END--------- #

        # -------dist-------------------  #
        def dis(p1, p2):
            if p1[0] == -1 or p2[0] == -1:
                return INF / 10
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        # -------MIN DIST---------------- #
        def task1():
            for i in range(0, 6):
                for j in range(0, 6):
                    dist[i].append(dis(tri_tar[i], tri_tar[j]))
            min_dist = INF
            ans = [0, 1, 2, 3, 4, 5]
            permutation = list(itertools.permutations(order))
            for x in permutation:
                tmp = 0
                x = list(x)  # 元组转化为列表
                x.insert(0, 0)
                for j in range(1, 6):
                    tmp += dist[x[j - 1]][x[j]]
                if tmp < min_dist:
                    min_dist = tmp
                    ans = x
            # print("ans=", ans)
            return ans

        # -------识别出几个圆--------------- #
        def num_circles():
            tot = 0
            while targets[tot][0] != -1:
                tot += 1
                if tot >= 5:
                    break
            return tot

        # -----------是否出界--------------- #
        def out_range(p1):
            if p1[0] < 58 or p1[0] > 582:
                return True
            if p1[1] < 2 or p1[1] > 358:
                return True
            return False

        kf = kal.KalmanFilter4(kf=cv2.KalmanFilter(4, 2))
        kf0 = kal.KalmanFilter6(kf=cv2.KalmanFilter(6, 2))
        kf1 = kal.KalmanFilter6(kf=cv2.KalmanFilter(6, 2))
        kf2 = kal.KalmanFilter6(kf=cv2.KalmanFilter(6, 2))
        # start_time = time.time()
        send2Serial.send2serial(SIGNAL)  # run之后激光点的初始值
        while True:
            # new_time = time.time()
            # pass_time = new_time - start_time
            # start_time = new_time
            # print('image processing: ')
            image, targets, point, ok = imgProcess.img_process()
            # print('gap')
            self.work_widget.update_display(image, targets, point)
            # print('PID culculating: ')

            # --------PID CODE BEGIN-------- #
            flag4 = False  # 初始化

            if not ok:  # 没有识别出二维码跳过
                continue

            num_circle.append(num_circles())  # 更新识别出的圆的数组
            pt_num_circle += 1

            if num_circle[pt_num_circle] == 0:  # 没有识别出圆跳过
                continue

            if out_range(point):   # 激光点脱靶则寻求复位
                flag3 = True
                # cnt_kalman = 0  # 测试是否需要加这一句
                t1 = time.time()

            cur = np.array([point[0], point[1]])

            # ----------TASK ONE BEGIN------------- #
            if Mode == 1:
                # 引入目标点（连续两帧识别出五个圆）
                if not flag1:
                    if num_circle[pt_num_circle] == 5 and num_circle[pt_num_circle-1] == 5:
                        flag1 = True
                        for i in range(1, 6):
                            tri_tar[i] = (targets[i-1][0], targets[i-1][1])
                        tri_tar[0] = point  # 一开始尽可能让激光点打到正中间
                        seq = task1()
                    else:
                        continue

                # 切换目标操作（连续三帧识别出的圆数目减少1）
                if num_circle[pt_num_circle] == 5 - cnt and pt_num_circle >= 2:
                    if num_circle[pt_num_circle-1] == 5-cnt and num_circle[pt_num_circle-2] == 5-cnt:
                        cnt += 1

                sign = tri_tar[seq[cnt]]
                # if has_arrived():
                #   cnt += 1
                if cnt >= 6:
                    break

            # ----------TASK ONE END-------------- #

            # ----------TASK TWO BEGIN------------ #

            # 试验三 卡尔曼滤波
            if Mode == 2:
                measured = np.array([targets[0][0], targets[0][1]], np.float32)

                kf.kf.correct(measured)
                predicted = kf.kf.predict()

                sign = np.array([predicted[0] + predicted[2] * 4.0, predicted[1] + predicted[3] * 4.0])  # 4帧以后的位置？
                cnt_kalman += 1

                if dis(pre_predicted, measured) > 20:
                    # cnt_kalman = 0  # 不出现前后预测超出20的不稳定情况一旦出现稳5帧
                    flag4 = True   # 当前测试偏差过大（速度过大）,则调整量改为一半,待测试,两种改进二选一

                pre_predicted = predicted
                if cnt_kalman < MAX_CNT:
                    continue
                # cv2.circle(image, (int(sign[0]), int(sign[1])), 15, (0, 255, 0), 2)

            # ----------TASK TWO END-------------- #

            # ----------TASK THREE BEGIN---------- #
            if Mode == 3:
                if not flag2:  # 第一次连续两帧识别出三个圆才开始任务三
                    if num_circle[pt_num_circle] != 3 or num_circle[pt_num_circle-1] != 3:
                        continue
                    else:
                        flag2 = True

                if num_circle[pt_num_circle] == 3 - cnt and pt_num_circle >= 2:  # 连续三帧圆减少1切换目标
                    if num_circle[pt_num_circle - 1] == 3 - cnt and num_circle[pt_num_circle - 2] == 3 - cnt:
                        cnt += 1
                        flag4 = True
                        # cnt_kalman = 0
                if num_circle[pt_num_circle] == 5 - cnt and num_circle[pt_num_circle - 1] == 5 - cnt:  # 连续两帧识别多1则cnt--
                    cnt -= 1
                if cnt >= 4:  # 打完圆结束
                    break

                if cnt_kalman >= MAX_CNT:  # 圆的前后匹配
                    if cnt == 3:
                        cur_seq = [0, 0, 0, 0]
                    if cnt == 2:
                        d1 = dis(pre_predicted[1], targets[0]) + dis(pre_predicted[2], targets[1])
                        d2 = dis(pre_predicted[1], targets[1]) + dis(pre_predicted[2], targets[0])
                        if d1 > d2:
                            cur_seq = np.array([0, 1, 0])
                        else:
                            cur_seq = np.array([0, 0, 1])
                    if cnt == 1:
                        min_dist = INF
                        tot = 0
                        for i in range(6):
                            cur_dist = dis(pre_predicted[0], targets[prem[i][0]]) + \
                                       dis(pre_predicted[1], targets[prem[i][1]]) + \
                                       dis(pre_predicted[2], targets[prem[i][2]])
                            if cur_dist < min_dist:
                                min_dist = cur_dist
                                tot = i
                                cur_seq = np.array([prem[i][0], prem[i][1], prem[i][2]])
                            if targets[prem[tot][0]][0] == -1:
                                continue

                measured0 = np.array([targets[cur_seq[0]][0], targets[cur_seq[0]][1]], np.float32)
                measured1 = np.array([targets[cur_seq[1]][0], targets[cur_seq[1]][1]], np.float32)
                measured2 = np.array([targets[cur_seq[2]][0], targets[cur_seq[2]][1]], np.float32)

                kf0.kf.correct(measured0)
                kf1.kf.correct(measured1)
                kf2.kf.correct(measured2)

                predicted[0] = kf0.kf.predict()
                predicted[1] = kf1.kf.predict()
                predicted[2] = kf2.kf.predict()

                pre_predicted = predicted
                sign = np.array([predicted[cnt-1][0] + predicted[cnt-1][2] * 2.0, predicted[cnt-1][1] + predicted[cnt-1][3] * 2.0])  # 可能可以以速度为指标
                cnt_kalman += 1
                if cnt_kalman < MAX_CNT:
                    continue
                if targets[cur_seq[cnt-1]][0] == -1:  # 匹配得出当前跟踪的圆消失，跳过->是否解决切换目标过程中激光点打出的问题？
                    continue
            # ----------TASK THREE END------------ #

            # --------PID ------------------------ #
            if out_range(sign):   # 目标点确定失败则跳过，标准？
                continue
            K = 0.65 if flag4 else 1.0

            err = np.array([sign[0] - cur[0], sign[1] - cur[1]])
            urr = np.array([Kp * (err[0] - last_err[0]) + Ki * err[0] + Kd * (err[0] - 2 * last_err[0] + pre_err[0]),
                            Kp * (err[1] - last_err[1]) + Ki * err[1] + Kd * (err[1] - 2 * last_err[1] + pre_err[1])])
            pre_err = last_err
            last_err = err

            # --------PID END--------------------- #

            # control_signal = np.array([control_signal[0]+1, control_signal[1]])
            control_signal = np.array([control_signal[0] + K * urr[1], control_signal[1] - K * urr[0]])

            if flag3:  # 激光点脱靶复位一段时间
                t2 = time.time()
                control_signal = SIGNAL
                if t2 - t1 >= 0.1:
                    flag3 = False

            if control_signal[0] >= 2498.0 or control_signal[1] >= 2498.0:  # 限制输出信号值
                continue
            if control_signal[0] <= 502.0 or control_signal[1] <= 502.0:
                continue
            # --------PID CODE END---------- #

            # 通过串口通信将舵机运动信息发送至单片机
            send2Serial.send2serial(control_signal)
            # 计算帧率
            # print(num_circle[pt_num_circle])
            # print("cnt=", cnt)
            # print(targets[0])
            # print("cur=", cur)
            # print(seq)
            # print(measured2[0], measured2[1])
            # print("vx=", predicted[2], "vy=", predicted[3])
            # print(flag3)
            # print(predicted[0], predicted[1])
            # print("cnt_kalman=", cnt_kalman)
            # print(cur_seq)
            # print("sign=", sign)
            # print("cur=", cur)
            # print(f'FPS :{1 / pass_time }')
            # print("=" * 40)
