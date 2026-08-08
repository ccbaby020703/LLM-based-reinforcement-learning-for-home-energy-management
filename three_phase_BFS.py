# @description: three phase circuit simulation
import random

import scipy.io as sio

import numpy as np

import sys
import numpy

import pandas as pd
import copy as cp

import matplotlib.pyplot as plt
from typing import Optional, Tuple


class three_phase_BFS_ENV:
    def __init__(self):
        numpy.set_printoptions(threshold=sys.maxsize)  # 控制小数精度为最大值
        pd.set_option("display.max_rows", None, "display.max_columns", None)  # 控制输出全部行列

    def step(self, action,t) -> Tuple[np.ndarray, float, bool, float]:
        # action转换,action在-1~1之间，需要转换为实际值
        action2 = []
        PVP50 = np.array([0, 0, 0, 0, 0, 0.5173, 0.6201, 0.72448, 0.71834, 0.821476, .928768, 0.94279, 0.931949,
                          0.842391, 0.776828, 0.658307, 0.610582, 0.49699, 0.15, 0, 0, 0, 0, 0]) * 0.5 * 100
        PVP75 = PVP50 * 75 / 50
        PVP100 = PVP50 * 2
        PVP125 = PVP50 * 125 / 50
        PVP150 = PVP50 * 3
        PVP175 = PVP50 * 175 / 50

        # PV1 = 8  # phase a  agent1
        # PV2 = 35  # phase a
        # PV3 = 57  # phase b
        # PV4 = 75  # phase c  agent2
        # PV5 = 80  # phase b
        # PV6 = 107  # phase c
        # 容量分别为100 150 50 125 75 175的PV  对应  PV1-PV6  对应  PVP[0-5]

        PVP = [0 for i in range(6)]
        PVP[0] = PVP100[t] * (1 + 0.1 * (2 * random.random() - 1))
        PVP[1] = PVP150[t] * (1 + 0.1 * (2 * random.random() - 1))
        PVP[2] = PVP50[t] * (1 + 0.1 * (2 * random.random() - 1))
        PVP[3] = PVP125[t] * (1 + 0.1 * (2 * random.random() - 1))
        PVP[4] = PVP75[t] * (1 + 0.1 * (2 * random.random() - 1))
        PVP[5] = PVP175[t] * (1 + 0.1 * (2 * random.random() - 1))

        PVQ = [0 for i in range(6)]
        PVQ[0] = action[0][0] * np.sqrt(100 ** 2 - min(PVP[0], 100) ** 2)
        PVQ[1] = action[0][1] * np.sqrt(150 ** 2 - min(PVP[1], 150) ** 2)
        PVQ[2] = action[0][2] * np.sqrt(50 ** 2 - min(PVP[2], 50) ** 2)
        PVQ[3] = action[1][0] * np.sqrt(125 ** 2 - min(PVP[3], 125) ** 2)
        PVQ[4] = action[1][1] * np.sqrt(75 ** 2 - min(PVP[4], 75) ** 2)
        PVQ[5] = action[1][2] * np.sqrt(175 ** 2 - min(PVP[5], 175) ** 2)

        loadvar = 1  # 0.5~1
        Pd_24_total = np.array(
            [23, 22, 20, 28, 29, 45, 72, 85, 95, 99, 120, 99, 93, 102, 110, 108, 120, 122, 106, 98, 96, 80, 51, 32])
        Qd_24_total = Pd_24_total * 0.484
        Pd_rate = Pd_24_total / 75
        Qd_rate = Qd_24_total / 30

        # system parameter set up
        data = sio.loadmat('IEEE_123_3ph.mat')  # 载入mat文件

        net_para = data['IEEE_123_DF_3ph']

        n_branch = net_para.shape[0]
        n_bus = n_branch + 1
        ref_bus = 1 - 1
        Vr_m = 1.00

        cal = 2

        # In[17]:

        # BIBC BCBV
        K = np.zeros((n_branch, n_bus))
        Z = data['Z_3ph_pu']

        for br in range(n_branch):
            a = net_para[br, 2 - 1] - 1
            b = net_para[br, 3 - 1] - 1
            if b <= ref_bus:
                K[br, a] = -1
                K[br, b] = 1
            else:
                K[br, a] = 1
                K[br, b] = -1

        Gamma = np.delete(K, ref_bus, 1)

        # In[18]:

        # BIBC_0= -inv(Gamma)';
        BIBC_0 = - np.linalg.inv(np.transpose(Gamma))

        BIBC = np.zeros((3 * n_branch, 3 * n_branch))
        for b in range(n_branch):
            for bb in range(n_branch):
                BIBC[b * 3:(b * 3 + 3), bb * 3:(bb * 3 + 3)] = BIBC_0[b, bb] * np.eye(3)

        BCBV = np.complex_(np.zeros((3 * n_branch, 3 * n_branch)))
        for b in range(n_branch):
            for bb in range(n_branch):
                if BIBC_0[bb, b] == 1:
                    BCBV[b * 3:(b * 3 + 3), bb * 3:(bb * 3 + 3)] = Z[bb * 3:(bb * 3 + 3), 0:3]

        # In[19]:

        BCBV_Real = BCBV.real
        BCBV_Imag = BCBV.imag

        Pbus = np.zeros((3 * n_branch, 1))
        Qbus = np.zeros((3 * n_branch, 1))

        for i in range(n_branch):
            Pbus[i * 3 + 0, 0] = data['load_3ph'][i, 0] * Pd_rate[t] * 0.8 / 1000 / data['Sbase']
            Pbus[i * 3 + 1, 0] = data['load_3ph'][i, 2] * Pd_rate[t] * 0.7 / 1000 / data['Sbase']
            Pbus[i * 3 + 2, 0] = data['load_3ph'][i, 4] * Pd_rate[t] * 0.6 / 1000 / data['Sbase']
            Qbus[i * 3 + 0, 0] = data['load_3ph'][i, 1] * Qd_rate[t] * 0.8 / 1000 / data['Sbase']
            Qbus[i * 3 + 1, 0] = data['load_3ph'][i, 3] * Qd_rate[t] * 0.7 / 1000 / data['Sbase']
            Qbus[i * 3 + 2, 0] = data['load_3ph'][i, 5] * Qd_rate[t] * 0.6 / 1000 / data['Sbase']

        PV1 = 8  # phase a
        PV2 = 35  # phase a
        PV3 = 57  # phase b
        PV4 = 75  # phase c
        PV5 = 80  # phase b
        PV6 = 107  # phase c

        # MATLAB : Pbus(3*(PV1-1)+(1:3),1)=Pbus(3*(PV1-1)+(1:3),1)-[0]0;PVP(1)]/1000/S.Sbase;
        for i in range(3):
            tmp = [PVP[0], 0, 0]
            Pbus[3 * (PV1 - 1 - 1) + i, 0] = Pbus[3 * (PV1 - 1 - 1) + i, 0] - tmp[i] / 1000 / data['Sbase']
            # Qbus(3*(PV1-1)+(1:3),1)=Qbus(3*(PV1-1)+(1:3),1)+[0;0;PVQ(1)]/1000/S.Sbase;
            tmp = [PVQ[0], 0, 0]
            Qbus[3 * (PV1 - 1 - 1) + i, 0] = Qbus[3 * (PV1 - 1 - 1) + i, 0] - tmp[i] / 1000 / data['Sbase']
            # Pbus(3*(PV2-1)+(1:3),1)=Pbus(3*(PV2-1)+(1:3),1)-[PVP(2);0;0]/1000/S.Sbase;
            tmp = [PVP[1], 0, 0]
            Pbus[3 * (PV2 - 1 - 1) + i, 0] = Pbus[3 * (PV2 - 1 - 1) + i, 0] - tmp[i] / 1000 / data['Sbase']
            # Qbus(3*(PV2-1)+(1:3),1)=Qbus(3*(PV2-1)+(1:3),1)+[PVQ(2);0;0]/1000/S.Sbase;
            tmp = [PVQ[1], 0, 0]
            Qbus[3 * (PV2 - 1 - 1) + i, 0] = Qbus[3 * (PV2 - 1 - 1) + i, 0] - tmp[i] / 1000 / data['Sbase']
            # Pbus(3*(PV3-1)+(1:3),1)=Pbus(3*(PV3-1)+(1:3),1)-[0;0;PVP(3)]/1000/S.Sbase;
            tmp = [0, PVP[2], 0]
            Pbus[3 * (PV3 - 1 - 1) + i, 0] = Pbus[3 * (PV3 - 1 - 1) + i, 0] - tmp[i] / 1000 / data['Sbase']
            # Qbus(3*(PV3-1)+(1:3),1)=Qbus(3*(PV3-1)+(1:3),1)+[0;0;PVQ(3)]/1000/S.Sbase;
            tmp = [0, PVQ[2], 0]
            Qbus[3 * (PV3 - 1 - 1) + i, 0] = Qbus[3 * (PV3 - 1 - 1) + i, 0] - tmp[i] / 1000 / data['Sbase']
            # Pbus(3*(PV4-1)+(1:3),1)=Pbus(3*(PV4-1)+(1:3),1)-[PVP(4);0;0]/1000/S.Sbase;
            tmp = [0, 0, PVP[3]]
            Pbus[3 * (PV4 - 1 - 1) + i, 0] = Pbus[3 * (PV4 - 1 - 1) + i, 0] - tmp[i] / 1000 / data['Sbase']
            # Qbus(3*(PV4-1)+(1:3),1)=Qbus(3*(PV4-1)+(1:3),1)+[PVQ(4);0;0]/1000/S.Sbase;
            tmp = [0, 0, PVQ[3]]
            Qbus[3 * (PV4 - 1 - 1) + i, 0] = Qbus[3 * (PV4 - 1 - 1) + i, 0] - tmp[i] / 1000 / data['Sbase']
            # Pbus(3*(PV5-1)+(1:3),1)=Pbus(3*(PV5-1)+(1:3),1)-[PVP(5);0;0]/1000/S.Sbase;
            tmp = [0, PVP[4], 0]
            Pbus[3 * (PV5 - 1 - 1) + i, 0] = Pbus[3 * (PV5 - 1 - 1) + i, 0] - tmp[i] / 1000 / data['Sbase']
            # Qbus(3*(PV5-1)+(1:3),1)=Qbus(3*(PV5-1)+(1:3),1)+[PVQ(5);0;0]/1000/S.Sbase;
            tmp = [0, PVQ[4], 0]
            Qbus[3 * (PV5 - 1 - 1) + i, 0] = Qbus[3 * (PV5 - 1 - 1) + i, 0] - tmp[i] / 1000 / data['Sbase']
            # Pbus(3*(PV6-1)+(1:3),1)=Pbus(3*(PV6-1)+(1:3),1)-[PVP(6);0;0]/1000/S.Sbase;
            tmp = [0, 0, PVP[5]]
            Pbus[3 * (PV6 - 1 - 1) + i, 0] = Pbus[3 * (PV6 - 1 - 1) + i, 0] - tmp[i] / 1000 / data['Sbase']
            # Qbus(3*(PV6-1)+(1:3),1)=Qbus(3*(PV6-1)+(1:3),1)+[PVQ(6);0;0]/1000/S.Sbase;
            tmp = [0, 0, PVQ[5]]
            Qbus[3 * (PV6 - - 1) + i, 0] = Qbus[3 * (PV6 - 1 - 1) + i, 0] - tmp[i] / 1000 / data['Sbase']

        # In[20]:

        def cosd(a):
            return np.cos(np.deg2rad(a))

        def sind(a):
            return np.sin(np.deg2rad(a))

        # In[21]:

        # initialization
        Vr_3ph = Vr_m * np.array([[1], [cosd(-120) + (1j * sind(-120))], [cosd(120) + (1j * sind(120))]])
        Vr_n = np.complex_(np.ones((3 * n_branch, 1)))
        for b in range(n_branch):
            Vr_n[b * 3:b * 3 + 3, 0] = Vr_3ph[:, 0]

        # In[22]:

        # BFS - separate calculation
        it = 0

        V_Real_epsilon = 1
        suc = 1

        if cal == 1:
            V_Real = Vr_n.real
            V_Imag = Vr_n.imag

            while V_Real_epsilon > 1E-4:
                it = it + 1
                if it == 21:
                    suc = 0
                    break

                I_Real = (Pbus * V_Real + Qbus * V_Imag) / (V_Real * V_Real + V_Imag * V_Imag)
                I_Imag = (Pbus * V_Imag - Qbus * V_Real) / (V_Real * V_Real + V_Imag * V_Imag)

                I_br_Real = np.matmul(BIBC, I_Real)
                I_br_Imag = np.matmul(BIBC, I_Imag)

                delta_V_Real = np.matmul(BCBV_Real, I_br_Real) - np.matmul(BCBV_Imag, I_br_Imag)
                delta_V_Imag = np.matmul(BCBV_Real, I_br_Imag) + np.matmul(BCBV_Imag, I_br_Real)

                V_Real_epsilon = max(abs(V_Real - (Vr_n.real + delta_V_Real)))
                V_Real = Vr_n.real + delta_V_Real
                V_Imag = Vr_n.imag + delta_V_Imag

        # In[23]:

        # complex calculation
        if cal == 2:
            V = cp.deepcopy(Vr_n)
            while V_Real_epsilon > 1E-4:
                it = it + 1
                if it == 21:
                    suc = 0
                    break

                I = np.conj((Pbus + (1j * Qbus)) / V)
                delta_V = np.matmul(np.matmul(BCBV, BIBC), I)
                V_Real_epsilon = max(abs(V.real - (Vr_n.real + delta_V.real)))
                V = Vr_n - delta_V
            V_Real = V.real
            V_Imag = V.imag

            # In[24]:

            # result
            # if suc != 0:
            V_res = V_Real + (1j * V_Imag)
            V_magn = np.sqrt(V_Real * V_Real + V_Imag * V_Imag)
            V_angle = np.angle(V_res) / np.pi * 180

            V_comp = np.concatenate((V_magn, V_angle), axis=1)

            result = V_magn
            resultA = np.zeros((n_branch, 1))
            resultB = np.zeros((n_branch, 1))
            resultC = np.zeros((n_branch, 1))
            for b in range(n_branch):
                resultA[b, :] = result[b * 3 + 0, :]
                resultB[b, :] = result[b * 3 + 1, :]
                resultC[b, :] = result[b * 3 + 2, :]

            # todo:result*[:,2]->0.95~1.05 罚函数
            # todo:sum(Ploss)的绝对值 目标
            # todo:obs->resultA,resultB,resultC + 20个太阳能板输入 + loadvar
        R = 0  # 奖励
        VUF_step=0
        min_voltage = 0.96
        max_voltage = 1.04
        for i in range(113):
            if resultA[i, 0] < min_voltage or resultA[i, 0] > max_voltage:
                R = R - 4 * (abs(resultA[i, 0] - min_voltage) * (resultA[i, 0] < min_voltage) - abs(
                    resultA[i, 0] - max_voltage) *
                             (resultA[i, 0] > max_voltage)) - 0.4
            if resultB[i, 0] < min_voltage or resultB[i, 0] > max_voltage:
                R = R - 4 * (abs(resultB[i, 0] - min_voltage) * (resultB[i, 0] < min_voltage) - abs(
                    resultB[i, 0] - max_voltage) *
                             (resultB[i, 0] > max_voltage)) - 0.4
            if resultC[i, 0] < min_voltage or resultC[i, 0] > max_voltage:
                R = R - 4 * (abs(resultC[i, 0] - min_voltage) * (resultC[i, 0] < min_voltage) - abs(
                    resultC[i, 0] - max_voltage) *
                             (resultC[i, 0] > max_voltage)) - 0.4

        # V_res = V_Real + (1j * V_Imag)
        result_neg = np.zeros((n_branch, 1), dtype=complex)
        result_pos = np.zeros((n_branch, 1), dtype=complex)
        result_VUF = np.zeros((n_branch, 1))
        neg_factor = np.array([[1, cosd(-120) + (1j * sind(-120)), cosd(120) + (1j * sind(120))]])
        pos_factor = np.array([[1, cosd(120) + (1j * sind(120)), cosd(-120) + (1j * sind(-120))]])
        for i in range(113):
            result_neg[i, 0] = np.dot(neg_factor.flatten(), V_res[3 * i:3 * i + 3, 0])
            result_pos[i, 0] = np.dot(pos_factor.flatten(), V_res[3 * i:3 * i + 3, 0])
            result_VUF[i, 0] = np.abs(result_neg[i, 0]) / np.abs(result_pos[i, 0])
            VUF_single=100 * result_VUF[i, 0] * 0.2
            VUF_step = VUF_step+VUF_single
            R = R - VUF_single

        obs1 = np.append(np.append(resultA[0:59, 0], resultB[0:59, 0]), resultC[0:59, 0])
        # 把PVP加入obs
        obs1 = np.append(obs1, np.array(PVP[0:3]) / 100)
        # 把PVQ加入obs
        # obs = np.append(obs, np.array(PVQ) / 100)
        # 把loadvar加入obs
        obs1 = np.append(obs1, np.array(Pd_rate[t]))
        obs1 = np.append(obs1, np.array(Qd_rate[t]))

        obs2 = np.append(np.append(resultA[59:113, 0], resultB[59:113, 0]), resultC[59:113, 0])
        # 把PVP加入obs
        obs2 = np.append(obs2, np.array(PVP[3:6]) / 100)
        # 把PVQ加入obs
        # obs2 = np.append(obs2, np.array(PVQ) / 100)
        # 把loadvar加入obs
        obs2 = np.append(obs2, np.array(Pd_rate[t]))
        obs2 = np.append(obs2, np.array(Qd_rate[t]))

        obs=[obs1,obs2]

        if t >= 24 - 1:
            IsDone = True
        else:
            IsDone = False

        return obs, R, IsDone, PVQ, VUF_step

    def reset(self):
        self.t = 0
        obs1 = np.array([1 for _ in range(3*59+5)])
        obs2 = np.array([1 for _ in range(3*54+5)])
        obs=[obs1,obs2]

        return obs
