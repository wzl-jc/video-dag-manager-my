import numpy as np
import math
from logging_utils import root_logger

class Sniffer():
    CONTENT_ELE_MAXN = 50

    def __init__(self, job_uid):
        self.job_uid = job_uid
        self.runtime_pkg_list = dict()
        self.runtime_info_list = dict()  # key为任务名，value为列表

    # TODO：根据taskname解析output_ctx，得到运行时情境
    def sniff(self, taskname, output_ctx):
        if taskname == 'end_pipe':
            if 'delay' not in self.runtime_pkg_list:
                self.runtime_pkg_list['delay'] = list()
            
            if len(self.runtime_pkg_list['delay']) > Sniffer.CONTENT_ELE_MAXN:
                del self.runtime_pkg_list['delay'][0]
            self.runtime_pkg_list['delay'].append(output_ctx['delay'])

        # 对face_detection的结果，提取运行时情境
        # TODO：目标数量、目标大小、目标速度
        if taskname == 'face_detection':
            # 定义运行时情境字段
            if 'obj_n' not in self.runtime_pkg_list:
                self.runtime_pkg_list['obj_n'] = list()
            if 'obj_size' not in self.runtime_pkg_list:
                self.runtime_pkg_list['obj_size'] = list()

            # 更新各字段序列（防止爆内存）
            if len(self.runtime_pkg_list['obj_n']) > Sniffer.CONTENT_ELE_MAXN:
                del self.runtime_pkg_list['obj_n'][0]
            self.runtime_pkg_list['obj_n'].append(len(output_ctx['faces']))

            obj_size = 0
            for x_min, y_min, x_max, y_max in output_ctx['bbox']:
                # TODO：需要依据分辨率转化
                obj_size += (x_max - x_min) * (y_max - y_min)
            obj_size /= len(output_ctx['bbox'])

            if len(self.runtime_pkg_list['obj_size']) > Sniffer.CONTENT_ELE_MAXN:
                del self.runtime_pkg_list['obj_size'][0]
            self.runtime_pkg_list['obj_size'].append(obj_size)
        
        # 对car_detection的结果，提取目标数量
        # TODO：目标数量、目标大小、目标速度
        if taskname == 'car_detection':
            # 定义运行时情境字段
            if 'obj_n' not in self.runtime_pkg_list:
                self.runtime_pkg_list['obj_n'] = list()

            # 更新各字段序列（防止爆内存）
            if len(self.runtime_pkg_list['obj_n']) > Sniffer.CONTENT_ELE_MAXN:
                del self.runtime_pkg_list['obj_n'][0]
            self.runtime_pkg_list['obj_n'].append(
                sum(list(output_ctx['count_result'].values()))
            )

        # 保存运行时情境画像需要的数据
        if taskname == 'runtime_portrait':
            for service_name in output_ctx:
                if service_name not in self.runtime_info_list:
                    self.runtime_info_list[service_name] = list()

                # 保存用户的任务约束（任务约束针对整个任务而不是某个服务）
                if service_name == 'user_constraint':
                    self.runtime_info_list[service_name].append(output_ctx[service_name])
                # 保存每一个服务的资源情境、工况情境、任务可配置参数
                else:
                    temp_runtime_dict = dict()
                    # 保存服务的资源情境
                    temp_runtime_dict['resource_runtime'] = output_ctx[service_name]['proc_resource_info']
                    # 保存服务的任务可配置参数
                    temp_runtime_dict['task_conf'] = output_ctx[service_name]['task_conf']
                    # 保存服务的工况情境
                    temp_runtime_dict['work_runtime'] = dict()
                    if service_name == 'face_detection':
                        temp_runtime_dict['work_runtime']['obj_n'] = len(output_ctx[service_name]['faces'])
                    if service_name == 'face_alignment':
                        temp_runtime_dict['work_runtime']['obj_n'] = output_ctx[service_name]['count_result']['total']
                    if service_name == 'car_detection':
                        temp_runtime_dict['work_runtime']['obj_n'] = sum(list(output_ctx[service_name]['count_result'].values()))
                    self.runtime_info_list[service_name].append(temp_runtime_dict)
                # 避免保存过多的内容导致爆内存
                if len(self.runtime_info_list[service_name]) > Sniffer.CONTENT_ELE_MAXN:
                    del self.runtime_info_list[service_name][0]

    def describe_runtime(self):
        # TODO：聚合情境感知参数的时间序列，给出预估值/统计值
        runtime_desc = dict()
        for k, v in self.runtime_pkg_list.items():
            runtime_desc[k] = sum(v) * 1.0 / len(v)
        
        # 获取场景稳定性
        if 'obj_n' in self.runtime_pkg_list.keys():
            runtime_desc['obj_stable'] = True if np.std(self.runtime_pkg_list['obj_n']) < 0.3 else False

        # 每次调用agg后清空
        self.runtime_pkg_list = dict()

        # 获取运行时情境画像+知识库所需参数
        service_list = list(self.runtime_info_list.keys())
        if len(service_list) == 0:  # service_list长度为0，说明任务还未完整地执行一次，尤其是第一次执行任务时时间很长
            return runtime_desc
        assert 'user_constraint' in service_list
        service_list.remove('user_constraint')
        runtime_desc['runtime_portrait'] = dict()  # key为任务名，value为列表

        for service_name in service_list:
            runtime_desc['runtime_portrait'][service_name] = list()

        for i in range(len(self.runtime_info_list['user_constraint'])):
            delay_constraint = self.runtime_info_list['user_constraint'][i]['delay']  # 用户的时延约束
            delay_exec = 0  # 整个任务实际的执行时延
            for service_name in service_list:
                delay_exec += self.runtime_info_list[service_name][i]['resource_runtime']['all_latency']

            for service_name in service_list:
                cpu_util_use = self.runtime_info_list[service_name][i]['resource_runtime']['cpu_util_use']
                cpu_util_limit = self.runtime_info_list[service_name][i]['resource_runtime']['cpu_util_limit']
                mem_util_use = self.runtime_info_list[service_name][i]['resource_runtime']['mem_util_use']
                mem_util_limit = self.runtime_info_list[service_name][i]['resource_runtime']['mem_util_limit']

                if delay_exec <= delay_constraint:  # 任务执行的时延低于用户约束
                    # 计算资源画像
                    if cpu_util_limit - cpu_util_use >= 0.1:
                        self.runtime_info_list[service_name][i]['resource_runtime']['cpu_portrait'] = 0  # 0表示强，1表示中，2表示弱
                    elif math.fabs(cpu_util_limit - cpu_util_use) < 0.1:
                        self.runtime_info_list[service_name][i]['resource_runtime']['cpu_portrait'] = 1
                    else:
                        root_logger.warning("Compute resource limit is useless, cpu_util_limit:{}, cpu_util_use{}!"
                                            .format(cpu_util_limit, cpu_util_use))
                        self.runtime_info_list[service_name][i]['resource_runtime']['cpu_portrait'] = 1

                    # 存储资源画像
                    if mem_util_limit - mem_util_use >= 0.1:
                        self.runtime_info_list[service_name][i]['resource_runtime']['mem_portrait'] = 0
                    elif math.fabs(mem_util_limit - mem_util_use) < 0.1:
                        self.runtime_info_list[service_name][i]['resource_runtime']['mem_portrait'] = 1
                    else:
                        root_logger.warning("Memory resource limit is useless, mem_util_limit:{}, mem_util_use{}!"
                                            .format(mem_util_limit, mem_util_use))
                        self.runtime_info_list[service_name][i]['resource_runtime']['mem_portrait'] = 1

                else:  # 任务执行的时延高于用户约束
                    if cpu_util_limit - cpu_util_use >= 0.1:
                        self.runtime_info_list[service_name][i]['resource_runtime']['cpu_portrait'] = 0
                    elif math.fabs(cpu_util_limit - cpu_util_use) < 0.1:
                        self.runtime_info_list[service_name][i]['resource_runtime']['cpu_portrait'] = 2
                    else:
                        root_logger.warning("Compute resource limit is useless, cpu_util_limit:{}, cpu_util_use{}!"
                                            .format(cpu_util_limit, cpu_util_use))
                        self.runtime_info_list[service_name][i]['resource_runtime']['cpu_portrait'] = 2

                    # 存储资源画像
                    if mem_util_limit - mem_util_use >= 0.1:
                        self.runtime_info_list[service_name][i]['resource_runtime']['mem_portrait'] = 0
                    elif math.fabs(mem_util_limit - mem_util_use) < 0.1:
                        self.runtime_info_list[service_name][i]['resource_runtime']['mem_portrait'] = 2
                    else:
                        root_logger.warning("Memory resource limit is useless, mem_util_limit:{}, mem_util_use{}!"
                                            .format(mem_util_limit, mem_util_use))
                        self.runtime_info_list[service_name][i]['resource_runtime']['mem_portrait'] = 2

                runtime_desc['runtime_portrait'][service_name].append(self.runtime_info_list[service_name][i])

        self.runtime_info_list = dict()  # 每次获取运行时情境之后清空

        return runtime_desc

