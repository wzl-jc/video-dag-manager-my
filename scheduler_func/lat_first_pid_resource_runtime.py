from logging_utils import root_logger
import pandas as pd
import os
import time
import math
import requests
import json

prev_video_conf = dict()

prev_flow_mapping = dict()

prev_runtime_info = dict()


available_fps = [1, 5, 10, 20, 30]
available_resolution = ["360p", "480p", "720p", "1080p"]
# available_npxpf = [480*360, 858*480, 1280*720, 1920*1080]

lastTime = time.time()


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.dt = dt
        self.previous_error = 0
        self.integral = 0

    def update(self, current_value):
        error = self.setpoint - current_value
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        print(output)
        return output


# 给定flow_map，根据kb获取处理时延
def get_process_delay(resolution=None, flow_map=None):
    sum_delay = 0.0
    for taskname in flow_map:
        pf_filename = 'profile/{}.pf'.format(taskname)
        pf_table = None
        if os.path.exists(pf_filename):
            pf_table = pd.read_table(pf_filename, sep='\t', header=None,
                                     names=['resolution', 'node_role', 'delay'])
        else:
            root_logger.warning("using profile/face_detection.pf for taskname={}".format(taskname))
            pf_table = pd.read_table('profile/face_detection.pf', sep='\t', header=None,
                                     names=['resolution', 'node_role', 'delay'])
        # root_logger.info(pf_table)
        node_role = 'cloud' if flow_map[taskname]['node_role'] == 'cloud' else 'edge'
        pf_table['node_role'] = pf_table['node_role'].astype(str)
        matched_row = pf_table.loc[
            (pf_table['node_role'] == node_role) & \
            (pf_table['resolution'] == resolution)
            ]
        delay = matched_row['delay'].values[0]
        root_logger.info('get profiler delay={} for taskname={} node_role={}'.format(
            delay, taskname, flow_map[taskname]['node_role']
        ))

        sum_delay += delay

    root_logger.info('get sum_delay={} by knowledge base'.format(sum_delay))

    return sum_delay


# TODO：给定flow_map，获取传输时延
def get_transfer_delay(resolution=None, flow_map=None, resource_info=None):
    return 0.0


# 获取总预估的时延
def get_pred_delay(conf_fps=None, cam_fps=None, resolution=None, flow_map=None, resource_info=None):
    # 给定flow_map，
    # resolution vs process_delay：基于kb
    # resolution vs transfer_delay：基于带宽计算
    # fps vs delay：比例关系

    process_sum_delay = get_process_delay(resolution=resolution, flow_map=flow_map)
    transfer_sum_delay = get_transfer_delay(resolution=resolution, flow_map=flow_map, resource_info=resource_info)

    total_delay = (process_sum_delay + transfer_sum_delay) * conf_fps / cam_fps

    return total_delay


# TODO：给定fps和resolution，结合运行时情境，获取预测时延
def get_pred_acc(conf_fps=None, cam_fps=None, resolution=None, runtime_info=None):
    if runtime_info and 'obj_stable' in runtime_info:
        if not runtime_info['obj_stable'] and conf_fps < 20:
            return 0.6
    return 0.9


# ---------------
# ---- 冷启动 ----
def get_flow_map(dag=None, resource_info=None, offload_ptr=None):
    cold_flow_mapping = dict()
    flow = dag["flow"]

    for idx in range(len(flow)):
        taskname = flow[idx]
        if idx <= offload_ptr:
            cold_flow_mapping[taskname] = {
                "model_id": 0,
                "node_role": "host",
                "node_ip": list(resource_info["host"].keys())[0]
            }
        else:
            cold_flow_mapping[taskname] = {
                "model_id": 0,
                "node_role": "cloud",
                "node_ip": list(resource_info["cloud"].keys())[0]
            }

    return cold_flow_mapping


def get_cold_start_plan(
        job_uid=None,
        dag=None,
        resource_info=None,
        user_constraint=None,
):
    assert job_uid, "should provide job_uid"

    global prev_video_conf, prev_flow_mapping
    global available_fps, available_resolution

    # 时延优先策略：算量最小，算力最大
    cold_video_conf = {
        "resolution": "360p",
        "fps": 30,
        # "ntracking": 5,
        "encoder": "JPEG",
    }
    cold_flow_mapping = dict()
    cold_resource_alloc = dict()
    for taskname in dag["flow"]:
        cold_flow_mapping[taskname] = {
            "model_id": 0,
            "node_role": "host",
            "node_ip": list(resource_info["host"].keys())[0]
        }
        cold_resource_alloc[taskname] = {
            "opt_type": 1
        }

    delay_ub = user_constraint["delay"]
    delay_lb = delay_ub
    acc_ub = user_constraint["accuracy"]
    acc_lb = acc_ub

    min_delay_delta = None
    min_acc_delta = None

    # 调度维度：nproc，切分点，fps，resolution
    for fps in available_fps:
        for resol in available_resolution:
            for offload_ptr in range(0, len(dag["flow"])):
                # 枚举所有策略，根据knowledge base预测时延和精度，找出符合用户约束的。
                # 若无法同时满足，优先满足时延要求。尽量满足精度要求（不要求是最优解，所以可以提前退出）
                flow_map = get_flow_map(dag=dag,
                                        resource_info=resource_info,
                                        offload_ptr=offload_ptr)
                cam_fps = 30.0
                delay = get_pred_delay(conf_fps=fps, cam_fps=cam_fps,
                                       resolution=resol,
                                       flow_map=flow_map,
                                       resource_info=resource_info)
                acc = get_pred_acc(conf_fps=fps, cam_fps=cam_fps,
                                   resolution=resol)

                if delay < delay_ub:
                    # 若时延符合要求，找最符合精度要求的
                    # 防止符合要求的配置被替换
                    min_delay_delta = 0.0
                    if not min_acc_delta or min_acc_delta > abs(acc_lb - acc):
                        cold_video_conf["resolution"] = resol
                        cold_video_conf["fps"] = fps
                        cold_flow_mapping = flow_map
                        min_acc_delta = abs(acc_lb - acc)
                else:
                    # 若时延不符合要求，找出尽量符合的
                    if not min_delay_delta or min_delay_delta > abs(delay_ub - delay):
                        cold_video_conf["resolution"] = resol
                        cold_video_conf["fps"] = fps
                        cold_flow_mapping = flow_map
                        min_delay_delta = abs(delay_ub - delay)

    prev_video_conf[job_uid] = cold_video_conf
    prev_flow_mapping[job_uid] = cold_flow_mapping

    return prev_video_conf[job_uid], prev_flow_mapping[job_uid], cold_resource_alloc


# -------------------------------------------
# ---- TODO：根据资源情境，尝试分配更多资源 ----
def try_adjust_service_location(next_flow_mapping=None, err_level=None, resource_info=None):
    # 尝试将任务从边端放到云端做
    tune_msg = None
    for taskname, task_mapping in reversed(list(next_flow_mapping.items())):
        if task_mapping["node_role"] == "host":
            print(" -------- send to cloud --------")
            next_flow_mapping[taskname]["node_role"] = "cloud"
            next_flow_mapping[taskname]["node_ip"] = list(
                resource_info["cloud"].keys())[0]
            tune_msg = "task-{} send to cloud".format(taskname)
            break

    return tune_msg, next_flow_mapping


def try_expand_resource(runtime_resource_info=None):
    # 尝试为工作进程提供更多计算资源
    tune_msg = None
    next_resource_alloc = dict()

    cur_runtime_resource_info = runtime_resource_info['cur_runtime_resource_info']
    pre_runtime_resource_info = None
    if 'pre_runtime_resource_info' in runtime_resource_info:
        pre_runtime_resource_info = runtime_resource_info['pre_runtime_resource_info']

    flag_list = []  # 判断各个子任务是否可以（需要）调整资源分配
    # 0表示不需要调整也可以达到要求；1表示可以尝试调整，下一个调度周期再看效果；2表示调整了也达不到效果
    assert cur_runtime_resource_info
    for task_name in cur_runtime_resource_info:
        if not cur_runtime_resource_info[task_name]['flow_mapping']:
            # 异常情况：某个任务的执行节点为空，可能是由于前面的任务执行的太慢，导致当前任务在调度器访问describe_runtime时此任务还没有执行（尤其是冷启动阶段）
            # 因此，此任务的flow_mapping为空，proc_resource_info_list长度也为0
            # 出现这种情况的原因是前面的任务执行太慢，而无法判断当前任务执行的速度，因此不调整当前任务的资源分配
            print("In try_expand_resource, {}'s flow_mapping is empty!".format(task_name))
            next_resource_alloc[task_name] = dict()
            next_resource_alloc[task_name]['opt_type'] = 1
            flag_list.append(0)
            continue

        proc_resource_info_list = cur_runtime_resource_info[task_name]['proc_resource_info_list']

        if len(proc_resource_info_list) == 0:
            # 异常情况：某个任务的执行节点不为空，但过去一个调度周期内该任务没有执行结果。由于flow_mapping非空，所以该任务之前一定执行过
            # 可能是由于前面的任务执行的太慢，导致在一个调度周期内当前任务还没执行，调度器就访问了describe_runtime
            print("In try_expand_resource, {}'s proc_resource_info_list is empty!".format(task_name))
            next_resource_alloc[task_name] = dict()
            next_resource_alloc[task_name]['opt_type'] = 1
            flag_list.append(0)
            continue

        elif len(proc_resource_info_list) == 1:  # 当前子任务只有一个工作进程执行
            if math.fabs(proc_resource_info_list[0]['cpu_util_limit']-proc_resource_info_list[0]['cpu_util_use']) < 0.1:
                # 此时cpu利用率成为资源瓶颈，增加计算资源
                next_resource_alloc[task_name] = dict()
                next_resource_alloc[task_name]['opt_type'] = 2
                next_resource_alloc[task_name]['proc_resource_limit'] = dict()
                pid_str = str(proc_resource_info_list[0]['pid'])
                next_resource_alloc[task_name]['proc_resource_limit'][pid_str] = dict()
                new_cpu_util_limit = min(1.0, proc_resource_info_list[0]['cpu_util_limit'] * 2)
                next_resource_alloc[task_name]['proc_resource_limit'][pid_str]['cpu_util_limit'] = new_cpu_util_limit

                flag_list.append(1)

            else:
                # 此时cpu利用率不是资源瓶颈，增加工作进程
                next_resource_alloc[task_name] = dict()
                next_resource_alloc[task_name]['opt_type'] = 3

                flag_list.append(1)

        else:
            assert pre_runtime_resource_info is not None
            # TODO：在多进程处理任务仍超时的情况下进一步判断如何操作：
            #              （1）分配更多进程；
            #              （2）减少工作进程数量（同步开销超过并发收益）；

            '''
            cur_task_flow_mapping = cur_runtime_resource_info[task_name]['flow_mapping']
            cur_task_proc_resource = cur_runtime_resource_info[task_name]['proc_resource_info_list']
            pre_task_flow_mapping = pre_runtime_resource_info[task_name]['flow_mapping']
            pre_task_proc_resource = pre_runtime_resource_info[task_name]['proc_resource_info_list']
            if not pre_task_flow_mapping or len(pre_task_proc_resource) == 0:
                # 如果前一段时间的进程资源信息不可用，则无法对比，保持现有资源分配
                next_resource_alloc[task_name] = dict()
                next_resource_alloc[task_name]['opt_type'] = 1
                flag_list.append(0)
                continue
            if pre_task_flow_mapping['node_ip'] != cur_task_flow_mapping['node_ip']:
                # 如果前一段时间的执行节点与当前不同，则无法对比，保持现有资源分配
                next_resource_alloc[task_name] = dict()
                next_resource_alloc[task_name]['opt_type'] = 1
                flag_list.append(0)
                continue
            if len(cur_task_proc_resource) == len(pre_task_proc_resource):
                # 如果前一段时间的工作进程数与当前相同，尝试增加工作进程
                next_resource_alloc[task_name] = dict()
                next_resource_alloc[task_name]['opt_type'] = 3
                flag_list.append(1)
            elif len(cur_task_proc_resource) < len(pre_task_proc_resource):
                # 如果前一段时间的工作进程数比当前多，说明上次调度时考虑到同步开销减少了工作进程，保持现有资源分配，且当前任务无法通过资源优化
                next_resource_alloc[task_name] = dict()
                next_resource_alloc[task_name]['opt_type'] = 1
                flag_list.append(2)
            else:
                # 如果前一段时间的工作进程数比当前少，则需要观察增加工作进程的收益
                pre_max_latency = 0  # 增加工作进程数之前执行任务的时延
                cur_max_latency = 0  # 增加工作进程数之后执行任务的时延
                for proc_info in pre_task_proc_resource:
                    pre_max_latency = max(pre_max_latency, proc_info['latency'])
                for proc_info in cur_task_proc_resource:
                    cur_max_latency = max(cur_max_latency, proc_info['latency'])
                if cur_max_latency - pre_max_latency > 0 or (pre_max_latency - cur_max_latency) < 0.1*pre_max_latency:
                    # 如果增加之后的时延大于增加之前的时延，或者增加之后收益不大（不超过原时延的10%），则减少工作进程，且说明当前任务无法通过资源分配优化
                    request_dict = {
                        "task_name": task_name
                    }
                    headers = {"Content-type": "application/json"}
                    url = "http://{}:5500/decrease_work_process".format(cur_task_flow_mapping['node_ip'])
                    r = requests.post(url, data=json.dumps(request_dict), headers=headers).text
                    r = json.loads(r)
                    next_resource_alloc[task_name] = dict()
                    next_resource_alloc[task_name]['opt_type'] = 4
                    next_resource_alloc[task_name]['node_ip'] = cur_task_flow_mapping['node_ip']
                    flag_list.append(2)
                else:
                    # 如果增加之后有明显的时延收益，则继续增加工作进程
                    next_resource_alloc[task_name] = dict()
                    next_resource_alloc[task_name]['opt_type'] = 3
                    flag_list.append(1)
            '''

            # 简单处理方法：只要多进程无法满足时延要求就认为无法继续进行资源优化
            flag_list.append(2)

    # TODO：根据每个任务各自的时延情况调整各自的资源分配和云边切分：

    # 目前的方式比较粗暴，只要有一个任务无法调整资源，就认为整体无法进行资源分配，开始整体进行边到云的转移，而不是只把不满足的任务进行边到云
    res_flag = True  # 综合所有子任务的结果，判断能否通过调整资源的方式降低时延
    for flag in flag_list:
        if flag == 2:
            res_flag = False
            break

    if res_flag:  # 若可以调整资源，则返回资源调整结果
        tune_msg = "Scheduler adjust resource allocate"
        return tune_msg, next_resource_alloc

    tune_msg = None  # 否则无法调整资源，所有任务保持原有资源分配方式不变，开始调整执行位置从边到云
    for task_name in cur_runtime_resource_info:
        next_resource_alloc[task_name] = dict()
        next_resource_alloc[task_name]['opt_type'] = 1
    return tune_msg, next_resource_alloc


# -----------------------------------------
# ---- TODO：根据应用情境，尝试减少计算量 ----
def try_reduce_calculation(
        next_video_conf=None,
        err_level=None,
        runtime_info=None,
        init_prior=1,
        best_effort=False
):
    global available_fps, available_resolution

    resolution_index = available_resolution.index(
        next_video_conf["resolution"])
    fps_index = available_fps.index(next_video_conf["fps"])

    tune_msg = None

    # TODO：根据运行时情境初始化优先级，实现最佳匹配
    total_prior = 2
    curr_prior = init_prior

    # 无法最佳匹配时，根据收益大小优先级调度
    while True:
        if curr_prior == 1:
            if fps_index > 0:
                print(" -------- fps lower -------- (init_prior={})".format(init_prior))
                next_video_conf["fps"] = available_fps[fps_index - 1]
                tune_msg = "fps {} -> {}".format(available_fps[fps_index],
                                                 available_fps[fps_index - 1])

        if curr_prior == 0:
            if resolution_index > 0:
                print(" -------- resolution lower -------- (init_prior={})".format(init_prior))
                next_video_conf["resolution"] = available_resolution[resolution_index - 1]
                tune_msg = "resolution {} -> {}".format(available_resolution[resolution_index],
                                                        available_resolution[resolution_index - 1])

        # 按优先级依次选择可调的配置
        if best_effort and not tune_msg:
            curr_prior = (curr_prior + 1) % total_prior
            if curr_prior == init_prior:
                break
        if best_effort and tune_msg:
            break
        if not best_effort:
            break

    return tune_msg, next_video_conf


# ----------------
# ---- 负反馈 ----
def adjust_parameters(output=0, job_uid=None,
                      dag=None,
                      user_constraint=None,
                      resource_info=None,
                      runtime_info=None):
    assert job_uid, "should provide job_uid"

    global prev_video_conf, prev_flow_mapping, prev_runtime_info
    global available_fps, available_resolution

    next_video_conf = prev_video_conf[job_uid]
    next_flow_mapping = prev_flow_mapping[job_uid]
    next_resource_alloc = dict()

    # 仅支持pipeline
    flow = dag["flow"]
    assert isinstance(flow, list), "flow not list"

    resolution_index = available_resolution.index(
        next_video_conf["resolution"])
    fps_index = available_fps.index(next_video_conf["fps"])

    err_level = round(output)
    if err_level < -3:
        err_level = -3
    elif err_level > 3:
        err_level = 3

    tune_msg = None

    # TODO：参照对应的边端sniffer解析运行时情境
    print('---- runtime_info in the past time slot ----')
    print('runtime_info = {}'.format(runtime_info))
    # obj_n = runtime_info['obj_n']

    if err_level > 0:
        # level > 0，时延满足要求
        # TODO：结合运行时情境（应用），可以进一步优化其他目标（精度、云端开销等）：
        #              优化目标优先级：时延 > 精度 > 云端开销
        #              若优化目标为最大化精度，在达不到要求时，可以提高fps和resolution；
        #              若优化目标为最小化云端开销，可以拉回到边端计算；
        tune_level = err_level
        pred_acc = get_pred_acc(conf_fps=next_video_conf['fps'], cam_fps=30.0,
                                resolution=next_video_conf["resolution"],
                                runtime_info=runtime_info)

        # 若此时预测精度达不到要求，可以提高fps和resolution
        if pred_acc < user_constraint["accuracy"]:
            # 根据不同程度的 delay-acc trade-off，在不同的delay级别调整不同的参数
            while not tune_msg and tune_level > 0:
                if tune_level == 2:
                    if fps_index + 1 < len(available_fps):
                        print(" -------- fps higher -------- (err_level={}, tune_msg={})".format(err_level, tune_msg))
                        next_video_conf["fps"] = available_fps[fps_index + 1]
                        tune_msg = "fps {} -> {}".format(available_fps[fps_index],
                                                         available_fps[fps_index + 1])

                elif tune_level == 1:
                    if resolution_index + 1 < len(available_resolution):
                        print(" -------- resolution higher -------- (err_level={}, tune_msg={})".format(err_level,
                                                                                                        tune_msg))
                        next_video_conf["resolution"] = available_resolution[resolution_index + 1]
                        tune_msg = "resolution {} -> {}".format(available_resolution[resolution_index],
                                                                available_resolution[resolution_index + 1])

                # 按优先级依次选择可调的配置
                if not tune_msg:
                    tune_level -= 1
        else:
            if 'obj_stable' in runtime_info and runtime_info['obj_stable']:
                # 场景稳定，优先降低帧率
                init_prior = 1
                best_effort = False
                tune_msg, next_video_conf = try_reduce_calculation(next_video_conf=next_video_conf,
                                                                   err_level=err_level,
                                                                   runtime_info=runtime_info,
                                                                   init_prior=init_prior, best_effort=best_effort)

        # 若时延满足要求，则不调整资源分配策略
        for task_name in flow:
            next_resource_alloc[task_name] = {
                'opt_type': 1
            }

    elif err_level < 0:
        # level < 0，时延不满足要求
        # TODO：结合运行时情境（资源），应该调整策略，以降低时延：
        #              （1）分配更多资源；
        #              （2）任务卸载到空闲节点（云/边）；
        #              （3）若场景稳定性，降低帧率；若场景目标较大，降低分辨率
        #              （4）最后考虑降低fps和resolution；
        #       结合运行时情境（应用），调整fps和resolution，比如：
        #              场景稳定则优先降低fps（对精度影响较小）
        #              物体较大则降低resolution（对精度影响较小）
        tune_msg, next_resource_alloc = try_expand_resource(runtime_info['runtime_resource_info'])

        if not tune_msg:
            tune_msg, next_flow_mapping = try_adjust_service_location(next_flow_mapping=next_flow_mapping,
                                                                      err_level=err_level, resource_info=resource_info)

        if not tune_msg:
            if 'obj_stable' in runtime_info and runtime_info['obj_stable']:
                # 场景稳定，优先降低帧率
                init_prior = 1
                best_effort = False
                tune_msg, next_video_conf = try_reduce_calculation(next_video_conf=next_video_conf,
                                                                   err_level=err_level,
                                                                   runtime_info=runtime_info,
                                                                   init_prior=init_prior, best_effort=best_effort)
            elif 'obj_size' in runtime_info and runtime_info['obj_size'] > 500:
                # 场景不稳定，但物体够大，优先降低分辨率
                init_prior = 0
                best_effort = False
                tune_msg, next_video_conf = try_reduce_calculation(next_video_conf=next_video_conf,
                                                                   err_level=err_level,
                                                                   runtime_info=runtime_info,
                                                                   init_prior=init_prior, best_effort=best_effort)

        if not tune_msg:
            # 资源分配完毕，且无法根据情境降低计算量，则按收益大小降低计算量
            init_prior = 1
            best_effort = True
            tune_msg, next_video_conf = try_reduce_calculation(next_video_conf=next_video_conf,
                                                               err_level=err_level,
                                                               runtime_info=runtime_info,
                                                               init_prior=init_prior, best_effort=best_effort)

    prev_video_conf[job_uid] = next_video_conf
    prev_flow_mapping[job_uid] = next_flow_mapping
    prev_runtime_info[job_uid] = runtime_info

    print(prev_flow_mapping[job_uid])
    print(prev_video_conf[job_uid])
    print(prev_runtime_info[job_uid])
    print("Job: {}, next resource_alloc: {}".format(job_uid, next_resource_alloc))
    root_logger.info("tune_msg: {}".format(tune_msg))

    return prev_video_conf[job_uid], prev_flow_mapping[job_uid], next_resource_alloc


# -----------------
# ---- 调度入口 ----
def scheduler(
        job_uid=None,
        dag=None,
        resource_info=None,
        runtime_info=None,
        user_constraint=None,
):
    assert job_uid, "should provide job_uid for scheduler to get prev_plan of job"

    root_logger.info(
        "scheduling for job_uid-{}, runtime_info=\n{}".format(job_uid, runtime_info))

    global lastTime

    if not runtime_info or not user_constraint:  # 如果运行时情境为空，说明任务还未开始执行，进行冷启动
        root_logger.info("to get COLD start executation plan")
        return get_cold_start_plan(
            job_uid=job_uid,
            dag=dag,
            resource_info=resource_info,
            user_constraint=user_constraint
        )

    # ---- 若有负反馈结果，则进行负反馈调节 ----
    global prev_video_conf, prev_flow_mapping

    assert job_uid in prev_video_conf, \
        "job_uid not in prev_video_conf(keys={})".format(
            prev_video_conf.keys())
    assert job_uid in prev_flow_mapping, \
        "job_uid not in prev_video_conf(keys={})".format(
            prev_flow_mapping.keys())

    video_conf = None
    flow_mapping = None

    delay_ub = user_constraint["delay"]
    delay_lb = delay_ub

    # set pidController param
    Kp, Ki, Kd = 1, 0.1, 0.01
    setpoint = delay_ub
    dt = time.time() - lastTime
    pidControl = PIDController(Kp, Ki, Kd, setpoint, dt)

    # TODO：参照对应的边端sniffer解析运行时情境
    print('---- runtime_info in the past time slot ----')
    print('runtime_info = {}'.format(runtime_info))

    avg_delay = runtime_info['delay']
    output = pidControl.update(avg_delay)

    # adjust parameters

    return adjust_parameters(output, job_uid=job_uid,
                             dag=dag,
                             user_constraint=user_constraint,
                             resource_info=resource_info,
                             runtime_info=runtime_info)
