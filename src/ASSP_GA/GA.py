import sys
import copy
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import math
import os
from src.ASSP_GA.objects import *
import palettable
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mpl_toolkits.axisartist as axisartist
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

class Node(object):
	def __init__(self, population, fitness, all_objs, all_tardiness, my_tardiness):
		self.parent = None
		self.children = []
		self.population = population
		self.fitness = fitness
		self.all_objs = all_objs
		self.all_tardiness = all_tardiness
		self.my_tardiness = my_tardiness

	def get_parent(self):
		return self.parent

	def set_parent(self, parent):
		self.parent = parent

	def get_children(self):
		return self.children

	def add_child(self, sub_node):
		sub_node.set_parent(self)
		self.children.append(sub_node)

	def __repr__(self):
		return "Node: {}, fitness: {}".format(hash(self), self.fitness)

class GA():
	def __init__(self, args):
		reader = Reader(args.flightFileName, args.sheetName, args.shuttle_capacity)
		instance = reader.get_instance()
		self.instance = instance
		self.generation = args.maxGen  							# 迭代次数
		self.popSize = args.popSize  							# 种群规模
		self.pr = args.pr  										# 选择率
		self.num_shuttle = len(instance.shuttles)  				# 摆渡车总数
		self.num_flight = len(instance.flights)  				# 航班总数
		self.dist_matrix = instance.dist_matrix					# 距离矩阵
		self.depot_id = instance.depot_id  						# 基地编号
		self.charging_coef = args.coef							# 充电惩罚系数
		self.shuttle_info = dict()  							# 字典，将所有摆渡车按照摆渡车id作为‘key’指示相应摆渡车类
		for shuttle in self.instance.shuttles:
			self.shuttle_info[shuttle.id] = shuttle

		self.flight_info = dict()  								# 字典，将所有航班按照编号1,2,3，即id_2作为‘key’指示相应航班类
		for flight in self.instance.flights:
			self.flight_info[flight.id_2] = flight
			#flight.__info__()                                  # 输出每辆摆渡车的信息

	def initialize(self):
		p=[[] for _ in range(self.num_shuttle)]
		num = int(self.num_flight / self.num_shuttle)
		remainder = self.num_flight % self.num_shuttle         		#不能整除的余量
		all_flights_NO = list(self.flight_info.keys())
		all_shuttles_NO = list(self.shuttle_info.keys())

		np.random.shuffle(all_flights_NO)
		index = 0
		for i in range(self.num_shuttle):
			p[i] = all_flights_NO[index:index+num]
			index += num

		#分配余量航班
		remd_flights = all_flights_NO[index:index+remainder]
		for i in range(len(remd_flights)):
			random_shuttle = np.random.choice(all_shuttles_NO)		# 随机选择一辆摆渡车编号
			p[random_shuttle-1].append(remd_flights[i])				# 加入航班

		#根据航班起飞时刻，调整航班编码顺序
		for i in range(len(p)):
			fligt_time_list = []
			for j in range(len(p[i])):
				fligt_time_list.append(self.flight_info[p[i][j]].get_service_time())
			fligt_time_array = np.array(fligt_time_list)
			sorted_index_array = np.argsort(fligt_time_array)
			sorted_index_list = sorted_index_array.tolist()
			sorted_fligt_time_list = []
			for index in sorted_index_list:
				sorted_fligt_time_list.append(p[i][index])
			p[i] = sorted_fligt_time_list
		return p

	def initialize_in_turn(self):
		# ‘Service_Time_in_Turn'
		# 按照服务时间依次分配车(0-14)
		p = [[] for _ in range(self.num_shuttle)]  # 记录所有分配结果
		# 将航班依次排序
		all_flights_no = list(self.flight_info.keys())
		# 根据航班服务时刻，调整航班编码顺序
		fligt_time_list = [self.flight_info[all_flights_no[i]].get_service_time() for i in range(len(all_flights_no))]
		events = zip(all_flights_no, fligt_time_list)
		flight_events = sorted(events, key=lambda x: x[1])

		for i in range(len(flight_events)):
			p[i % self.num_shuttle].append(flight_events[i][0])
		# 依次将这些航班按顺序放入
		return p

	def my_fifo(self):
		# 'Minimum Completion Time'
		# 每次按最早开始时间分配车，上一任务完成时间最小的车优先被分配
		# 计算所有航班被分配任务后的完成时刻
		# 每次找最小的服务时间，先分配一批。
		# 得到一个p，计算该p的solution，p[-1]即为每辆摆渡车的完成时间
		p = [[] for _ in range(self.num_shuttle)]  # 记录所有分配结果
		CT = [[i, 0] for i in range(self.num_shuttle)]  # 记录每辆摆渡车编号和其完成时间
		# 将航班依次排序
		all_flights_no = list(self.flight_info.keys())
		# 根据航班服务时刻，调整航班编码顺序
		fligt_time_list = [self.flight_info[all_flights_no[i]].get_service_time() for i in range(len(all_flights_no))]
		events = zip(all_flights_no, fligt_time_list)
		flight_events = sorted(events, key=lambda x: x[1])

		while len(flight_events) != 0:
			min_time = flight_events[0][-1]
			flight_list = []  # 记录本次循环需要分配的航班
			index_list = []
			for i in range(len(flight_events)):
				if flight_events[i][-1] == min_time:
					index_list.append(i)
					flight_list.append(flight_events[i][0])

			# 为同一批次航班分配摆渡车，选择按照CT排序的前len(flight_list)辆
			sorted_CT = copy.deepcopy(CT)
			# 判断需分配的航班数量和摆渡车的数量关系
			flight_num = len(flight_list)
			# 如果航班数量大于车，则不够分，将剩余部分移除flight_list,下次再派车
			if flight_num > self.num_shuttle:
				flight_list = flight_list[0:self.num_shuttle]
				index_list = index_list[0:self.num_shuttle]

			assigned_vehicles = sorted(sorted_CT, key=lambda x: x[1])[0:len(flight_list)]

			for i in range(len(assigned_vehicles)):
				p[assigned_vehicles[i][0]].append(flight_list[i])

			index_list.reverse()  # 反转序列，删除已分配航班
			for index in index_list:
				flight_events.pop(index)

			# 更新航班完成时间
			time_list = self.get_gantt_time(p)[0]
			for i in range(len(time_list)):
				if len(time_list[i]) != 0:
					CT[i][1] = time_list[i][-1][-1]

		return p

	def get_sort_p(self, p):
		# 根据航班起飞时刻，调整航班编码顺序
		for i in range(len(p)):
			fligt_time_list = []
			for j in range(len(p[i])):
				fligt_time_list.append(self.flight_info[p[i][j]].get_service_time())
			fligt_time_array = np.array(fligt_time_list)
			sorted_index_array = np.argsort(fligt_time_array)
			sorted_index_list = sorted_index_array.tolist()
			sorted_fligt_time_list = []
			for index in sorted_index_list:
				sorted_fligt_time_list.append(p[i][index])
			p[i] = sorted_fligt_time_list
		return p

	def get_gantt_time(self, p):
		'输入：p是所有摆渡车服务的航班列表，形如:[[1,2,3],[5,6,7]]'
		'输出：所有航班[开始服务时间，进出港类型，第一段行程时间，计划服务时间，预离预到时间，终止服务时间],形如[[(6,0,2,3,5,20),(3,1,4,3,5,50)]]'
		't1:开始服务时间'
		't2:计划服务时间'
		't3:终止服务时间'
		# 't4:摆渡车实际到达时间'
		'travel_time1: 上一节点到当前节点的行程时间'
		'travel_time2: 远机位/出发口——到达口/远机位的行程时间'
		'行程时间均向上取整'
		time_list = [[] for _ in range(len(p))]
		travel_time = [[] for _ in range(len(p))]
		for i in range(len(p)):
			# self.shuttle_info[i+1]是对应的摆渡车类
			for j in range(len(p[i])):
				# self.flight_info[p[i][j]] 是对应的航班类

				flight_type = self.flight_info[p[i][j]].get_arrival()
				flight_id = self.flight_info[p[i][j]].get_id()

				t2 = self.flight_info[p[i][j]].get_service_time() -	self.shuttle_info[i+1].get_pick_up_time()

				last_task_to_depot_time = 0  # 如果赶往休息区，要记录从上一任务结束到休息区的时间

				#判断前一节点是基地？还是其他：远机位/到达口；并在摆渡车的位置列表中添加相应编号
				if j == 0:
					if self.flight_info[p[i][j]].get_arrival() == 1:
						# 当前任务的开始位置
						location = self.depot_id
						# 到港航班：基地到航班远机位的行驶时间
						travel_time1 = math.ceil(self.dist_matrix[self.depot_id-1][self.flight_info[p[i][j]].get_remote_stand_id()-1] / self.shuttle_info[i+1].get_speed() * 60)
						# 更新摆渡车当前任务的终止位置
						self.shuttle_info[i + 1].update_location(self.flight_info[p[i][j]].get_stand_id())
					else:
						location = self.depot_id
						# 出港航班：基地到航班出发口的行驶时间
						travel_time1 = math.ceil(self.dist_matrix[self.depot_id-1][self.flight_info[p[i][j]].get_stand_id()-1] / self.shuttle_info[i+1].get_speed() * 60)
						# 更新摆渡车的位置
						self.shuttle_info[i + 1].update_location(self.flight_info[p[i][j]].get_remote_stand_id())

					t1 = t2 - travel_time1

				else:
					if self.flight_info[p[i][j]].get_arrival() == 1:
						#进港航班
						#time_list[i][-1][-1]表示上一航班的终止服务时间
						#self.shuttle_info[i+1].get_location()选择摆渡车的终止位置
						# location记录当前任务开始位置
						location = self.shuttle_info[i+1].get_location()
						travel_time1 = math.ceil(self.dist_matrix[self.shuttle_info[i+1].get_location()-1][self.flight_info[p[i][j]].get_remote_stand_id()-1]/ self.shuttle_info[i+1].get_speed() * 60)
						t4 = time_list[i][-1][-1] + travel_time1
						if t2 - travel_time1 - time_list[i][-1][-1] > self.instance.rest_time:
							if time_list[i][-1][1] == 0: # 上一航班为出港航班
								# 上一航班的结束位置编号
								if 11 <= self.flight_info[p[i][j-1]].get_remote_stand_id() and self.flight_info[p[i][j-1]].get_remote_stand_id() <= 60:
									# 更新摆渡车的位置
									# 当前任务的开始位置和前一任务的结束位置均需事后变更,而当前任务的结束位置就是航班的结束位置
									# location记录当前任务开始位置
									location = self.instance.A_id
									travel_time1 = math.ceil(self.dist_matrix[location-1][self.flight_info[p[i][j]].get_remote_stand_id()-1]/ self.shuttle_info[i+1].get_speed() * 60)
								if 61 <= self.flight_info[p[i][j-1]].get_remote_stand_id() and self.flight_info[p[i][j-1]].get_remote_stand_id() <= 120:
									# location记录当前任务开始位置
									location = self.instance.B_id
									travel_time1 = math.ceil(self.dist_matrix[location - 1][self.flight_info[p[i][j]].get_remote_stand_id() - 1] / self.shuttle_info[i + 1].get_speed() * 60)

								if 121 <= self.flight_info[p[i][j-1]].get_remote_stand_id() and self.flight_info[p[i][j-1]].get_remote_stand_id() <= 170:
									# location记录当前任务开始位置
									location = self.instance.C_id
									travel_time1 = math.ceil(self.dist_matrix[location - 1][self.flight_info[p[i][j]].get_remote_stand_id() - 1] / self.shuttle_info[i + 1].get_speed() * 60)
								last_task_to_depot_time = math.ceil(self.dist_matrix[location - 1][self.flight_info[p[i][j - 1]].get_remote_stand_id() - 1] /self.shuttle_info[i + 1].get_speed() * 60)
								t4 = time_list[i][-1][-1] + last_task_to_depot_time + travel_time1
							elif time_list[i][-1][1] == 1:
								# location记录当前任务开始位置
								location = self.instance.depot_id
								travel_time1 = math.ceil(self.dist_matrix[location - 1][self.flight_info[p[i][j]].get_remote_stand_id() - 1] / self.shuttle_info[i + 1].get_speed() * 60)
								last_task_to_depot_time = math.ceil(self.dist_matrix[location - 1][self.flight_info[p[i][j-1]].get_stand_id() - 1] / self.shuttle_info[i + 1].get_speed() * 60)
								t4 = time_list[i][-1][-1] + last_task_to_depot_time + travel_time1
						# 更新当前任务，摆渡车的结束位置
						self.shuttle_info[i + 1].update_location(self.flight_info[p[i][j]].get_stand_id())
					else:
						# location记录当前任务开始位置
						location = self.shuttle_info[i + 1].get_location()
						travel_time1 = math.ceil(self.dist_matrix[self.shuttle_info[i + 1].get_location() - 1][self.flight_info[p[i][j]].get_stand_id() - 1] / self.shuttle_info[i + 1].get_speed() * 60)
						t4 = time_list[i][-1][-1] + travel_time1
						if t2 - travel_time1 - time_list[i][-1][-1] > self.instance.rest_time:
							if time_list[i][-1][1] == 0:
								if 11 <= self.flight_info[p[i][j-1]].get_remote_stand_id() and self.flight_info[p[i][j-1]].get_remote_stand_id() <= 60:
									# location记录当前任务开始位置
									location = self.instance.A_id
									travel_time1 = math.ceil(self.dist_matrix[location-1][self.flight_info[p[i][j]].get_stand_id()-1]/ self.shuttle_info[i+1].get_speed() * 60)
								if 61 <= self.flight_info[p[i][j-1]].get_remote_stand_id() and self.flight_info[p[i][j-1]].get_remote_stand_id() <= 120:
									# location记录当前任务开始位置
									location = self.instance.B_id
									travel_time1 = math.ceil(self.dist_matrix[location - 1][self.flight_info[p[i][j]].get_stand_id() - 1] / self.shuttle_info[i + 1].get_speed() * 60)
								if 121 <= self.flight_info[p[i][j-1]].get_remote_stand_id() and self.flight_info[p[i][j-1]].get_remote_stand_id() <= 170:
									# location记录当前任务开始位置
									location = self.instance.C_id
									travel_time1 = math.ceil(self.dist_matrix[location - 1][self.flight_info[p[i][j]].get_stand_id() - 1] / self.shuttle_info[i + 1].get_speed() * 60)
								last_task_to_depot_time = math.ceil(self.dist_matrix[location - 1][self.flight_info[p[i][j - 1]].get_remote_stand_id() - 1] /self.shuttle_info[i + 1].get_speed() * 60)
								t4 = time_list[i][-1][-1] + last_task_to_depot_time + travel_time1
							elif time_list[i][-1][1] == 1:
								# location记录当前任务开始位置
								location = self.instance.depot_id
								travel_time1 = math.ceil(self.dist_matrix[location - 1][self.flight_info[p[i][j]].get_stand_id() - 1] / self.shuttle_info[i + 1].get_speed() * 60)
								last_task_to_depot_time = math.ceil(self.dist_matrix[location - 1][self.flight_info[p[i][j - 1]].get_stand_id() - 1] /self.shuttle_info[i + 1].get_speed() * 60)
								t4 = time_list[i][-1][-1] + last_task_to_depot_time + travel_time1

						# 更新摆渡车的结束位置
						self.shuttle_info[i + 1].update_location(self.flight_info[p[i][j]].get_remote_stand_id())
					if t4 <= t2:
						t1 = t2 - travel_time1
					else:
						t1 = time_list[i][-1][-1]
				# 无论是从远机位到到达口或从出发口到远机位，第二段行程的时间计算是一样的
				travel_time2 = math.ceil(self.dist_matrix[self.flight_info[p[i][j]].get_remote_stand_id() - 1][self.flight_info[p[i][j]].get_stand_id() - 1] / self.shuttle_info[i + 1].get_speed() * 60)
				# print(travel_time)
				t3 = t1 + self.shuttle_info[i+1].get_pick_up_time() + self.shuttle_info[i+1].get_service_time() * 2 + travel_time1+ travel_time2
				time_list[i].append((t1, flight_type, travel_time1, self.flight_info[p[i][j]].get_service_time(), self.flight_info[p[i][j]].get_time(),travel_time2, location, t3))
				travel_time[i].append(travel_time1+travel_time2)
		return time_list, travel_time


	def get_start_end_time(self, solution):
		'获得gantt图上的开始时间和终止时间'
		x_start=sys.maxsize
		for i in range(len(solution)):
			if solution[i][0][0] < x_start:
				x_start = solution[i][0][0]

		x_end = -sys.maxsize
		for i in range(len(solution)):
			if solution[i][-1][-1] > x_end:
				x_end = solution[i][-1][-1]

		return x_start,x_end

	def calculate_tardiness(self, solution):
		all_tardiness = []
		# print('solution = ', solution)
		# print(len(solution))

		for i in range(len(solution)):
			tardiness = []
			for j in range(len(solution[i])):
				if solution[i][j][1] == 1:
					tardiness.append(max(solution[i][j][0]+solution[i][j][2]-solution[i][j][4], 0))
				elif solution[i][j][1] == 0:
					tardiness.append(max(solution[i][j][-1]-solution[i][j][4]+10, 0))
			all_tardiness.append(tardiness)
		return all_tardiness

	def calculate_charging_punishment(self, solution):
		all_punishment = []
		for i in range(len(solution)):
			punishment = []
			threshold = self.shuttle_info[i+1].get_battery_life()			# 续航时间
			recover = self.shuttle_info[i+1].get_charging_time()			# 充电时间
			tempWork = 0													# 连续工作时间
			remainWork = threshold - tempWork								# 剩余可工作时间
			for j in range(len(solution[i])):
				# 一开始的摆渡车是满电的,且不计算空闲时间
				if j == 0:
					tempWork += solution[i][j][-1] - solution[i][j][0]
					remainWork -= solution[i][j][-1] - solution[i][j][0]
					punishment.append(0)
				else:
					# 判断摆渡车是否充电了，如果充电，则连续工作时间tempWork清零，剩余可工作时间remainWork等于续航时间
					tempBreak = solution[i][j][0] - solution[i][j-1][-1]	# 此任务开始时间-上一任务结束时间
					# 当前任务和上一任务的间隔大于充电时间，则表示摆渡车是充满电状态,重新赋值tempWork和remainWork
					if tempBreak > recover:
						tempWork = 0
						remainWork = threshold - tempWork

					#判断摆渡车当前任务长度是否超出剩余工作时间，如果超出则记录惩罚，否则惩罚为0
					if solution[i][j][-1]-solution[i][j][0] - remainWork > 0:
						punishment.append((solution[i][j][-1] - solution[i][j][0] - remainWork) * self.charging_coef)
					else:
						punishment.append(0)
					#无论怎样都更新连续工作时间tempWork，和剩余可工作时间remainWork
					tempWork += solution[i][j][-1]-solution[i][j][0]
					if remainWork - solution[i][j][-1]-solution[i][j][0] <= 0:
						remainWork = 0
					else:
						remainWork = remainWork - solution[i][j][-1] - solution[i][j][0]
			all_punishment.append(punishment)

		return all_punishment

	def calculate_fitness(self, p, solution):
		'输出每辆摆渡车的适应度值，输出是一个摆渡车数量大小的列表'
		all_tardiness = self.calculate_tardiness(solution)
		all_objs = []
		my_tardiness = []  # 个体延迟值
		my_tardiness_obj = []  # 个体延迟值
		all_tardiness_obj = copy.deepcopy(all_tardiness)
		for i in range(len(all_tardiness)):
			for j in range(len(all_tardiness[i])):
				all_tardiness_obj[i][j] = self.flight_info[p[i][j]].get_arrival()*0.2*self.flight_info[p[i][j]].priority * all_tardiness[i][j]  + (1 - self.flight_info[p[i][j]].get_arrival())*0.8*self.flight_info[p[i][j]].priority * all_tardiness[i][j]

		for i in range(len(all_tardiness)):
			sum_tardiness_obj = 0
			sum_tardiness = 0
			for j in range(len(all_tardiness[i])):
				sum_tardiness_obj += self.flight_info[p[i][j]].priority * all_tardiness_obj[i][j]
				sum_tardiness += self.flight_info[p[i][j]].priority * all_tardiness[i][j]
			my_tardiness.append(sum_tardiness)
			my_tardiness_obj.append(sum_tardiness_obj)
			all_objs.append(sum_tardiness_obj)

		return all_objs,my_tardiness_obj,all_tardiness_obj,my_tardiness,all_tardiness

	def visualize(self, p, solution, all_tardiness, path, fileName):
		# print('p',p)
		# print('len(p)', len(p))
		# print('pall_tardiness', all_tardiness)
		# print('len(all_tardiness)', len(all_tardiness))
		colors1 = palettable.cartocolors.sequential.Burg_3.mpl_colors # 进港颜色
		colors2 = palettable.cartocolors.sequential.Teal_3.mpl_colors # 出港颜色

		fig = plt.figure(figsize=(50, 5), dpi=300)  						# 画布设置，大小与分辨率
		ax = fig.add_subplot(111)

		#注：刻度间隔不仅受下式约束，还受到画布大小约束，画布长间隔才能大
		x_major_locator = MultipleLocator(30)  							# x轴刻度间隔设为30
		y_major_locator = MultipleLocator(1)  							# y轴刻度间隔设为1
		ax.xaxis.set_major_locator(x_major_locator)
		ax.yaxis.set_major_locator(y_major_locator)
		plt.ylim(0.5, self.num_shuttle + 0.5)  							# y轴刻度总长为摆渡车数

		plt.xticks(fontsize=10)  										# XY轴刻度标签大小
		plt.yticks(fontsize=10)

		ax.set_xlabel('time', fontsize=15)
		ax.set_ylabel('ferry vehicle', fontsize=15)
		for i in range(len(solution)):
			for j in range(len(solution[i])):
				if all_tardiness[i][j] > 0:
					if solution[i][j][1] == 1:
						myColor = colors1[1]
					else:
						myColor = colors2[1]

				else:
					if solution[i][j][1] == 1:
						myColor = colors1[0]
					else:
						myColor = colors2[0]
				ax.barh(i+1, solution[i][j][-1]-solution[i][j][0], height=0.2, left=solution[i][j][0],
						color=myColor,
						edgecolor="black")
				if j == len(solution[i])-1:
					plt.text(solution[i][j][0] + (solution[i][j][-1]-solution[i][j][0]) / 2, i+1 + 0.1, 'flight'+str(self.flight_info[p[i][j]].id)+'-endloc-depot', ha='center', va='bottom',fontsize=8)
				else:
					plt.text(solution[i][j][0] + (solution[i][j][-1] - solution[i][j][0]) / 2, i + 1 + 0.1,
							 'flight' + str(self.flight_info[p[i][j]].id) + '-endloc'+str(solution[i][j+1][6]), ha='center', va='bottom',
							 fontsize=8)
				# 绘制航班计划服务时间
				plt.axvline(x=self.flight_info[p[i][j]].get_time(), color='red', linestyle='dashed')
				plt.axvline(x=self.flight_info[p[i][j]].get_time(), ymin = 2,ymax = 1, color='blue', linestyle='dashed')
				plt.text(self.flight_info[p[i][j]].get_time(), self.num_shuttle + 0.5, 'F'+str(self.flight_info[p[i][j]].id), ha='center', va='bottom', fontsize=8)

		#替换x轴坐标的表示，写成时刻
		x_start,x_end = self.get_start_end_time(solution)
		all_x_lables = list(range(x_start,(int(x_end/30)+1)*30,30)) # 终止时间要大于x_end,否则range的范围不包括x_end

		#新的时刻标签，如368对应6:08
		new_lables = []
		for x_lable in all_x_lables:
			if x_lable < 0:
				if 60 - x_lable % 60 < 10:
					new_lables.append(str(12 -math.floor(x_lable / 60)) + ':0' + str(60 - x_lable % 60))
				else:
					new_lables.append(str(12 -math.floor(x_lable / 60)) + ':' + str(60 - x_lable % 60))

			else:
				if x_lable % 60 < 10:
					new_lables.append(str(int(x_lable / 60)) + ':0' + str(x_lable % 60))

				else:
					new_lables.append(str(int(x_lable / 60)) + ':' + str(x_lable % 60))
		plt.xticks(all_x_lables, new_lables, rotation=0)  # 替换x轴坐标的表示，写成时刻

		colors = [colors1[0],colors1[1],colors2[0],colors2[1]]
		labels = ['arriving flight', 'tardiness≠0','departing flight', 'tardiness≠0']
		patches = [mpatches.Patch(color=colors[i], label="{:s}".format(labels[i])) for i in range(4)]
		plt.legend(handles=patches, loc='best')
		if not os.path.isdir(path):
			os.makedirs(path)
		plt.savefig(path + "/" + fileName + "-gantt.png")
		plt.clf()

	def visualize_tardiness(self, p, solution, all_tardiness, path, fileName):
		# data需要包含航班编号，进出港0/1，延迟直接写正负
		colors1 = palettable.cartocolors.sequential.Burg_3.mpl_colors  # 进港颜色
		colors2 = palettable.cartocolors.sequential.Teal_3.mpl_colors  # 出港颜色
		x_data = [] #存储航班编号
		y_data = [] #存储航班延迟，正值为出港航班，负值为进港航班
		colors = []
		for i in range(len(all_tardiness)):
			for j in range(len(all_tardiness[i])):
				if all_tardiness[i][j] > 0:
					x_data.append(p[i][j])
					if solution[i][j][1] == 1:
						colors.append(colors1[1])
						y_data.append(-all_tardiness[i][j])
					else:
						colors.append(colors2[1])
						y_data.append(all_tardiness[i][j])
		zip_data = zip(x_data,y_data,colors)
		sorted_data = sorted(zip_data, key = lambda x:x[0])
		x_data = [str(x[0]) for x in sorted_data]
		print('x_data',x_data)
		y_data = [x[1] for x in sorted_data]
		colors = [x[2] for x in sorted_data]
		# 创建画布
		fig = plt.figure(figsize=(5, 5), dpi=300)
		# 使用axisartist.Subplot方法创建一个绘图区对象ax
		ax = axisartist.Subplot(fig, 111)
		# 将绘图区对象添加到画布中
		fig.add_axes(ax)
		ax.axis[:].set_visible(False)  # 通过set_visible方法设置绘图区所有坐标轴隐藏
		ax.axis["x"] = ax.new_floating_axis(0, 0)  # ax.new_floating_axis代表添加新的坐标轴
		ax.axis["x"].set_axisline_style("->", size=1.0)  # 给x坐标轴加上箭头
		# 添加y坐标轴，且加上箭头
		ax.axis["y"] = ax.new_floating_axis(1, -0.3)
		ax.axis["y"].set_axisline_style("-|>", size=1.0)
		# 设置x、y轴上刻度显示方向
		ax.axis["x"].set_axis_direction("top")
		ax.axis["y"].set_axis_direction("left")
		# 设置标签
		# ax.axis["x"].label.set_text("Ferry vehicle No.")
		ax.axis["y"].label.set_text("Tardiness")

		x_major_locator = MultipleLocator(1)  # x轴刻度间隔设为1
		ax.xaxis.set_major_locator(x_major_locator)

		#plt.xlim(0.5, len(x_data)+0.5)
		#plt.ylim(0, max(y_data))

		plt.bar(x_data, y_data,color=colors ,width=0.5)

		new_lables = [str(self.flight_info[int(id2)].get_id()) for id2 in x_data]
		# print('new_lables',new_lables)
		plt.xticks(x_data, new_lables, rotation=0)

		if not os.path.isdir(path):
			os.makedirs(path)
		plt.savefig(path + "/" + fileName + "-each_flight_tardiness.png")
		plt.clf()

	def draw_change(self, all_fitness, all_tardiness, path, fileName):
		plt.figure(dpi=300)
		# print('all_tardiness', all_tardiness)
		# print('all_worktime', all_worktime)
		ln1, = plt.plot(range(len(all_fitness)),all_fitness, c=palettable.cartocolors.sequential.Magenta_7.mpl_colors[-1],linestyle='-')
		ln2, = plt.plot(range(len(all_fitness)),all_tardiness, c=palettable.cartocolors.sequential.Mint_7.mpl_colors[-2], linestyle=':')
		# ln3, = plt.plot(range(len(all_fitness)),all_punishement, c=palettable.cartocolors.sequential.RedOr_3.mpl_colors[-1], linestyle='-.')
		#ln4, = plt.plot(range(len(all_fitness)),all_worktime, c=palettable.cartocolors.sequential.RedOr_3.mpl_colors[-2], linestyle='--')
		# plt.legend(handles=[ln1, ln2, ln3, ln4],
		#            labels = ['fitness', 'tardiness', 'charging_punishement', 'work_time'], loc = 'best')
		plt.legend(handles=[ln1, ln2],
				   labels=['fitness', 'tardiness'], loc='best')
		#作h1函数图像

		if not os.path.isdir(path):
			os.makedirs(path)
		plt.savefig(path + "/" + fileName + "-plot.png")
		plt.clf()

	def get_utilization(self, solution):
		'输出：results,每辆摆渡车的利用率'
		results = np.zeros(self.num_shuttle)  # 存储每辆摆渡车利用率
		results2 = np.zeros(self.num_shuttle)  # 存储每辆摆渡车服务航班数量
		'shuttle_start_end:[[start,end],[start,end]]'
		for i in range(len(solution)):
			# 此时为每架摆渡车上的所有航班的(t1,travel_time1,t2,t3)
			x_start = solution[i][0][0]  # 每辆摆渡车第一架航班的起始时间即为该摆渡车的起始时间
			x_end = solution[i][-1][-1]  # 每辆摆渡车最后一架架航班的结束时间即为该摆渡车的结束时间
			x_intervel = x_end - x_start
			sum_shuttle_intervel = 0
			for j in range(len(solution[i])):
				sum_shuttle_intervel += solution[i][j][-1] - solution[i][j][0]
			results[i] = sum_shuttle_intervel / x_intervel
			results2[i] = len(solution[i])
		return results, results2

	def visualize_utilization(self, utilization, my_worktime, num_shuttle, path, fileName):
		# 画图，plt.bar()可以画柱状图
		x_data = [str(i+1) for i in range(len(utilization))]
		plt.rcParams['xtick.labelsize'] = 10
		plt.rcParams['ytick.labelsize'] = 10
		fig = plt.figure(figsize=(5, 4), dpi=300)  # 画布设置，大小与分辨率
		ax = fig.add_subplot(111)
		# 注：刻度间隔不仅受下式约束，还受到画布大小约束，画布长间隔才能大
		x_major_locator = MultipleLocator(1)  # x轴刻度间隔设为1
		# ax.xaxis.set_major_locator(x_major_locator)

		plt.rcParams['ytick.direction'] = 'in'
		ax.yaxis.tick_right()

		font = {'family': 'Times New Roman','weight': 'normal','size': 12,}

		ax.set_xlabel('Ferry vehicle No.', font)
		ax.set_ylabel('Utilization rate', font)


		norm = plt.Normalize(0, 20)
		norm_values = norm(num_shuttle)
		map_vir = cm.get_cmap(name='summer')
		print('num_shuttle',num_shuttle)
		print('utilization', utilization)
		colors = map_vir(norm_values)

		ax.bar(x_data, utilization,width=0.5,color=colors,edgecolor='black')

		mean_utilization = sum(utilization)/len(utilization)
		mean_utilization_plot = [mean_utilization for _ in range(len(utilization))]
		ax.plot(x_data, mean_utilization_plot, color='red', linewidth = 1)

		def to_percent(temp, position):
			return '%1.0f' % (100 * temp) + '%'

		plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
		# 标注服务的航班架次
		x_locations = list(range(len(utilization)))
		for i in range(len(num_shuttle)):
			ax.text(x_locations[i], utilization[i], int(num_shuttle[i]), ha='center', va='bottom', fontsize=8)


		ax2 = ax.twinx()  # 创建共用x轴的第二个y轴
		ax2.set_ylabel('Total travel time', font)
		ax2.plot(x_data, my_worktime, color='blue', linewidth = 1, marker='+')

		# 设置坐标刻度值的大小以及刻度值的字体
		plt.tick_params(labelsize=10)
		labels = ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_yticklabels()
		[label.set_fontname('Times New Roman') for label in labels]


		# norm = plt.Normalize(0, 20)
		# sm = cm.ScalarMappable(cmap=map_vir, norm=norm)  # norm设置最大最小值
		# sm.set_array([])
		# plt.colorbar(sm)
		if not os.path.isdir(path):
			os.makedirs(path)
		plt.savefig(path + "/" + fileName + "-utilization.png")
		plt.clf()



	def move_random(self, p):
		new_p = []
		copy_p = copy.deepcopy(p)
		#print(p)
		for i in range(self.num_shuttle):
			for j in range(len(p[i])):
				index_list = list(range(self.num_shuttle))
				index_list.remove(i)
				for k in index_list:
					p[k].append(p[i][j])
					p[i].remove(p[i][j])
					new_p.append(p)
					p = copy.deepcopy(copy_p)
		new_p = [self.get_sort_p(new_p[i]) for i in range(len(new_p))]
		return new_p

	def move_value(self, p, fitness):
		new_p = []
		copy_p = copy.deepcopy(p)
		i = fitness.index(max(fitness))
		#print('i',i)
		for j in range(len(p[i])):
			index_list = list(range(self.num_shuttle))
			index_list.remove(i)
			for k in index_list:
				p[k].append(p[i][j])
				p[i].remove(p[i][j])
				new_p.append(p)
				p = copy.deepcopy(copy_p)

		new_p = [self.get_sort_p(new_p[i]) for i in range(len(new_p))]
		return new_p

	def swap(self,p):
		new_p = []
		copy_p = copy.deepcopy(p)
		for i in range(self.num_shuttle):
			for j in range(len(p[i])):
				index_list = list(range(self.num_shuttle))
				index_list.remove(i)
				for k in index_list:
					for l in range(len(p[k])):
						p[k].append(p[i][j])
						p[i].append(p[k][l])
						p[i].remove(p[i][j])
						p[k].remove(p[k][l])
						new_p.append(p)
						p = copy.deepcopy(copy_p)
		return new_p

	def localSearch(self, p, x, action, all_tardiness, my_tardiness):

		def move1(p, x, my_tardiness, all_tardiness): #移动违法最大的摆渡车上违法从大到小的前x架航班
			new_p = []
			copy_p = copy.deepcopy(p)
			i = my_tardiness.index(max(my_tardiness))

			index_list = list(range(self.num_shuttle))
			index_list.remove(i)
			a = np.array(all_tardiness[i])
			move_list = np.argsort(-a)
			move_list = move_list[:x]

			for j in index_list:
				for k in move_list:
					p[j].append(p[i][k])
					p[i].remove(p[i][k])

					new_p.append(p)
					p = copy.deepcopy(copy_p)
			new_p = [self.get_sort_p(new_p[i]) for i in range(len(new_p))]
			return new_p

		def move2(p, all_tardiness):  # 移动违法最大的航班
			new_p = []
			copy_p = copy.deepcopy(p)

			max_tardiness = 0
			for i in range(len(all_tardiness)):
				for j in range(len(all_tardiness[i])):
					if all_tardiness[i][j] > max_tardiness:
						max_tardiness = all_tardiness[i][j]


			index1 = 0
			index2 = 0
			for x in range(0, len(all_tardiness)):
				try:
					index2 = all_tardiness[x].index(max_tardiness)
					break
				except:
					pass
			index1 = x
			index_list = list(range(self.num_shuttle))
			index_list.remove(index1)

			for j in index_list:
				p[j].append(p[index1][index2])
				p[index1].remove(p[index1][index2])
				new_p.append(p)
				p = copy.deepcopy(copy_p)
				new_p = [self.get_sort_p(new_p[i]) for i in range(len(new_p))]

			return new_p


		def move3(p, all_tardiness):  # 移动违法第二大的航班
			new_p = []
			copy_p = copy.deepcopy(p)
			all_tardiness_sort = []
			for i in range(len(all_tardiness)):
				for j in range(len(all_tardiness[i])):
					all_tardiness_sort.append(all_tardiness[i][j])
			all_tardiness_sort.sort(reverse=True)

			# index_i1 = 0
			# index_j1 = 0
			# for m in range(0, len(all_tardiness)):
			# 	try:
			# 		index_j1 = all_tardiness[m].index(all_tardiness_sort[0])
			# 		break
			# 	except:
			# 		pass
			# index_i1 = m

			index_i2 = 0
			index_j2 = 0
			for n in range(0, len(all_tardiness)):
				try:
					index_j2 = all_tardiness[n].index(all_tardiness_sort[1])
					break
				except:
					pass
			index_i2 = n

			# index_list1 = list(range(self.num_shuttle))
			# index_list1.remove(index_i1)

			index_list2 = list(range(self.num_shuttle))
			index_list2.remove(index_i2)

			# for j in index_list1:
			# 	p[j].append(p[index_i1][index_j1])
			# 	p[index_i1].remove(p[index_i1][index_j1])
			# 	new_p.append(p)
			# 	p = copy.deepcopy(copy_p)

			for k in index_list2:
				p[k].append(p[index_i2][index_j2])
				p[index_i2].remove(p[index_i2][index_j2])
				new_p.append(p)
				p = copy.deepcopy(copy_p)
			new_p = [self.get_sort_p(new_p[i]) for i in range(len(new_p))]
			return new_p


		def move4(p, all_tardiness):  # 移动违法第四大的航班
			new_p = []
			copy_p = copy.deepcopy(p)
			all_tardiness_sort = []
			for i in range(len(all_tardiness)):
				for j in range(len(all_tardiness[i])):
					all_tardiness_sort.append(all_tardiness[i][j])
			all_tardiness_sort.sort(reverse=True)

			index_i3 = 0
			index_j3 = 0
			for n in range(0, len(all_tardiness)):
				try:
					index_j3 = all_tardiness[n].index(all_tardiness_sort[3])
					break
				except:
					pass
			index_i3 = n

			index_list3 = list(range(self.num_shuttle))
			index_list3.remove(index_i3)

			for k in index_list3:
				p[k].append(p[index_i3][index_j3])
				p[index_i3].remove(p[index_i3][index_j3])
				new_p.append(p)
				p = copy.deepcopy(copy_p)
			new_p = [self.get_sort_p(new_p[i]) for i in range(len(new_p))]
			return new_p


		def move_value(p, my_tardiness):   #遍历移动违法最大的摆渡车上的所有航班
			new_p = []
			copy_p = copy.deepcopy(p)
			i = my_tardiness.index(max(my_tardiness))
			for j in range(len(p[i])):
				index_list = list(range(self.num_shuttle))
				index_list.remove(i)
				for k in index_list:
					p[k].append(p[i][j])
					p[i].remove(p[i][j])
					new_p.append(p)
					p = copy.deepcopy(copy_p)
			new_p = [self.get_sort_p(new_p[i]) for i in range(len(new_p))]
			return new_p

		def move0(p, all_tardiness):  # 遍历移动违法的所有航班
			new_p = []
			copy_p = copy.deepcopy(p)
			delay_flight_i = []
			delay_flight_j = []
			for i in range(len(p)):
				for j in range(len(p[i])):
					if all_tardiness[i][j] != 0:
						delay_flight_i.append(i)
						delay_flight_j.append(j)

			for n in range(len(delay_flight_i)):
				index_list = list(range(self.num_shuttle))
				index_list.remove(delay_flight_i[n])
				for m in index_list:
					p[m].append(p[delay_flight_i[n]][delay_flight_j[n]])
					p[delay_flight_i[n]].remove(p[delay_flight_i[n]][delay_flight_j[n]])
					new_p.append(p)
					p = copy.deepcopy(copy_p)
			new_p = [self.get_sort_p(new_p[i]) for i in range(len(new_p))]
			return new_p


		def move_random(p):
			new_p = []
			copy_p = copy.deepcopy(p)
			# print(p)
			for i in range(self.num_shuttle):
				for j in range(len(p[i])):
					index_list = list(range(self.num_shuttle))
					index_list.remove(i)
					for k in index_list:
						p[k].append(p[i][j])
						p[i].remove(p[i][j])
						new_p.append(p)
						p = copy.deepcopy(copy_p)
			new_p = [self.get_sort_p(new_p[i]) for i in range(len(new_p))]
			return new_p


		def swap(p):
			new_p = []
			copy_p = copy.deepcopy(p)
			for i in range(self.num_shuttle):
				for j in range(len(p[i])):
					index_list = list(range(num_shuttle))
					index_list.remove(i)
					for k in index_list:
						for l in range(len(p[k])):
							p[k].append(p[i][j])
							p[i].append(p[k][l])
							p[i].remove(p[i][j])
							p[k].remove(p[k][l])
							new_p.append(p)
							p = copy.deepcopy(copy_p)
			new_p = [self.get_sort_p(new_p[i]) for i in range(len(new_p))]
			return new_p

		if action == 0:
			new_p = move1(p, x, my_tardiness, all_tardiness)
		elif action == 1:
			new_p = move2(p, all_tardiness)
		elif action == 2:
			new_p = move3(p, all_tardiness)
		elif action == 3:
			new_p = move0(p, all_tardiness)
			# new_p = move_value(p, my_tardiness)
		elif action == 4:
			new_p = move_value(p, my_tardiness)
		# 	new_p = move4(p, all_tardiness)
			# new_p = move_random(p)
		return new_p

	def select(self, num_node, node_list, neighbors_fitness):
		'返回被选中的节点类'
		sorted_ids = list(np.argsort(neighbors_fitness))
		select_sorted_ids = sorted_ids[0:num_node]
		array_node_list = np.array(node_list)
		select_nodes = array_node_list[select_sorted_ids].tolist()
		return select_nodes

	def treeSearch(self, p, fitness, all_objs, num_node, depth, my_tardiness, all_tardiness, action):
		best_p = copy.deepcopy(p)
		#first_p = copy.deepcopy(p)
		first_fitness = copy.deepcopy(fitness)
		best_obj = copy.deepcopy(fitness)
		my_depth = 0
		x = 1

		while my_depth < depth:
			if my_depth <= 0:
				current_node = Node(p, fitness, all_objs, all_tardiness,my_tardiness)
				# neighbors_p = self.move_value(p, all_objs)
				neighbors_p = self.localSearch(p, x, action, all_tardiness, my_tardiness)
				if len(neighbors_p) == 0:
					break
				# print(action)
				solutions = [[] for _ in range(len(neighbors_p))]
				work_time = [[] for _ in range(len(neighbors_p))]
				for i in range(len(neighbors_p)):
					solutions[i], work_time[i] = self.get_gantt_time(neighbors_p[i])
				objs = [self.calculate_fitness(neighbors_p[i], solutions[i])[0] for i in range(len(neighbors_p))]
				all_tardiness_all = [self.calculate_fitness(neighbors_p[i], solutions[i])[2] for i in range(len(neighbors_p))]
				my_tardiness_my = [self.calculate_fitness(neighbors_p[i], solutions[i])[1] for i in range(len(neighbors_p))]
				#my_worktime_my = [self.calculate_fitness(neighbors_p[i], solutions[i])[6] for i in range(len(neighbors_p))]
				obj = [sum(objs[i]) for i in range(len(objs))]
				for i in range(len(neighbors_p)):
					sub_node = Node(neighbors_p[i], obj[i], objs[i], all_tardiness_all[i], my_tardiness_my[i])
					current_node.add_child(sub_node)
				node_list = current_node.get_children()
				min_obj = min(obj)
				#min_p = neighbors_p[obj.index(min_obj)]
				if min_obj < best_obj:
					min_index_list = []
					for i in range(len(obj)):
						if min_obj == obj[i]:
							min_index_list.append(i)
					best_p = neighbors_p[np.random.choice(min_index_list)]
					best_obj = min_obj

			else:
				select_nodes = self.select(num_node, node_list, obj)
				node_list = []
				for node in select_nodes:
					current_node = node
					neighbors_p = self.localSearch(current_node.population, x, action, current_node.all_tardiness, current_node.my_tardiness)
					solutions = [[] for _ in range(len(neighbors_p))]
					work_time = [[] for _ in range(len(neighbors_p))]
					for i in range(len(neighbors_p)):
						solutions[i], work_time[i] = self.get_gantt_time(neighbors_p[i])

					objs = [self.calculate_fitness(neighbors_p[i], solutions[i])[0] for i in range(len(neighbors_p))]
					all_tardiness_all = [self.calculate_fitness(neighbors_p[i], solutions[i])[2] for i in range(len(neighbors_p))]
					my_tardiness_my = [self.calculate_fitness(neighbors_p[i], solutions[i])[1] for i in range(len(neighbors_p))]
					obj = [sum(objs[i]) for i in range(len(objs))]
					min_obj = min(obj)
					min_p = neighbors_p[obj.index(min_obj)]
					if min_obj < best_obj:
						best_p = min_p
						best_obj = min_obj
					for i in range(len(neighbors_p)):
						sub_node = Node(neighbors_p[i], obj[i], objs[i], all_tardiness_all[i], my_tardiness_my[i])
						current_node.add_child(sub_node)
						# print('#####################################执行替换################################')
					node_list += current_node.get_children()
			my_depth += 1
		return best_p,first_fitness

	def ga_total(self, args):
		all_fitness_list = []										# 存储每一代的适应度值，用于后面绘制曲线图
		all_tardiness_list = []
		#all_punishement_list = []
		#all_worktime_list = []
		action = 0
		flag = 0
		previous_action = -1
		for gen in range(args.maxGen):
			if gen < 1:
				'''种群初始化操作'''
				if args.init_action == 'random':
					p = self.initialize()
				elif args.init_action == 'in_turn':
					p = self.initialize_in_turn()
				elif args.init_action == 'fifo':
					print('进入fifo')
					p = self.my_fifo()
				else:
					print('没有正确输入初始化动作')
					print(args.init_action)
					print(type(args.init_action))
				solution, work_time = self.get_gantt_time(p)  				# 保留,用于绘制甘特图
				all_objs, my_tardiness_obj, all_tardiness_obj, my_tardiness, all_tardiness = self.calculate_fitness(p, solution)  # all_objs是列表
				fitness = sum(all_objs)
				best_p = copy.deepcopy(p)
				best_solution = copy.deepcopy(solution)
				best_fitness = fitness
				best_all_tardiness = all_tardiness
				best_tardiness = sum(my_tardiness)
				best_all_tardiness_obj = all_tardiness_obj
				best_tardiness_obj = sum(my_tardiness_obj)
				all_fitness_list.append(best_fitness)
				all_tardiness_list.append(best_tardiness)

			else:
				# 选择进行迭代的下一个个体
				p,first_fitness = self.treeSearch(p, fitness, all_objs, 1, 1, my_tardiness_obj, all_tardiness_obj, action)
				solution, work_time = self.get_gantt_time(p)
				all_objs,my_tardiness_obj, all_tardiness_obj, my_tardiness, all_tardiness= self.calculate_fitness(p, solution)
				fitness = sum(all_objs)

				if fitness < best_fitness:
					best_p = copy.deepcopy(p)
					best_solution = copy.deepcopy(solution)
					best_fitness = fitness
					best_all_tardiness = all_tardiness
					best_tardiness = sum(my_tardiness)
					best_all_tardiness_obj = all_tardiness_obj
					best_tardiness_obj = sum(my_tardiness_obj)

				if fitness == first_fitness:
					previous_action = action
					flag = flag + 1
					action = flag % 5

				# action = 1 - action

			all_fitness_list.append(best_fitness)
			all_tardiness_list.append(best_tardiness)

			print('Iter = ', gen, 'Tardiness = ', best_tardiness ,'fitness = ', best_fitness, 'action = ', action)
			if best_fitness == 0 or previous_action-action == 4:
				break

		return best_p,best_solution,best_fitness, best_tardiness, best_all_tardiness, all_fitness_list,all_tardiness_list

	def getWorkOrderExcel(self, p, solution, all_tardiness, path, filename):
		'输入:p,每辆摆渡车服务的航班号,形如:[[32,5,33,45],[4,3,23]]'
		results = [[] for _ in range(self.num_shuttle)]   	#存储每辆摆渡车上每架航班航班序号、预到时间、预离时间等等
		results1 = []	#按照航班编号（即时间顺序）记录每架航班航班序号、预到时间、预离时间等等
		shuttle_flight_id = [[] for _ in range(len(p))]
		for i in range(len(p)):
			for j in range(len(p[i])):
				shuttle_flight_id[i].append(self.flight_info[p[i][j]].get_id())

		for i in range(len(p)):
			for j in range(len(p[i])):
				flight_results = []
				flight_results.append(self.flight_info[p[i][j]].get_id())
				if self.flight_info[p[i][j]].get_arrival() == 1:
					flight_results.append(self.flight_info[p[i][j]].get_std_time())
					flight_results.append('—')
				else:
					flight_results.append('—')
					flight_results.append(self.flight_info[p[i][j]].get_std_time())
				flight_results.append(self.flight_info[p[i][j]].get_remote_stand_id())
				flight_results.append(self.flight_info[p[i][j]].get_stand_id())
				if j == 0:
					if self.flight_info[p[i][j]].get_arrival() == 1:
						# 判断是否回休息区,当前任务的结束位置是否是下一任务的开始位置
						if self.flight_info[p[i][j]].get_stand_id() == solution[i][j+1][6]:
							flight_results.append('基地—' + '远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id()) + '—到达口' +str(self.flight_info[p[i][j]].get_stand_id()))
						else:
							flight_results.append('基地—' + '远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id()) + '—到达口' +str(self.flight_info[p[i][j]].get_stand_id())+'-基地'+str(solution[i][j+1][6]))
					else:
						if self.flight_info[p[i][j]].get_remote_stand_id() == solution[i][j+1][6]:
							flight_results.append('基地—' + '登机口' + str(self.flight_info[p[i][j]].get_stand_id()) + '—' + '远机位' +str(self.flight_info[p[i][j]].get_remote_stand_id()))
						else:
							flight_results.append('基地—' + '登机口' + str(self.flight_info[p[i][j]].get_stand_id()) + '—' + '远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id())+'-休息区'+str(solution[i][j+1][6]))
				elif j == len(p[i]) -1:
					if self.flight_info[p[i][j]].get_arrival() == 1:
						if self.flight_info[p[i][j-1]].get_arrival() == 1:
							# 最后一个任务一定回休息区
							# 判断当前任务的开始位置是否是上一任务的结束位置
							if solution[i][j][6] == self.flight_info[p[i][j - 1]].get_stand_id():
								flight_results.append('到达口' + str(self.flight_info[p[i][j-1]].get_stand_id()) +'—' + '远机位' +str(self.flight_info[p[i][j]].get_remote_stand_id()) + '—到达口' + str(self.flight_info[p[i][j]].get_stand_id())+'-基地')
							else:
								flight_results.append('基地' + str(solution[i][j][6]) +'—' + '远机位' +str(self.flight_info[p[i][j]].get_remote_stand_id()) + '—到达口' + str(self.flight_info[p[i][j]].get_stand_id())+'-基地')
						else:
							if solution[i][j][6] == self.flight_info[p[i][j - 1]].get_remote_stand_id():
								flight_results.append('远机位' + str(self.flight_info[p[i][j-1]].get_remote_stand_id()) +'—' + '远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id()) + '—到达口' + str(self.flight_info[p[i][j]].get_stand_id())+'-基地')
							else:
								flight_results.append('休息区' + str(solution[i][j][6]) +'—' + '远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id()) + '—到达口' + str(self.flight_info[p[i][j]].get_stand_id())+'-基地')
					else:
						if self.flight_info[p[i][j-1]].get_arrival() == 1:
							# 判断当前任务的开始位置是否是上一任务的结束位置
							if solution[i][j][6] == self.flight_info[p[i][j - 1]].get_stand_id():
								flight_results.append('到达口' + str(self.flight_info[p[i][j-1]].get_stand_id()) +'—' + '登机口' + str(self.flight_info[p[i][j]].get_stand_id()) + '—远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id())+'-基地')
							else:
								flight_results.append('基地' + str(solution[i][j][6]) +'—' + '登机口' + str(self.flight_info[p[i][j]].get_stand_id()) + '—远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id())+'-基地')
						else:
							# 判断当前任务的开始位置是否是上一任务的结束位置
							if solution[i][j][6] == self.flight_info[p[i][j - 1]].get_remote_stand_id():
								flight_results.append('远机位' + str(self.flight_info[p[i][j-1]].get_remote_stand_id()) +'—' + '登机口' + str(self.flight_info[p[i][j]].get_stand_id()) + '—远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id())+'-基地')
							else:
								flight_results.append('休息区' + str(solution[i][j][6]) +'—' + '登机口' + str(self.flight_info[p[i][j]].get_stand_id()) + '—远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id())+'-基地')
				else:
					if self.flight_info[p[i][j]].get_arrival() == 1:
						if self.flight_info[p[i][j-1]].get_arrival() == 1:
							# 判断是否回休息区,当前任务的结束位置是否是下一任务的开始位置，如果不是需再加一段行程。
							if self.flight_info[p[i][j]].get_stand_id() == solution[i][j + 1][6]:
								# 判断当前任务的开始位置是否是上一任务的结束位置
								if solution[i][j][6] == self.flight_info[p[i][j-1]].get_stand_id():
									flight_results.append('到达口' + str(self.flight_info[p[i][j-1]].get_stand_id()) +'—' + '远机位' +str(self.flight_info[p[i][j]].get_remote_stand_id()) + '—到达口' + str(self.flight_info[p[i][j]].get_stand_id()))
								else:
									flight_results.append('基地' + str(solution[i][j][6]) +'—' + '远机位' +str(self.flight_info[p[i][j]].get_remote_stand_id()) + '—到达口' + str(self.flight_info[p[i][j]].get_stand_id()))
							else:
								if solution[i][j][6] == self.flight_info[p[i][j - 1]].get_stand_id():
									flight_results.append('到达口' + str(self.flight_info[p[i][j-1]].get_stand_id()) +'—' + '远机位' +str(self.flight_info[p[i][j]].get_remote_stand_id()) + '—到达口' + str(self.flight_info[p[i][j]].get_stand_id())+'-基地'+str(solution[i][j+1][6]))
								else:
									flight_results.append('基地' + str(solution[i][j][6]) +'—' + '远机位' +str(self.flight_info[p[i][j]].get_remote_stand_id()) + '—到达口' + str(self.flight_info[p[i][j]].get_stand_id())+'-基地'+str(solution[i][j+1][6]))
						else:
							# 判断是否回休息区,当前任务的结束位置是否是下一任务的开始位置，如果不是需再加一段行程。
							if self.flight_info[p[i][j]].get_stand_id() == solution[i][j + 1][6]:
								# 判断当前任务的开始位置是否是上一任务的结束位置
								if solution[i][j][6] == self.flight_info[p[i][j - 1]].get_remote_stand_id():
									flight_results.append('远机位' + str(self.flight_info[p[i][j-1]].get_remote_stand_id()) +'—' + '远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id()) + '—到达口' + str(self.flight_info[p[i][j]].get_stand_id()))
								else:
									flight_results.append('休息区' + str(solution[i][j][6]) +'—' + '远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id()) + '—到达口' + str(self.flight_info[p[i][j]].get_stand_id()))
							else:
								if solution[i][j][6] == self.flight_info[p[i][j - 1]].get_remote_stand_id():
									flight_results.append('远机位' + str(self.flight_info[p[i][j-1]].get_remote_stand_id()) +'—' + '远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id()) + '—到达口' + str(self.flight_info[p[i][j]].get_stand_id())+'-基地'+str(solution[i][j+1][6]))
								else:
									flight_results.append('休息区' + str(solution[i][j][6]) +'—' + '远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id()) + '—到达口' + str(self.flight_info[p[i][j]].get_stand_id())+'-基地'+str(solution[i][j+1][6]))
					else:
						if self.flight_info[p[i][j-1]].get_arrival() == 1:
							# 判断是否回休息区,当前任务的结束位置是否是下一任务的开始位置，如果不是需再加一段行程。
							if self.flight_info[p[i][j]].get_remote_stand_id() == solution[i][j + 1][6]:
								# 判断当前任务的开始位置是否是上一任务的结束位置
								if solution[i][j][6] == self.flight_info[p[i][j - 1]].get_stand_id():
									flight_results.append('到达口' + str(self.flight_info[p[i][j-1]].get_stand_id()) +'—' + '登机口' + str(self.flight_info[p[i][j]].get_stand_id()) + '—远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id()))
								else:
									flight_results.append('基地' + str(solution[i][j][6]) +'—' + '登机口' + str(self.flight_info[p[i][j]].get_stand_id()) + '—远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id()))
							else:
								# 判断当前任务的开始位置是否是上一任务的结束位置
								if solution[i][j][6] == self.flight_info[p[i][j - 1]].get_stand_id():
									flight_results.append('到达口' + str(self.flight_info[p[i][j-1]].get_stand_id()) +'—' + '登机口' + str(self.flight_info[p[i][j]].get_stand_id()) + '—远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id())+'-休息区'+str(solution[i][j+1][6]))
								else:
									flight_results.append('基地' + str(solution[i][j][6]) +'—' + '登机口' + str(self.flight_info[p[i][j]].get_stand_id()) + '—远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id())+'-休息区'+str(solution[i][j+1][6]))
						else:
							# 判断是否回休息区,当前任务的结束位置是否是下一任务的开始位置，如果不是需再加一段行程。
							if self.flight_info[p[i][j]].get_remote_stand_id() == solution[i][j + 1][6]:
								# 判断当前任务的开始位置是否是上一任务的结束位置
								if solution[i][j][6] == self.flight_info[p[i][j - 1]].get_remote_stand_id():
									flight_results.append('远机位' + str(self.flight_info[p[i][j-1]].get_remote_stand_id()) +'—' + '登机口' + str(self.flight_info[p[i][j]].get_stand_id()) + '—远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id()))
								else:
									flight_results.append('休息区' + str(solution[i][j][6]) +'—' + '登机口' + str(self.flight_info[p[i][j]].get_stand_id()) + '—远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id()))
							else:
								# 判断当前任务的开始位置是否是上一任务的结束位置
								if solution[i][j][6] == self.flight_info[p[i][j - 1]].get_remote_stand_id():
									flight_results.append('远机位' + str(self.flight_info[p[i][j-1]].get_remote_stand_id()) +'—' + '登机口' + str(self.flight_info[p[i][j]].get_stand_id()) + '—远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id())+'-休息区'+str(solution[i][j+1][6]))
								else:
									flight_results.append('休息区' + str(solution[i][j][6]) +'—' + '登机口' + str(self.flight_info[p[i][j]].get_stand_id()) + '—远机位' + str(self.flight_info[p[i][j]].get_remote_stand_id())+'-休息区'+str(solution[i][j+1][6]))
				if solution[i][j][0] % 60 < 10:
					flight_results.append(str(int(solution[i][j][0] / 60)) + ':0' + str(solution[i][j][0] % 60))

				else:
					flight_results.append(str(int(solution[i][j][0] / 60)) + ':' + str(solution[i][j][0] % 60))

				flight_results.append(str(int(solution[i][j][2]))) #添加第一段行程时间
				flight_results.append(str(int(solution[i][j][5]))) #添加第二段行程时间

				if solution[i][j][-1] % 60 < 10:
					flight_results.append(str(int(solution[i][j][-1] / 60)) + ':0' + str(solution[i][j][-1] % 60))

				else:
					flight_results.append(str(int(solution[i][j][-1] / 60)) + ':' + str(solution[i][j][-1] % 60))


				if j < len(p[i])-1:
					flight_results.append(solution[i][j+1][0] - solution[i][j][-1])
				else:
					flight_results.append(0)
				flight_results.append(all_tardiness[i][j])
				results[i].append(flight_results)
		# print('results',results)
		for i in range(len(results)):
			for j in range(len(results[i])):
				results1.append(results[i][j][0:6] + [i] + results[i][j][6:])
		results1 = sorted(results1)
		workbook = xlsxwriter.Workbook(path+'/'+filename)  # 建立文件并命名
		'''写入总表'''
		worksheet = workbook.add_worksheet('总表')
		worksheet.write(0, 0, '航班序号')
		worksheet.write(0, 1, '预到时间')
		worksheet.write(0, 2, '预离时间')
		worksheet.write(0, 3, '机位')  # 标准差
		worksheet.write(0, 4, '登机口（出发口/到达口）')
		worksheet.write(0, 5, '服务地点')
		worksheet.write(0, 6, '摆渡车序号')
		worksheet.write(0, 7, '预定发车时间')
		worksheet.write(0, 8, '第一段行程时间')
		worksheet.write(0, 9, '第二段行程时间')
		worksheet.write(0, 10, '预估结束时间')
		worksheet.write(0, 11, '原地等待时间')
		worksheet.write(0, 12, '拖期')

		# print(results1)

		for i in range(len(results1)):
			for k in range(13):
				worksheet.write(i+1, k, results1[i][k])

		# print(results)
		'''写入每辆摆渡车工单'''
		for i in range(len(p)):
			sheetName = str(i)	#摆渡车编号
			worksheet = workbook.add_worksheet(sheetName)
			worksheet.write(0, 0, '航班序号')
			worksheet.write(0, 1, '预到时间')
			worksheet.write(0, 2, '预离时间')
			worksheet.write(0, 3, '机位')  # 标准差
			worksheet.write(0, 4, '登机口（出发口/到达口）')
			worksheet.write(0, 5, '服务地点')
			worksheet.write(0, 6, '预定发车时间')
			worksheet.write(0, 7, '第一段行程时间')
			worksheet.write(0, 8, '第二段行程时间')
			worksheet.write(0, 9, '预估结束时间')
			worksheet.write(0, 10, '原地等待时间')
			worksheet.write(0, 11, '拖期')
			for j in range(len(p[i])):
				for k in range(12):
					worksheet.write(j+1, k, results[i][j][k])

		workbook.close()  # 关闭表格并保存内容