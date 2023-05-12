import numpy as np
import pandas as pd
import math
import xlsxwriter

class Instance:
    def __init__(self, file_name, shuttles, flights, dist_matrix, depot_id, A_id,B_id,C_id):
        self.file_name = file_name
        self.shuttles = shuttles                        # 所有的摆渡车类
        self.flights = flights                          # 所有的航班类
        self.dist_matrix = dist_matrix
        self.depot_id = depot_id                        # 基地编号
        self.A_id = A_id                                # A休息区编号
        self.B_id = B_id                                # B休息区编号
        self.C_id = C_id                                # C休息区编号
        self.rest_time = 2                              # 如果两个任务之间小于2分钟，则摆渡车去休息区

    def get_shuttles(self):
        return self.shuttles

    def get_flights(self):
        return self.flights

    def get_dist_matrix(self):
        return self.dist_matrix

    def get_depot(self):
        return self.depot_id

class Shuttle:
    def __init__(self):
        self.id = None                                  # 摆渡车编号
        self.capacity = None                            # 载客人数
        self.service_time = None                        # 上下客服务时间
        self.battery_life = None                        # 电池续航时间
        self.charging_time = None                       # 电池充电时间
        self.speed = None                               # 行驶速度
        self.pick_up_time = 5                           # 提前5min时间到达
        self.location = None                            # 记录摆渡车服务的结束位置

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def set_capacity(self, capacity):
        self.capacity = capacity

    def get_capacity(self):
        return self.capacity

    def set_service_time(self, service_time):
        self.service_time = service_time

    def get_service_time(self):
        return self.service_time

    def set_battery_life(self, battery_life):
        self.battery_life = battery_life

    def get_battery_life(self):
        return self.battery_life

    def set_charging_time(self, charging_time):
        self.charging_time = charging_time

    def get_charging_time(self):
        return self.charging_time

    def set_speed(self, speed):
        self.speed = speed

    def get_speed(self):
        return self.speed

    def get_pick_up_time(self):
        return self.pick_up_time

    def update_location(self,loc_id):
        self.location = loc_id

    def get_location(self):
        return self.location

    def __info__(self):
        print('*******************************')
        print('摆渡车编号:', str(self.id))
        print('载客人数:', str(self.capacity))
        print('上下客时间:', str(self.service_time))
        print('续航时间:', str(self.battery_life))
        print('充电时间:', str(self.charging_time))
        print('行驶速度:', str(self.speed))
        print('当前位置:', str(self.location))

class Flight:
    def __init__(self, id_2, early_arrive_time = 60):
        self.id = None                                  # 航班编号
        self.id_2 = id_2                                # 航班编码编号
        self.arrival = None                             # 进港航班表示为1，出港航班表示为0
        self.stand_id = None                            # 到达口/出发口编号
        self.remote_stand_id = None                     # 远机位编号
        self.time = None                                # 航班出发/到达分钟累计时间
        self.service_time = None                        # 航班的摆渡车计划服务分钟累计时间
        self.early_arrive_time = early_arrive_time      # 出港航班需要摆渡车的提前时间
        self.std_time = None                            # 航班出发/到达标准时间
        self.num_passengers = None                      # 旅客人数
        self.priority = None                            #航班延迟或提前的优先级

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def set_arrival(self, arrival):
        self.arrival = arrival

    def get_arrival(self):
        return self.arrival

    def set_stand_id(self, stand_id):
        self.stand_id = stand_id

    def get_stand_id(self):
        return self.stand_id

    def set_remote_stand_id(self, remote_stand_id):
        self.remote_stand_id = remote_stand_id

    def get_remote_stand_id(self):
        return self.remote_stand_id

    def set_time(self, time):
        self.time = time

    def get_time(self):
        return self.time

    def set_service_time(self):
        if self.arrival == 1:
            self.service_time = self.time
        if self.arrival == 0:
            self.service_time = self.time - self.early_arrive_time

    def get_service_time(self):
        return self.service_time

    def set_std_time(self, time_h, time_min):
        if time_min < 10:
            self.std_time = str(time_h) + ':' + '0' +str(time_min)
        else:
            self.std_time = str(time_h) + ':' + str(time_min)

    def get_std_time(self):
        return self.std_time

    def set_num_passengers(self, num_passengers):
        self.num_passengers = num_passengers

    def get_num_passengers(self):
        return self.num_passengers

    def set_priority(self, priority):
        self.priority = priority

    def get_priority(self):
        return self.priority

    def __info__(self):
        print('*******************************')
        print('航班号:', str(self.id))
        print('航班编码编号:', str(self.id_2))
        print('登机时刻:', str(self.std_time))
        if self.arrival == 1:
            print('进出港类型: 进港航班')
            print('到达口:', str(self.stand_id))
        else:
            print('进出港类型: 出港航班')
            print('出发口:', str(self.stand_id))
        print('远机位:', str(self.remote_stand_id))
        print('旅客人数:', str(self.num_passengers))

class Reader:
    def __init__(self, file_name, sheet_name, shuttle_capacity):
        if True:  # try:
            ########################创建所有摆渡车类#####################
            df = pd.read_excel(r'instances/shuttle_info.xlsx')
            shuttles = []                                                    # 装所有的摆渡车类
            num_rows = df.shape[0]                                           # 行数(不算表头）
            for i in range(num_rows):
                shuttle = Shuttle()                                          # 创建一个摆渡车类
                shuttle.set_id(int(df.loc[i, '摆渡车编号']))                   # 依次为摆渡车属性赋值
                shuttle.set_capacity(int(df.loc[i, '载客人数']))
                shuttle.set_service_time(int(df.loc[i, '上下客时间/min']))
                shuttle.set_battery_life(int(df.loc[i, '续航时间/h']))
                shuttle.set_charging_time(int(df.loc[i, '充电时间/h']))
                shuttle.set_speed(int(df.loc[i, '行驶速度/km/h']))
                shuttles.append(shuttle)

            ########################创建所有航班类#####################
            df = pd.read_excel(r'instances/'+file_name,sheet_name=sheet_name)
            flights = []
            num_rows = df.shape[0]
            id_2 = 0
            for i in range(num_rows):
                for j in range(math.ceil(int(df.loc[i, '旅客人数']) / shuttle_capacity)):
                    flight = Flight(id_2 = id_2+1)                               # 创建一个航班类
                    flight.set_id(int(df.loc[i, '航班号']))                       # 依次为航班属性赋值
                    flight.set_time(int(df.loc[i, '计划服务时间']))
                    flight.set_std_time(int(df.loc[i, '计划服务时间/h']),int(df.loc[i, '计划服务时间/min']))
                    flight.set_arrival(int(df.loc[i, '进出港类型']))
                    flight.set_stand_id(int(df.loc[i, '到达/出发口']))
                    flight.set_remote_stand_id(int(df.loc[i, '远机位']))
                    flight.set_num_passengers(int(df.loc[i, '旅客人数']))
                    flight.set_priority(int(df.loc[i, '优先级']))
                    flight.set_service_time()
                    flights.append(flight)
                    id_2 += 1

            ########################读取地图信息，生成距离矩阵#####################
            df = pd.read_excel(r'instances/map_info.xlsx')
            num_rows = df.shape[0]
            dist_matrix = np.zeros((num_rows,num_rows))
            for i in range(num_rows):
                for j in range(num_rows):
                    x_diff = float(df.loc[i,'X轴坐标'])-float(df.loc[j,'X轴坐标'])
                    y_diff = float(df.loc[i, 'Y轴坐标']) - float(df.loc[j, 'Y轴坐标'])
                    dist_matrix[i][j] = pow(x_diff**2+y_diff**2,1/2)/2

            depot_id = int(df.loc[0, '编号'])
            A_id = int(df.loc[7, '编号'])
            B_id = int(df.loc[8, '编号'])
            C_id = int(df.loc[9, '编号'])
            self.instance = Instance(file_name+'-'+sheet_name, shuttles, flights, dist_matrix, depot_id, A_id, B_id, C_id)

        else:  # except:
            print(
                'Could not read the problem specification. Check if the path is correct and the problem specification is in the expected format.')
            raise SystemExit(0)

    def get_instance(self):
        return self.instance


# reader = Reader('real_flight_info0703.xlsx','Sheet2',80)
# instance = reader.get_instance()
# print('len(instance.shuttles)',len(instance.shuttles))
# for shuttle in instance.shuttles:
#     shuttle.__info__()
# print('len(instance.flights)',len(instance.flights))
# for flight in instance.flights:
#     flight.__info__()
#     travel_dist = instance.dist_matrix[10][169]
#     travel_time = travel_dist/instance.shuttles[0].get_speed()*60
#     print('travel_time',travel_time)
# print('instance.dist_matrix',instance.dist_matrix)
#
#
# def getExcel(path, dist_matrix):
#     workbook = xlsxwriter.Workbook(path)  # 建立文件并命名
#     '写入表头'
#     worksheet = workbook.add_worksheet()
#     for i in range(len(dist_matrix)):
#         for j in range(len(dist_matrix)):
#             worksheet.write(i, j, dist_matrix[i][j])
#     workbook.close()  # 关闭表格并保存内容
#
# getExcel('instances/dist_matrix_info.xlsx',instance.dist_matrix)





