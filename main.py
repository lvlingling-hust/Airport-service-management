from src.ASSP_GA.GA import *
from src.utils.txt import *
import argparse
import time
import numpy as np

parser = argparse.ArgumentParser(description='FIFO')
parser.add_argument('--flightFileName', type=str, default="flight_info.xlsx",
                    help='test data file name')
parser.add_argument('--sheetName', type=str, default="Sheet1",
                    help='test data file name')
parser.add_argument('--shuttle_capacity', type=int, default=80,
                    help='The capacity of the shuttle')
parser.add_argument('--coef', type=int, default=0,
                    help='adjust charging penalty')
parser.add_argument('--popSize', type=int, default=30,
                    help='Size of population, must be even for crossover')
parser.add_argument('--maxGen', type=int, default=100,
                    help='How many Generations to get the best solution')
parser.add_argument('--pr', type=float, default=0.2,
                    help='Reproduction')
parser.add_argument('--pc', type=float, default=0.8,
                    help='Crossover')
parser.add_argument('--pm', type=float, default=0.2,
                    help='Mutation')
parser.add_argument('--b', type=int, default=2,
                    help='b-tournament')
parser.add_argument('--runNum', type=int, default=10,
                    help='How many run for an instance')
parser.add_argument('--seed', type=int, default=1,
                    help='random_seed')
parser.add_argument('--init_action', type=str, default='random',
                    help='initialize_actions:random,in_turn,fifo')

if __name__ == '__main__':
    args = parser.parse_args()
    #np.random.seed(args.seed)
    for k in range(args.runNum):
        np.random.seed(k+1)
        t0 = time.time()
        ga = GA(args)

        best_p,best_solution,best_fitness, best_tardiness, best_all_tardiness, all_fitness_list,all_tardiness_list = ga.ga_total(args)
        # best_p =[[1,3,4,10],[8,2,5],[7,9,6]]
        # 12号测试数据
        # best_p =[[1, 23, 27, 40, 90, 99, 118, 127, 155, 70, 80, 112, 164, 113, 181], [2, 24, 30, 88, 93, 56, 128, 63, 67, 141, 146, 158, 177, 188], [3, 17, 31, 89, 94, 57, 119, 65, 143, 76, 150, 166, 175, 187], [4, 15, 33, 85, 95, 60, 122, 125, 130, 72, 148, 160, 171, 185], [5,16, 36, 46, 86, 107, 133, 100, 105, 103, 115, 154, 170, 183], [6, 18, 34, 44, 51, 97, 124, 135, 139, 109, 78, 151, 163, 114, 182], [7,19, 35, 45, 92, 58, 120, 66, 144, 77, 111, 167, 179, 192], [8, 20, 32, 48, 54, 61, 132, 102, 137, 74, 82, 162, 172, 186], [9, 21, 37, 47, 87, 59, 121, 126, 131, 73, 149, 161, 173, 190], [10, 29, 41, 91, 96, 69, 62, 156, 71, 81, 152, 168, 180], [11, 25, 42, 50, 55, 129,64, 68, 142, 147, 159, 174, 191], [12, 26, 43, 52, 106, 134, 101, 138, 75, 83, 165, 176, 189], [13, 22, 38, 84, 53, 117, 108, 104, 145,116, 157, 178], [14, 28, 39, 49, 98, 123, 136, 140, 110, 79, 153, 169, 184]]

        # best_solution = ga.get_gantt_time(best_p)[0]
        # print('solution',best_solution)
        # all_objs, my_tardiness_obj, all_tardiness_obj, my_tardiness, best_all_tardiness = ga.calculate_fitness(best_p,best_solution)
        # best_fitness = sum(all_objs)
        # print('!!!!!!!!all_tardiness', best_all_tardiness)
        # print('best_fitness',best_fitness)

        t1 = time.time()
        run_time = round(t1-t0,2)

        all_solutions = [round(best_fitness,2), best_tardiness,run_time]
        getTxt('GA/results', args, all_solutions, k+1)

        #所有的图先不画，可以根据种子，找相应输出
        # ga.getWorkOrderExcel(best_p, best_solution, best_all_tardiness, 'GA/results/'+args.sheetName, args.flightFileName + '-' + args.sheetName+'-'+args.init_action +'-seed'+str(args.seed) + 'workOrder.xlsx')
        # ga.draw_change(all_fitness_list,all_tardiness_list,  'GA/results/'+args.sheetName,args.flightFileName+'-'+args.sheetName+'-'+args.init_action+'-seed'+str(args.seed))
        # ga.visualize(best_p,best_solution, best_all_tardiness, 'GA/results/'+args.sheetName, args.flightFileName+'-'+args.sheetName+'-'+args.init_action+'-seed'+str(args.seed))
        # ga.visualize_tardiness(best_p,best_solution, best_all_tardiness, 'GA/results/'+args.sheetName, args.flightFileName+'-'+args.sheetName+'-'+args.init_action+'-seed'+str(args.seed))
        # utilization, num_shuttle = ga.get_utilization(best_solution)
        # worktime_list = ga.get_gantt_time(best_p)[1]
        # my_worktime = [sum(time) for time in worktime_list]
        # ga.visualize_utilization(utilization, my_worktime, num_shuttle, 'GA/results/'+args.sheetName, args.flightFileName + '-' + args.sheetName+'-'+args.init_action+'-seed'+str(args.seed))
