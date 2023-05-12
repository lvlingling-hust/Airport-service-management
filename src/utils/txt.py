import os

def getTxt(dir_path, args, all_solutions, gen):
    if not os.path.isdir(dir_path+ '/'+ args.sheetName):
        os.makedirs(dir_path+ '/'+ args.sheetName)
    f = open(dir_path+ '/' + args.sheetName + '/' + args.flightFileName + '_' + args.sheetName + args.init_action +'_gen_' + str(gen) +'.txt', 'w')
    for i in range(len(all_solutions)):
        f.write(str(all_solutions[i])+'\n')
    f.close()

def readTxt(file_path):
    # all_solutions = [best_p, best_solution, best_fitness, best_all_tardiness, all_fitness_list, all_tardiness_list]
    f = open(file_path, 'r')
    lines = f.readlines()
    best_fitness = eval(lines[0])
    best_tardiness = eval(lines[1])
    run_time = float(lines[2])
    f.close()
    return best_fitness, best_tardiness,run_time