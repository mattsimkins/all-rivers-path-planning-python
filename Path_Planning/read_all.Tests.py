'''This is for reading in all .txt files from a dataset and runs the model'''

import pvf
import os
import time

#Create PVF model with node spaing=60
T = pvf.TrainModel(80)

#Name of directory containing data
data_set_dir = "Dataset_temp"

#Read in all .txt files from dataset directory

file_list = os.listdir(data_set_dir)
print (file_list)
file_list.sort()
for file_name in file_list:
    if file_name[-4:] != '.txt':
        file_list.remove(file_name)

for file_name in file_list:
    traj = T.train_on_trajectory(path=data_set_dir, extents=[-2200, 2200, -1300, 1300], file_name=file_name)


#T.save_model()
#pvf.plot_trajectory(traj, extents=[-2200, 2200, -1300, 1300], image_file = "Maze.jpg")


############Intersections
#maze beginning (-1697, 858)
#possible start points at intersections
start_time = time.time()
traj0 = T.train_on_trajectory(path=data_set_dir, start_point = (-1521, 826))
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

pvf.plot_trajectory(traj0, extents=[-2200, 2200, -1300, 1300], image_file = "Maze.jpg")
'''traj1 = T.train_on_trajectory(path=data_set_dir, start_point = (-1495, 495))
traj2 = T.train_on_trajectory(path=data_set_dir, start_point = (-1800, -660))
traj3 = T.train_on_trajectory(path=data_set_dir, start_point = (-1190, -160))
traj4 = T.train_on_trajectory(path=data_set_dir, start_point = (-590, 123))
traj5 = T.train_on_trajectory(path=data_set_dir, start_point = (-435, 418))
traj6 = T.train_on_trajectory(path=data_set_dir, start_point = (-967, 402))
traj7 = T.train_on_trajectory(path=data_set_dir, start_point = (-941, -668))
traj8 = T.train_on_trajectory(path=data_set_dir, start_point = (-156, -187))
traj9 = T.train_on_trajectory(path=data_set_dir, start_point = (124, 190))
traj10 = T.train_on_trajectory(path=data_set_dir, start_point = (-202, 873))
traj11 = T.train_on_trajectory(path=data_set_dir, start_point = (434, 702))
traj12 = T.train_on_trajectory(path=data_set_dir, start_point = (1163, 454))
traj13 = T.train_on_trajectory(path=data_set_dir, start_point = (460, 159))
traj14 = T.train_on_trajectory(path=data_set_dir, start_point = (879, -518))
pvf.plot_trajectory(traj0, traj1, traj2, traj3, traj4, traj5, traj6,\
    traj7, traj8, traj9, traj10, traj11, traj12, traj13, traj14,\
        extents=[-2200, 2200, -1300, 1300], image_file = "Maze.jpg")'''


#############Dead Ends ################

'''traj0 = T.train_on_trajectory(path=data_set_dir, start_point = (-1800, 483))
traj1 = T.train_on_trajectory(path=data_set_dir, start_point = (-1800, 163))
traj2 = T.train_on_trajectory(path=data_set_dir, start_point = (-1242, -930))
traj3 = T.train_on_trajectory(path=data_set_dir, start_point = (-940, 935))
traj4 = T.train_on_trajectory(path=data_set_dir, start_point = (-169, -591))
traj5 = T.train_on_trajectory(path=data_set_dir, start_point = (133,731))
traj6 = T.train_on_trajectory(path=data_set_dir, start_point = (479, -316))
traj7 = T.train_on_trajectory(path=data_set_dir, start_point = (1490, 731))
pvf.plot_trajectory(traj0, traj1, traj2, traj3, traj4, traj5, traj6,\
    traj7, extents=[-2200, 2200, -1300, 1300], image_file = "Maze.jpg")'''



############
#Calculate trajectory with last training as start point
'''pvf.plot_trajectory(traj, title="Pseudo-Average Trajectory",\
    extents=[-2200, 2200, -1300, 1300], image_file = "Maze.jpg")'''
##############



#for coord in traj:
#    print(coord[0], coord[1])


#For large grid sizes this plot can be skipped
#T.plot_grid()

##############plot all trajectories##########
'''
traj0 = pvf.read_traj(data_set_dir, file_list[0])
traj1 = pvf.read_traj(data_set_dir, file_list[1])
traj2 = pvf.read_traj(data_set_dir, file_list[2])
traj3 = pvf.read_traj(data_set_dir, file_list[3])
traj4 = pvf.read_traj(data_set_dir, file_list[4])
traj5 = pvf.read_traj(data_set_dir, file_list[5])
traj6 = pvf.read_traj(data_set_dir, file_list[6])
traj7 = pvf.read_traj(data_set_dir, file_list[7])
traj8 = pvf.read_traj(data_set_dir, file_list[8])
traj9 = pvf.read_traj(data_set_dir, file_list[9])
traj10 = pvf.read_traj(data_set_dir, file_list[10])
traj11 = pvf.read_traj(data_set_dir, file_list[11])
pvf.plot_trajectory(traj0, traj1, traj2, traj3, traj4, traj5, traj6,\
    traj7, traj8, traj9, traj10,traj11, extents=[-2200, 2200, -1300, 1300], image_file = "Maze.jpg")
'''
#traj0 = pvf.read_traj(data_set_dir, "zOptimal path fine.txt")
#pvf.plot_trajectory(traj0, extents=[-2200, 2200, -1300, 1300], image_file = "Maze.jpg")