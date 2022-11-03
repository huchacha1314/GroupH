import numpy as np
from utility_functions import *

def update_1(Grid,pedestrians_1,pedestrians_2,pedestrians_3,pedestrians_4,pedestrians_5,click_flag):
    '''
    Grid : List of Cells row X col
    pedestrian : x, y coordinate values of the pedestrian
    '''

    
    pedestrians_1 = np.array(pedestrians_1)
    pedestrians_2 = np.array(pedestrians_2)
    pedestrians_3 = np.array(pedestrians_3)
    pedestrians_4 = np.array(pedestrians_4)
    pedestrians_5 = np.array(pedestrians_5)
    
    row_size = len(Grid)
    
    loc = [0,1]
   
    # 1s
    flag = click_flag
    
      
        
    for i in range(len(pedestrians_1)):

        curr_loc_1 = np.array(pedestrians_1[i])
        last_loc_1 = curr_loc_1
        next_loc_1 = curr_loc_1 + loc
        if flag%9 == 0: 
            Grid[last_loc_1[0]][last_loc_1[1]].state = 'empty'
            Grid[next_loc_1[0]][next_loc_1[1]].state = 'pedestrians_1'
            pedestrians_1[i] = next_loc_1
        else:
            continue
            
             
    for i in range(len(pedestrians_2)):

        curr_loc_2 = np.array(pedestrians_2[i])
        last_loc_2 = curr_loc_2
        next_loc_2 = curr_loc_2 + loc
        if flag%5 == 0 : 
            Grid[last_loc_2[0]][last_loc_2[1]].state = 'empty'
            Grid[next_loc_2[0]][next_loc_2[1]].state = 'pedestrians_2'
            pedestrians_2[i] = next_loc_2
        else:
            continue
            
    for i in range(len(pedestrians_3)):

        curr_loc_3 = np.array(pedestrians_3[i])
        last_loc_3 = curr_loc_3
        next_loc_3 = curr_loc_3 + loc
        if flag%4 == 0: 
            Grid[last_loc_3[0]][last_loc_3[1]].state = 'empty'
            Grid[next_loc_3[0]][next_loc_3[1]].state = 'pedestrians_3'
            pedestrians_3[i] = next_loc_3
        else:
            continue
            
             
    for i in range(len(pedestrians_4)):

        curr_loc = np.array(pedestrians_4[i])
        last_loc = curr_loc
        next_loc = curr_loc + loc
        if flag%6 == 0: 
            Grid[last_loc[0]][last_loc[1]].state = 'empty'
            Grid[next_loc[0]][next_loc[1]].state = 'pedestrians_4'
            pedestrians_4[i] = next_loc
        else:
            continue 
            
    for i in range(len(pedestrians_5)):

        curr_loc = np.array(pedestrians_5[i])
        last_loc = curr_loc
        next_loc = curr_loc + loc
        if flag%7 == 0: 
            Grid[last_loc[0]][last_loc[1]].state = 'empty'
            Grid[next_loc[0]][next_loc[1]].state = 'pedestrians_5'
            pedestrians_5[i] = next_loc
        else:
            continue        
        
        
        
  
            
        
    
            
            
   

    return Grid, pedestrians_1,pedestrians_2,pedestrians_3,pedestrians_4,pedestrians_5
