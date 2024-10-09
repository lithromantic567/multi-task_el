#找不同任务：找出其中一张删除的物体类别
# 数据组成：一对图像（其中一张删除一类物体）+删除类别标签
from __future__ import annotations
import json
import pickle

import numpy as np

import pygame


from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.world_object import Ball, Door, Goal, Key, Wall, Box
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

from minigrid.envs.babyai.core.levelgen import LevelGen
from minigrid.envs.babyai.core.roomgrid_level import RejectSampling, RoomGridLevel
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc

from minigrid.wrappers import DictObservationSpaceWrapper, FullyObsWrapper

from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

from Param import *
from torch.utils.data import DataLoader
import random
from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
)


COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
'''
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}
'''
OBJECT_TO_IDX= {"key":0,"ball":1,"box":2}
NEW_CLASS_TO_IDX = {"red_key": 0, "red_ball": 1, "red_box": 2, "green_key": 3, "green_ball": 4, "green_box": 5, "blue_key": 6, "blue_ball": 7, "blue_box": 8, "purple_key": 9, "purple_ball": 10, "purple_box": 11, "yellow_key": 12, "yellow_ball": 13, "yellow_box": 14, "grey_key": 15, "grey_ball": 16, "grey_box": 17}
class GoTo(RoomGridLevel):
    """

    ## Description

    Go to an object, the object may be in another room. Many distractors.

    ## Mission Space

    "go to a/the {color} {type}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent goes to the object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoTo-v0`
    - `BabyAI-GoToOpen-v0`
    - `BabyAI-GoToObjMaze-v0`
    - `BabyAI-GoToObjMazeOpen-v0`
    - `BabyAI-GoToObjMazeS4R2-v0`
    - `BabyAI-GoToObjMazeS4-v0`
    - `BabyAI-GoToObjMazeS5-v0`
    - `BabyAI-GoToObjMazeS6-v0`
    - `BabyAI-GoToObjMazeS7-v0`
    """

    def __init__(
        self,
        room_size=8,
        num_rows=1,
        num_cols=1,
        num_dists=18,
        doors_open=True,
        
        **kwargs,
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open   
        self.agent_dir=None  
        
      
        
        
        super().__init__(
            
            num_rows=num_rows, 
            num_cols=num_cols, 
            room_size=room_size, 
            highlight=False,            
            **kwargs
        )
       

    def gen_mission(self):
        # We catch RecursionError to deal with rare cases where
        # rejection sampling gets stuck in an infinite loop
        # Place a goal square in the bottom-right corner
        #self.put_obj(Goal(), self.width - 2, self.height - 2)
        #self.place_agent()
        '''
        #增加了room
        self.agent_row=self.agent_pos[1]//7
        self.agent_col=self.agent_pos[0]//7
        self.agent_room = self.agent_row*self.num_rows+self.agent_col
        print(self.agent_pos)
        print(self.agent_room)
        print(self.agent_row)
        print(self.agent_col)
        '''
        #self.agent = None
        
        #self.connect_all()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        #self.check_objs_reachable()
        #obj = self._rand_elem(objs)
        #self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()

def output_obs(dic,output_file):
    with open(output_file, 'wb') as f:  
        #object,color,state [column,row,emb] 
        pickle.dump(dic, f)
def main():
    env = GoTo(render_mode="human")
    env._gen_grid(env.width, env.height)
    
    for i in range(5000):
        env.reset()
        obs=env.grid.encode() 
        dic={}

        dic["data"]=obs
        env.render()
        pygame.image.save(env.window, "Data/mt2/pic/obs{}.png".format(i))

        # Randomly select an object to remove from the room
        objects = env.room_grid[0][0].objs
        
        if objects:
            object_to_remove = random.choice(objects)
            while object_to_remove.type not in OBJECT_TO_IDX:
                object_to_remove = random.choice(objects)
            print(object_to_remove.type,object_to_remove.color,object_to_remove.cur_pos)
            
        for row in range(env.grid.height):
            for col in range(env.grid.width):
                cell = env.grid.get(row, col)

                # 如果格子中有，删除物体
                if cell is not None  and cell.color == object_to_remove.color :
                    env.grid.set(row, col, None)
        obs_d=env.grid.encode() 
        dic["data_d"]=obs_d
        dic["label"]= COLOR_TO_IDX[object_to_remove.color]
        output_file='Data/mt2/data'+'/obs{}.txt'.format(i)
        output_obs(dic,output_file)
        
        
        env.render()
        pygame.image.save(env.window, "Data/mt2/pic_d/obs{}.png".format(i))
    
if __name__ == "__main__":
    main()
    
