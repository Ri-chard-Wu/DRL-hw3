
import os
import importlib
# import torch
import gym
import sys
import numpy as np
import inspect
import gym_multi_car_racing


def check_file_integrity(foldernames):

    checking_list = []
    for file in foldernames:
        if not os.path.isdir(file) and "_hw3" in file:
            checking_list.append(file.split("_")[2])
 
    elements = ["data","test.py","train.py"]
    
    for name in elements:
        if name not in checking_list:
            print(f"\033[91m hw3_{name} IS MISSING\033[0m")
            return False
    
    return True
        


def check_agent(foldernames):

    test_file = None
    for file in foldernames:
        if "hw3_test.py" in file:
            test_file = file

    module_name = test_file.replace('/', '.').replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, test_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        Agent = getattr(module, 'Agent')
        return Agent
    except AttributeError as e:
        return None
    

        
def check_agent_act(Agent):

    if hasattr(Agent, 'act') and callable(getattr(Agent, 'act')):

        # cheking data form
        act_signature = inspect.signature(Agent.act)
        params = list(act_signature.parameters.keys())

        act_signature = inspect.signature(Agent.__init__)
        params_i = list(act_signature.parameters.keys())

        if len(params_i) == 1 and params[0] == 'self':
            print(f"\033[92mAgent Init Function Checking: PASS \033[0m")
        else:
            print(f"\033[91mAgent Init FUNCTION HAS WRONG PARAMETER LIST!\033[0m")
            return False

        if len(params) == 2 and params[1] == 'observation':
            print(f"\033[92mAgent Act Function Checking: PASS +5\033[0m")
        else:
            print(f"\033[91mAgent ACT FUNCTION HAS WRONG PARAMETER LIST!\033[0m")
            return False
    else:
        print(f"\033[91mAgent MISSING ACT FUNCTION!\033[0m")
        return False

    

    return True

def check_data_form(Agent):

    observation_space = gym.spaces.Box(0, 255, [1, 96,96, 3],dtype=np.uint8)
    test_obs = observation_space.sample()
    agent = Agent() 

    try:
        action = agent.act(test_obs)

        if len(action) != 3:
            print(f"\033[91mAgent ACT FUNCTION RETURNS ACTION IN WRONG FORM\033[0m")
            return False
    except Exception as e:
        print(e)
        print(f"\033[91mAgent ACT FUNCTION CANNOT HANDLE CORRECT OBSERVATION FORM\033[0m")
        return False
    
    print(f"\033[92mData Form Checking: PASS +5\033[0m")
    return True


def check_one_episode(Agent):

    env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False)

    agent = Agent()
    observation = env.reset()
    done = False
    total_reward = 0

    while not done:
        try:
            action = agent.act(observation)
        except Exception as e:
            print(e)
            print(f"\033[91mAgent CANNOT HANDLE CORRECT OBSERVATION FORM\033[0m")
            return False
        observation, reward, done, _ = env.step(action)
        total_reward += reward
    print(total_reward)
    print(f"\033[92mEnv Run Checking: PASS +5\033[0m")
    return True

# main function
def main():

    score = 0

    files = []
    foldernames = os.listdir()
    for file in foldernames:
        if file.startswith("1"):
            files.append(file)

    # checking
    if check_file_integrity(files):
        print(f"\033[92mFile Checking: PASS +5\033[0m")
        score += 5
    else:
        return
    
    Agent = check_agent(files)
    if Agent == None:
        print(f"\033[91mAgent Class IS MISSING IN TEST FILE\033[0m")
    else:
        print(f"\033[92mAgent Class Checking: PASS +5\033[0m")
        score += 5

    if check_agent_act(Agent):
        score += 5

    if check_data_form(Agent):
        score += 5
    
    if check_one_episode(Agent):
        score += 5

    # showing score    
    print(f"\033[93mScore: {score}/25\033[0m")

if __name__ == "__main__":
    main()
