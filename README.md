# Multi-Agent Reinforcement learning on StarCraft 2 

## Benchmarks

### Repository Introduction
1. `refs/pysc2-rl-agents`: A2C method referenced from [simonmeister's repo](https://github.com/simonmeister/pysc2-rl-agents), abbreviation as `simon-a2c`

### Results
| Map | `simon-a2c` | random agent | DeepMind human |
| --- | --- | --- | --- |
| DefeatRoaches | N/A | mean -4 (max 11) | mean 41(max 81) |
| DefeatZerglingsAndBanelings | N/A |mean 18.1 (max 72) | mean 729(max 757) |
| MoveToBeacon | N/A | mean 1.2 (max 3) | mean 26(max 28) |
| CollectMineralShards | N/A | mean 18.5 (max 32) | mean 133(max 142) |
| FindAndDefeatZerglings | N/A | N/A | mean 46(max 49) |

## Prepare

#### Python environment
You sould create or activate Virtual Environment now
```
$ source .env/bin/activate
(.env) ...$ 
```
#### Install Dependency
```
pip install -r requirements.txt
```

## Structure
![Call Tree](https://github.com/starcraft2-ai/rl-battle/raw/master/assets/Call%20Tree.png)

