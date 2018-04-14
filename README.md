# Multi-Agent Reinforcement learning on StarCraft 2 

## Benchmarks

### Repository Introduction
1. `refs/pysc2-rl-agents`: A2C method referenced from [simonmeister's repo](https://github.com/simonmeister/pysc2-rl-agents), abbreviation as `simon-a2c` separately

### Results
| Map | `simon-a2c` | random search | DeepMind human |
| --- | --- | --- | --- |
| DefeatRoaches | N/A | N/A | mean 41(max 81) |
| DefeatZerglingsAndBanelings | N/A | N/A | mean 729(max 757) |
| MoveToBeacon | N/A | N/A | mean 26(max 28) |
| CollectMineralShards | N/A | N/A | mean 133(max 142) |
| FindAndDefeatZerglings | N/A | N/A | mean 46(max 49) |

## Running
### Instruction
Run `MoveToBeacon` map with random agent
```shell
python -m run --map MoveToBeacon --agent refs.random.random_agent.RandomAgent
```

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

