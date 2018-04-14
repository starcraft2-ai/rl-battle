# Multi-Agent Reinforcement learning on StarCraft 2 

## Benchmarks

### Repository Introduction
1. `refs/pysc2-rl-agents`: A2C method referenced from [simonmeister's repo](https://github.com/simonmeister/pysc2-rl-agents), abbreviation as `simon-a2c`

### Results
| Map | Traning steps|`simon-a2c` | random agent | DeepMind human |
| --- | --- | --- | --- | --- |
| (mean) DefeatRoaches | 1000 | 5.74 | -4 | 41 |
| ( max ) DefeatRoaches | 1000 | 61 | 11 | 81 |
| (mean) DefeatZerglingsAndBanelings | N/A | N/A | 18.1  |  729 |
| ( max ) DefeatZerglingsAndBanelings | N/A | N/A |72| 757 |
| (mean) MoveToBeacon | 5000 | 25.26 | 1.2 | 26 |
| ( max ) MoveToBeacon | Same | 30.00 | 3 | 28 |
| (mean) CollectMineralShards | N/A | N/A | 18.5 | 133 |
| ( max ) CollectMineralShards | N/A | N/A | 32 | 142 |
| (mean) FindAndDefeatZerglings | N/A | N/A | N/A | 46 |
| ( max ) FindAndDefeatZerglings | N/A | N/A | N/A | 49 |

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

