# Multi-Agent Reinforcement learning on StarCraft 2 

## Benchmarks

### Benchmark Introduction
1. `simon-a2c`: A2C method referenced from [a fork of simonmeister's repo](https://github.com/starcraft2-ai/simon-a2c), abbreviation as `simon-a2c`
2. `random agent`: as the name shows, it does things randomly

### Results
| Map | Training iters | simon-a2c | random agent | DeepMind human |
| --- | --- | --- | --- | --- |
| (mean) DefeatRoaches | 1000×16 = 16k | 5.74 | -4 | 41 |
| ( max ) DefeatRoaches | Same | 61.00 | 11 | 81 |
| (mean) DefeatZerglingsAndBanelings | 500×8 = 4k | 24.8 | 18.1  |  729 |
| ( max ) DefeatZerglingsAndBanelings | Same | 108.00 |72| 757 |
| (mean) MoveToBeacon | 5000×6 = 30k | 25.26 | 1.2 | 26 |
| ( max ) MoveToBeacon | Same | 30.00 | 3 | 28 |
| (mean) CollectMineralShards | 2000×4 = 8k | 23.84 | 18.5 | 133 |
| ( max ) CollectMineralShards | Same | 39.00 | 32 | 142 |
| (mean) FindAndDefeatZerglings | 500×8 = 4k | 4.82 | N/A | 46 |
| ( max ) FindAndDefeatZerglings | Same | 18.00 | N/A | 49 |

## Running
### Instruction
Run `MoveToBeacon` map with random agent
```shell
python -m run --map MoveToBeacon --agent agents.random.random_agent.RandomAgent
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

