# Multi-Agent Reinforcement learning on StarCraft 2 
Reinforcement Learning Agents implmented in Tensorflow, featuring:
- morden Eager Execution (human readable) structure
- Mutiple CPU Support (HPC)

List of implemented Agents are:
- Random Agent
- [Modified A3C Agent](https://github.com/starcraft2-ai/rl-battle/tree/A/C-online)
- [Modified A2C Agent](https://github.com/starcraft2-ai/rl-battle/tree/a2c-episode-total-advantage)

# Important 
Checkout the following branch before running
- [A3C](https://github.com/starcraft2-ai/rl-battle/tree/A/C-online)
- [A2C](https://github.com/starcraft2-ai/rl-battle/tree/a2c-episode-total-advantage)

## Benchmarks
See our [this repo](https://github.com/starcraft2-ai/comparison)

## Running
### Benchmark
Run `MoveToBeacon` map with random agent
```shell
python benchmark.py --map MoveToBeacon --agent_name RandomAgent
```

### Training
Please checkout different [branch](#Important)

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
## Development
### Structure
![Call Tree](https://github.com/starcraft2-ai/rl-battle/raw/master/assets/Call%20Tree.png)

