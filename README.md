# hindsight-experience-replay

This repository implements the hingsight experience replay algorithm on the Gen3 robot arm running on the MuJoCo simulation environemnt.

![](demo.gif)

The project implements an AI agent in 7 DOF robotic arm with the goal of navigating to an object on the table.

## Copy the robot model's library to the tmp folder

```bash
cp -r ./assets/gen3/ /tmp/
```

## Run the training

```bash
python3 train.py
```

## Run the testing

```bash
python3 demo.py
```
