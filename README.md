# Visual-Interaction-Networks
An implementation of Deepmind visual interaction networks in Pytorch.
 # Introduction
> For the purpose of understanding the challenge of relational reasoning. they publised VIN that involves predicting the future in a physical scene. From just a glance, humans can infer not only what objects are where, but also what will happen to them over the upcoming seconds, minutes and even longer in some cases. For example, if you kick a football against a wall, your brain predicts what will happen when the ball hits the wall and how their movements will be affected afterwards (the ball will ricochet at a speed proportional to the kick and - in most cases - the wall will remain where it is).

<div align="center">

<img align="center" src="https://github.com/Mrgemy95/visual-interaction-networks-pytorch/blob/master/figures/1.gif?raw=true">
</div>


## Architecture
<div align="center">
<img hight="800" width="800" src="https://github.com/Mrgemy95/visual-interaction-networks-pytorch/blob/master/figures/2.png?raw=true">
</div>


### Data
I used [I@jaesik817](https://github.com/jaesik817/Interaction-networks_tensorflow) physics engine to generate the data.

Just run the `physics_engine.py`


## Usage
### Main Dependencies
``` 
Python 3.5
pytorch 0.3
numpy 1.13.1
```

### RUN
- Edit configration file to meet your need.
- Run `vin.py`

### References
* https://github.com/jaesik817/visual-interaction-networks_tensorflow