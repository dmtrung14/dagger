<br />
<div align="center" style="border-style: solid; border-width: 1px; border-color: white">
<h3 align="center">dagger-racing</h3>

  <p align="center">
    CarRacing-v2 Trained with DAggER Imitation Learning algorithm
    <br />
    <br />
    <br />
    <a href="https://github.com/dmtrung14/dagger">GitHub</a>
    ·
    <a href="https://github.com/dmtrung14/dagger/issues">Report Bug</a>
  </p>
</div>

This repository contains the code for an imitation learning model and the DAgger algorithm for the CarRacing-v2 Gym Environment. 

This is part of the [MassAI](https://discord.com/invite/47e96wJEVK) Beginner's Lecture Series. Lecture on Imitation Learning was on 11/06/2024.

This repository took inspiration from [https://github.com/kvgarimella/dagger](https://github.com/kvgarimella/dagger), even though this repository is deprecated and will not work for updated Gym versions.
![](https://github.com/dmtrung14/dagger/blob/main/media/dagger.gif)

DAgger helps the imitation learning agent learn correct actions when following sub-optimal trajectories:

![](https://github.com/dmtrung14/dagger/blob/main/media/self-correction.gif)

Check out [this paper](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf) to learn more about DAgger. 

### Installation
Clone this repository:
```
git clone https://github.com/dmtrung14/dagger.git
cd dagger
```
Install the requirements:
```
pip install -r requirements.txt
```
Run DAgger and train your model:
```
python dagger.py
```


