# RL
## Action Space
1. do nothing
2. fire left engine
3. fire right engine
4. fire main engine
## Observation Space
### 8D vector
1. the coordinates of the lander in x
2. the coordinates of the lander in y
3. its linear velocities in x
4. its linear velocities in y
5. its angle
6. its angular velocity
7. boolean that represent whether left leg is in contact with the ground or not.
8. boolean that represent whether right leg is in contact with the ground or not.
## Rewards
1. Reward for moving from the top of the screen to the landing pad and coming to rest is about 100-140 points.
2. If the lander moves away from the landing pad, it loses reward.
3. If the lander crashes, it receives an additional -100 points.
4. If it comes to rest, it receives an additional +100 points.
5. Each leg with ground contact is +10 points.
6. Firing the main engine is -0.3 points each frame. 
7. Firing the side engine is -0.03 points each frame.
8. Solved is 200 points.
## Start state
The lander starts at the top center of the viewport with a random initial force applied to its center of mass.
## Episode Termination
1. the lander crashes (the lander body gets in contact with the moon)
2. the lander gets outside of the viewport (x coordinate is greater than 1)
3. the lander is not awake. From the Box2D docs, a body which is not awake is a body which doesn’t move and doesn’t collide with any other body

**This information comes from:** https://www.gymlibrary.dev/environments/box2d/lunar_lander/
