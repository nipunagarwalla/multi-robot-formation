<SYSTEM_PROMPT> you are a senior robotics researcher, specializing in reinforcement learning. given the correct context, i want you to plan, implement, test and report back findings. </SYSTEM_PROMPT>


Given the code, and the research paper (https://arxiv.org/pdf/2404.01618), i want you to extend the original approach, and write me the code for the following goal:

## Goal 

I want to learn an RL policy using PPO wherein given the number of robots in a cluster, the cluster forms the following shapes:

2 Robots -- Horizonal Line 
3 Robots -- Triangle 
4 Robots -- Square

The environment always starts with 4 robots, spawned near each other so that they can form a square. Based on human-teleoperation any one of the robot can leave the cluster, and respectively the cluster reacts to form a new shape based on the number of robots in a cluster. The human can again choose to teleoperate a new robot -- making it leave the cluster or join the cluster in case the robot is out of the cluster. The teleoperation should give us individual control over all the robots in the cluster. 

For training the policy, you can randomly choose any one robot in the cluster and set it on a different trajectory away from the cluster, and then randomly bring it back as well. So that it mimics the behaviour we want while inference -- where we will be manually teleoperating any robot.

The pygame environment will be a long rectangular hallway with no obstacles, the robots start at one end of the hallway and their goal is to reach the other end of the hallway while ensuring that the cluster formation is maintained. The policy should penalize inter-robot collisons, breaking the formation, and the cluster stalling at one place (the cluster should always keep moving forward). The robot movement, communication logic, and formation logic can be inspired from the context in the current repo. 

Write your plan in plan.md in the root dir. Make it as simple as possible as we want to quickly test out this idea, and do not expect the polish of a professional paper. 





