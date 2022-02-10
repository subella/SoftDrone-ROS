Installation
============

First you need to install the softdrone trajectory optimization python package:
```
cd $SoftDrone_ROS_DIR/SoftDrone-TrajOpt
pip2 install -e .
```

Now install the rosdep dependencies
```
cd <path_to_catkin_ws>
rosdep install --from-paths src --ignore-src -r -y
```

I think this should take care of installing MRPT automatically, but if the next
step fails let mek now and you may need to install MRPT manually for now.

Next, build the softdrone core package:
```
catkin build softdrone_core
```

Simulation
==========

To launch the ROS simulation, in one terminal run:
```
roslaunch softdrone_core master.launch simulation_ros:=true
```
This should result in a new RVIZ window.

In another terminal, run
```
roslaunch softdrone_core setpoint_cli_node.launch send_to_mavros:=false
```

We use this window to (among other things) mimic arming the drone from the RC
controller.  Type ARM into the prompt and press enter. After 5 seconds the
drone should take off and start following the red trajectory.


Rebase-Oriented Git Workflow
============================

If you are making reasonably small changes that don't rely on coordination with others, we prefer a rebase-oriented workflow. 

```
git checkout master
git checkout -b my_branch
< make some edits >
< In the meantime, someone updated upstream master >
< Now, we want to merge our branch >
git checkout master
git pull
git checkout my_branch
git rebase master
< Fix any conflicts >
git checkout master
git merge --ff-only my_branch
git push origin master
```

Note that after you have rebased `my_branch` on master, if you want to push
your branch to github you will probably need to run `git push -f origin
my_branch`. See below for discussion on why this may not be a good idea if
others are simultaneously working on your branch.

Also, cleaning up your commits and commit messages before merging into master
would be greatly appreciated. You can squash small fixup commits and make the history more 
meaningful by doing an interactive rebase (something like `git rebase -i HEAD~5`).

While the current code is in so much flux and it's ok to have a small chance of breaking master,
this alternative workflow can be even faster:

```
git checkout master
< make some edits >
< In the meantime, someone updated upstream master >
< Now, we want to merge our branch >
git fetch
git rebase origin/master
git push origin master
```

In both cases, DO NOT rebase master on your branch. 
```
THIS IS BAD. NEVER DO THIS
git checkout master
git pull
git rebase my_branch
```

Luckily if you do this, github would prevent you from pushing this to
origin/master by default. While it might be alright to force push to your own
branch, please to not force push to master.


If there are commits on master that aren't on `my_branch`, you will change the
commit hash of the commits on master and make me very sad.  In general, some
people don't like `git rebase` because it re-writes history. This is because
the commit hash of your commits change. It is true that you need to be aware
of this when working with branches that have been shared with others. If you
have pushed your branch to github and are working on it in parallel with others,
then updating your branch with rebase can cause a lot of headaches. However,
an ideal development workflow is to have small, short-lived feature branches
that can be worked on independently and merged into master quickly. In such
a workflow, rebasing works very well and results in a clean, understandable
commit history.





