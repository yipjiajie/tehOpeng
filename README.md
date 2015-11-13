### TehOPeng
##### CS4243 Computer Vision Project

### How to use
#### Step 1:
Ensure that you have all these files in the same folder
* ``homo2.py``
* ``util.py``
* football_left.mp4, football_mid.mp4, football_right.mp4
* ``ExtractBackground.py``

#### Step 2:
* Run ``homo2.py`` to generate individual frames image of stitched video
* Run ``ExtractBackground.py`` to get output frames
* Run ``render.py`` to get final video output

#### Explanation:
* ``homo2.py`` (with the help of util.py) takes in the three football videos and * stitches them into one video and then generates individual video frames. It can take quite awhile, roughly 2-3 hours and can take up quite a big of space, roughly 8GB.

* ``ExtractBackground.py` then reads in the frames generated and output the topdown view and tracking of the players on the field in individual frames

* ``render.py`` finally reads in all the frames from the previous step to recreate the video in a lower resolution
