### TehOPeng
##### CS4243 Computer Vision Project

### How to use
#### Step 1:
Ensure that you have all these files in the same folder
* ``Stitch.py``
* ``util.py``
* football_left.mp4, football_mid.mp4, football_right.mp4
* ``Tracking.py``
* ``Render.py``
* ``pitch.jpg``

#### Step 2:
* Create folder ``pics`` in the same folder
* Run ``Stitch.py`` to generate individual frames image of stitched video

Ensure that you have all these folders and files created in the same folder
* ``pics/finalxxxx.jpg`` created by ``Stitch.py``

#### Step 3:
* Create folder ``enlarge`` in the same folder
* Run ``Tracking.py`` to get output frames

Ensure that you have all these folders and files created in the same folder
* ``enlarge/enlargedxxxx.jpg`` created by ``Tracking.py``

#### Step 4:
* Run ``Render.py`` to get final video output
* ``football_full.avi`` is rendered

#### Explanation:
* ``Stitch.py`` (with the help of util.py) takes in the three football videos and * stitches them into one video and then generates individual video frames. It can take quite awhile, roughly 2-3 hours and can take up quite a big of space, roughly 8GB.

* ``Tracking.py`` then reads in the frames generated and output the topdown view and tracking of the players on the field in individual frames

* ``Render.py`` finally reads in all the frames from the previous step to recreate the video in a lower resolution
