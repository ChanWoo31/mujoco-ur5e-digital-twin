import mujoco
import mujoco.viewer

import numpy as np
import os
import time

def deg2rad(x):
    y = x*np.pi/180
    return y

m = mujoco.MjModel.from_xml_path('my_xml.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()

        d.ctrl[0] = deg2rad(-45)

        mujoco.mj_step(m, d)

        viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)