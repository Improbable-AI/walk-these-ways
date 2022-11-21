import lcm
import threading
import time
import select

import numpy as np

from go1_gym_deploy.lcm_types.leg_control_data_lcmt import leg_control_data_lcmt
from go1_gym_deploy.lcm_types.rc_command_lcmt import rc_command_lcmt
from go1_gym_deploy.lcm_types.state_estimator_lcmt import state_estimator_lcmt
from go1_gym_deploy.lcm_types.vicon_pose_lcmt import vicon_pose_lcmt
from go1_gym_deploy.lcm_types.camera_message_lcmt import camera_message_lcmt
from go1_gym_deploy.lcm_types.camera_message_rect_wide import camera_message_rect_wide
from go1_gym_deploy.lcm_types.camera_message_rect_wide_mask import camera_message_rect_wide_mask


class UnitreeLCMInspector:
    def __init__(self, lc):
        self.lc = lc

        self.camera_names = ["front", "bottom", "left", "right", "rear"]
        for cam_name in self.camera_names:
            self.camera_subscription = self.lc.subscribe(f"rect_image_{cam_name}", self._rect_camera_cb)
        for cam_name in self.camera_names:
            self.camera_subscription = self.lc.subscribe(f"rect_image_{cam_name}_mask", self._mask_camera_cb)

        self.camera_image_left = None
        self.camera_image_right = None
        self.camera_image_front = None
        self.camera_image_bottom = None
        self.camera_image_rear = None

        self.ts = [time.time(), time.time(), time.time(), time.time(), time.time(),]

        self.num_low_states = 0

    def _rect_camera_cb(self, channel, data):

        # message_types = [camera_message_rect_front, camera_message_rect_front_chin, camera_message_rect_left,
        #                  camera_message_rect_right, camera_message_rect_rear_down]
        # image_shapes = [(200, 200, 3), (100, 100, 3), (100, 232, 3), (100, 232, 3), (200, 200, 3)]

        message_types = [camera_message_rect_wide, camera_message_rect_wide, camera_message_rect_wide, camera_message_rect_wide, camera_message_rect_wide]
        image_shapes = [(116, 100, 3), (116, 100, 3), (116, 100, 3), (116, 100, 3), (116, 100, 3)]

        cam_name = channel.split("_")[-1]

        cam_id = self.camera_names.index(cam_name) + 1

        print(channel, message_types[cam_id - 1])
        msg = message_types[cam_id - 1].decode(data)

        img = np.fromstring(msg.data, dtype=np.uint8)
        img = np.flip(np.flip(
            img.reshape((image_shapes[cam_id - 1][2], image_shapes[cam_id - 1][1], image_shapes[cam_id - 1][0])),
            axis=0), axis=1).transpose(1, 2, 0)

        if cam_id == 1:
            self.camera_image_front = img
        elif cam_id == 2:
            self.camera_image_bottom = img
        elif cam_id == 3:
            self.camera_image_left = img
        elif cam_id == 4:
            self.camera_image_right = img
        elif cam_id == 5:
            self.camera_image_rear = img
        else:
            print("Image received from camera with unknown ID#!")

        print(f"f{1. / (time.time() - self.ts[cam_id - 1])}: received py from {cam_name}!")
        self.ts[cam_id-1] = time.time()

        from PIL import Image
        im = Image.fromarray(img)
        im.save(f"{cam_name}_image.jpeg")

    def _mask_camera_cb(self, channel, data):

        message_types = [camera_message_rect_wide_mask for i in range(5)]
        image_shapes = [(116, 100, 1) for i in range(5)]

        cam_name = channel.split("_")[-2]

        cam_id = self.camera_names.index(cam_name) + 1

        print(channel, message_types[cam_id - 1])
        msg = message_types[cam_id - 1].decode(data)

        img = np.array(list(msg.data)).astype(np.uint8)
        img = np.flip(np.flip(
            img.reshape((image_shapes[cam_id - 1][2], image_shapes[cam_id - 1][1], image_shapes[cam_id - 1][0])),
            axis=0), axis=1)

        if cam_id == 1:
            self.camera_image_front = img
        elif cam_id == 2:
            self.camera_image_bottom = img
        elif cam_id == 3:
            self.camera_image_left = img
        elif cam_id == 4:
            self.camera_image_right = img
        elif cam_id == 5:
            self.camera_image_rear = img
        else:
            print("Image received from camera with unknown ID#!")

        print(f"f{1. / (time.time() - self.ts[cam_id - 1])}: received py from {cam_name}!")
        self.ts[cam_id-1] = time.time()

        from PIL import Image
        # print(img[0])
        im = Image.fromarray(img[0])
        im.save(f"{cam_name}_mask_image.jpeg")

    def publish_30Hz(self):
        msg = camera_message_rect_wide()
        msg.data = [0] * 34800
        self.lc.publish("rect_image_rear", msg.encode())
        time.sleep(1./30.)

    def poll(self, cb=None):
        t = time.time()
        try:
            while True:
                timeout = 0.01
                rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
                if rfds:
                    # print("message received!")
                    self.lc.handle()
                    # print(f'Freq {1. / (time.time() - t)} Hz'); t = time.time()
                else:
                    continue
                    # print(f'waiting for message... Freq {1. / (time.time() - t)} Hz'); t = time.time()
                #    if cb is not None:
                #        cb()
        except KeyboardInterrupt:
            pass

    def spin(self):
        self.run_thread = threading.Thread(target=self.poll, daemon=False)
        self.run_thread.start()

if __name__ == "__main__":
    import lcm

    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
    #check_lcm_msgs()
    print("init")
    insp = UnitreeLCMInspector(lc)
    print("polling")
    # insp.poll()
    while True:
        insp.publish_30Hz()
