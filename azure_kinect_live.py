import cv2
import torch
import argparse
import numpy as np
import open3d as o3d
from collections import deque
from PIL import Image
from cotracker.utils.azurekinect_visualizer import Visualizer
from cotracker.predictor_live import CoTrackerPredictor
from polygon_draw import PolygonDrawer
import matplotlib.pyplot as plt
from pdb import set_trace as bp


def preprocess_frame(frame):
    return np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# class ViewerWithCallback:

#     def __init__(self, config, device, align_depth_to_color):
#         self.flag_exit = False
#         self.align_depth_to_color = align_depth_to_color

#         self.sensor = o3d.io.AzureKinectSensor(config)
#         if not self.sensor.connect(device):
#             raise RuntimeError('Failed to connect to sensor')

#     def escape_callback(self, vis):
#         self.flag_exit = True
#         return False

#     def run(self):
#         glfw_key_escape = 256
#         vis = o3d.visualization.VisualizerWithKeyCallback()
#         vis.register_key_callback(glfw_key_escape, self.escape_callback)
#         vis.create_window('viewer', 1920, 540)
#         print("Sensor initialized. Press [ESC] to exit.")

#         vis_geometry_added = False
#         while not self.flag_exit:
#             rgbd = self.sensor.capture_frame(self.align_depth_to_color)
#             if rgbd is None:
#                 continue

#             if not vis_geometry_added:
#                 vis.add_geometry(rgbd)
#                 vis_geometry_added = True

#             vis.update_geometry(rgbd)
#             vis.poll_events()
#             vis.update_renderer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="input json kinect config"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list available azure kinect sensors"
    )
    parser.add_argument(
        "--cap_device",
        type=int,
        default=0,
        help="input kinect device id"
    )
    parser.add_argument(
        "-a",
        "--align_depth_to_color",
        action="store_true",
        help="enable align depth image to color"
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/cotracker_stride_8_wind_16.pth",
        help="cotracker model",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=30,
        help="Regular grid size"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="GPU selection"
    )

    args = parser.parse_args()

    if args.list:
        o3d.io.AzureKinectSensor.list_devices()
        exit()

    if args.config is not None:
        config = o3d.io.read_azure_kinect_sensor_config(args.config)
    else:
        config = o3d.io.AzureKinectSensorConfig()

    cap_device = args.cap_device
    if cap_device < 0 or cap_device > 255:
        print("Unsupported device id, fall back to 0")
        cap_device = 0

    sensor = o3d.io.AzureKinectSensor(config)
    if not sensor.connect(cap_device):
        raise RuntimeError("Failed to connect to sensor")

    S = 8

    new_frame_counter = 0
    new_frame_req = S
    frame_buffer = deque(maxlen=S)

    vis = Visualizer(pad_value=120, linewidth=3)

    # FPS
    curr_frame_count = 0

    ########################TEST CODE BLOCK 
    # while True:
    #     f = sensor.capture_frame(args.align_depth_to_color)
    #     curr_frame_count+=1
    #     if curr_frame_count==27:
    #         print(np.asarray(f.depth).shape)
    #         print(np.asarray(f.color).shape)
    ############################################
    while True:
        frame = sensor.capture_frame(args.align_depth_to_color)

        curr_frame_count += 1

        if curr_frame_count == 7:
            #Take the RGB stream input from camera
            rgb_frame = np.asarray(frame.color)
            depth_frame = np.asarray(frame.depth) 
            
            # trigger UI to create a segmentation mask
            polydraw = PolygonDrawer(
                frame=preprocess_frame(rgb_frame), window_name="Mask Selector"
            )
            segm_mask = polydraw.run()
            print(segm_mask.shape)
            background_indices_x, background_indices_y = np.where(
                segm_mask != 255
            )[0], np.where(segm_mask != 255)[1]
            segm_mask[background_indices_x, background_indices_y] = 0.0
            plt.imsave("polygon.png", segm_mask.astype(np.uint8))
            segm_mask = torch.from_numpy(
                np.expand_dims(np.mean(segm_mask, axis=-1), axis=(0, 1))
            )
            print(segm_mask.shape)

            model = CoTrackerPredictor(
                device=args.device,
                checkpoint=args.checkpoint,
                grid_size=args.grid_size,
                segm_mask=segm_mask,
            )

            model = model.to(args.device)

        if curr_frame_count > 7:
            frame = preprocess_frame(rgb_frame)
            frame_buffer.append(frame)

            new_frame_counter = (new_frame_counter + 1) % new_frame_req
            if new_frame_counter == 0:
                # Initially we have to wait for #window_size frames. Afterwards we only need the next #stride_size frames.
                new_frame_req = S // 2

                frames = np.stack(frame_buffer)
                frames = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float()
                frames = frames.to(args.device)

                pred_tracks, pred_visibility = model(frames)

                res_video = vis.visualize(
                    frames[:, : S // 2],
                    pred_tracks[:, : S // 2],
                    pred_visibility[:, : S // 2],
                    segm_mask=segm_mask.to(args.device),
                    save_video=False,
                    query_frame=0,
                ).squeeze(0).permute(0, 2, 3, 1)
                # Convert to numpy and convert color
                res_video = np.array(res_video)
                res_video = [
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in res_video
                ]

                for res_frame in res_video:
                    cv2.imshow("Capture", res_frame)
                    cv2.waitKey(1)  # artificially fake a 25 fps frame rate

            if cv2.waitKey(1) == ord("q"):
                break

    sensor.disconnect()
