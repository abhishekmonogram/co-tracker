import pyzed.sl as sl
import cv2
import numpy as np

import sys
import viewer as gl
import pyzed.sl as sl
import argparse

from polygon_draw import PolygonDrawer

from PIL import Image
from cotracker.utils.azurekinect_visualizer import Visualizer
from cotracker.predictor_live import CoTrackerPredictor
import open3d as o3d
import torch
from collections import deque
import matplotlib.pyplot as plt
# from pdb import set_trace as bp

torch.set_grad_enabled(False)

def preprocess_frame(frame):
    return np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def parse_args(init):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        init.camera_fps = 100
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")



def main():
    print("Running Depth Sensing sample ... Press 'Esc' to quit\nPress 's' to save the point cloud")

    init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
    parse_args(init)
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    #Set camera resolution
    camera_model = zed.get_camera_information().camera_model
    res = zed.get_camera_information().camera_configuration.resolution


    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(1, sys.argv, camera_model, res)

    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_zed = sl.Mat(res.width, res.height, sl.MAT_TYPE.U8_C4)

    #Queue width for cotracker
    S = 8

    new_frame_counter = 0
    new_frame_req = S
    frame_buffer = deque(maxlen=S)

    vis = Visualizer(pad_value=120, linewidth=3)


    curr_frame_count = 0

    while viewer.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            curr_frame_count+=1
            #Take the RGB stream input from camera
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            # Use get_data() to get the numpy array
            image_ocv = image_zed.get_data()
            frame_rgb = preprocess_frame(image_ocv)


            if curr_frame_count ==1:
                # trigger UI to create a segmentation mask
                polydraw = PolygonDrawer(
                    frame=(frame_rgb), window_name="Mask Selector"
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
                    device=opt.device,
                    checkpoint=opt.checkpoint,
                    grid_size=opt.grid_size,
                    segm_mask=segm_mask,
                )

                model = model.to(opt.device)


            if curr_frame_count > 1:
                frame_buffer.append(image_ocv[:,:,:3])

                new_frame_counter = (new_frame_counter + 1) % new_frame_req

                if new_frame_counter == 0:
                    # Initially we have to wait for #window_size frames. Afterwards we only need the next #stride_size frames.
                    new_frame_req = S // 2

                    frames = np.stack(frame_buffer)
                    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float()
                    frames = frames.to(opt.device)

                    pred_tracks, pred_visibility = model(frames)

                    res_video = vis.visualize(
                        frames[:, : S // 2],
                        pred_tracks[:, : S // 2],
                        pred_visibility[:, : S // 2],
                        segm_mask=segm_mask.to(opt.device),
                        save_video=False,
                        query_frame=0,
                    ).squeeze(0).permute(0, 2, 3, 1)
                    # Convert to numpy and convert color
                    res_video = np.array(res_video)
                    res_video = [frame for frame in res_video]
                    # res_video = [
                    #     cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in res_video
                    # ]

                    for res_frame in res_video:
                        cv2.imshow("Capture", res_frame)
                        cv2.waitKey(1)  # artificially fake a 25 fps frame rate

                if cv2.waitKey(1) == ord("q"):
                    break
            # print(frame_buffer[0].shape)

            # print(f'{image_ocv.shape=}')
            # print(f'{type(image_ocv)}')
            # bp()
            # Display the left image from the numpy array
            # cv2.imwrite('RGB stream.jpg',image_ocv)
            # viewer_rgb.update(image_ocv)


            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
            viewer.updateData(point_cloud)
            # bp()
            # print(dir(point_cloud))
            point_cloud_data = point_cloud.get_data()
            # point3D = point_cloud.get_value(33,33)
            # print(f'{point3D=}')
            # print(f'{point_cloud_data.shape=}')
            # bp()

            if(viewer.save_data == True):
                point_cloud_to_save = sl.Mat()
                zed.retrieve_measure(point_cloud_to_save, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
                err = point_cloud_to_save.write('Pointcloud.ply')
                if(err == sl.ERROR_CODE.SUCCESS):
                    print("Current .ply file saving succeed")
                else:
                    print("Current .ply file failed")
                viewer.save_data = False
    viewer.exit()
    zed.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    parser.add_argument('--device', type=str, help='GPU(cuda) or CPU(cpu)', default = 'cuda')
    parser.add_argument("--checkpoint",default="./checkpoints/cotracker_stride_8_wind_16.pth",help="cotracker model",)
    parser.add_argument("--grid_size",type=int,default=30,help="Regular grid size")
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.iogl_viewer.p_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main() 
