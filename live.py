import os
import cv2
import torch
import argparse
import numpy as np
import time

from collections import deque

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor_live import CoTrackerPredictor
from polygon_draw import PolygonDrawer


def preprocess_frame(frame):
    return np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture_device",
        type=int,
        default=0,
        help="device to capture video from",
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/cotracker_stride_4_wind_8.pth",
        help="cotracker model",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=10,
        help="Regular grid size"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="GPU selection")

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.capture_device)
    if not cap.isOpened():
        print("Failed to open capture device")
        exit(1)
    
    S = 8

    #TODO Create segmentation mask here. And not in the video stream


    # model = CoTrackerPredictor(
    #     device=args.device,
    #     checkpoint=args.checkpoint,
    #     grid_size=args.grid_size,
    #     segm_mask=torch.zeros(1,1,640,480))

    # model = model.to(args.device)

    new_frame_counter = 0
    new_frame_req = S
    frame_buffer = deque(maxlen=S)

    vis = Visualizer(pad_value=120, linewidth=3)

    #FPS
    prev_frame_time = 0
    new_frame_time = 0

    curr_frame_count = 0
    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Stream closed")
            break
        
        curr_frame_count+=1
        
        if curr_frame_count == 1:
            #trigger UI to create a segmentation mask
            polydraw = PolygonDrawer(frame=frame, window_name='Mask Selector')
            segm_mask = polydraw.run()
            print(segm_mask.shape)
            segm_mask = torch.from_numpy(np.expand_dims(np.mean(segm_mask,axis=-1),axis=(0,1)))
            print(segm_mask.shape)

            model = CoTrackerPredictor(
            device=args.device,
            checkpoint=args.checkpoint,
            grid_size=args.grid_size,
            segm_mask=segm_mask)

            model = model.to(args.device)


        # font which we will be using to display FPS 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # time when we finish processing for this frame 
        new_frame_time = time.time() 
    
        # Calculating the fps 
        # fps will be number of frame processed in given time frame 
        # since their will be most of time error of 0.001 second 
        # we will be subtracting it to get more accurate result 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
    
        # converting the fps into integer 
        fps = int(fps) 
    
        # converting the fps to string so that we can display it on frame 
        # by using putText function 
        fps = str(fps) 
    
        # putting the FPS count on the frame 
        cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        frame = preprocess_frame(frame)
        frame_buffer.append(frame)

        new_frame_counter = (new_frame_counter + 1) % new_frame_req
        if new_frame_counter == 0:
            # Initially we have to wait for #window_size frames. Afterwards we only need the next #stride_size frames.
            new_frame_req = S // 2

            frames = np.stack(frame_buffer)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float()
            frames = frames.to(args.device)

            pred_tracks, pred_visibility = model(frames)

            res_video = vis.visualize(frames[:,:S//2], pred_tracks[:,:S//2], pred_visibility[:,:S//2], save_video=False, query_frame=0).squeeze(0).permute(0, 2, 3, 1)
            # Convert to numpy and convert color
            res_video = np.array(res_video)
            res_video = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in res_video]

            for res_frame in res_video:
                cv2.imshow('Capture', res_frame)
                cv2.waitKey(1) # artificially fake a 25 fps frame rate

        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()