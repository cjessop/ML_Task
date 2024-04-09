import torch 
import torchvision.transforms as transforms
from torchvision import models
import cv2
from ultralytics import YOLO 
import pickle
import pandas as pd
import uuid
import os 
import time
import numpy as np

class YOLO_detector():
    def __init__(self, model, model_path, pretrained=True):
        self.pretrained=pretrained
        self.model_path = model_path
        self.model = models.resnet50(pretrained=self.pretrained)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.transform=transforms.Compose([
            transforms.ToPILImagte(),
            transforms.resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4,0.4,0.4], std=[0.2,0.2,0.2])
        ])

    def predict(self, image):
        im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_tensor = self.transform(im_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(im_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        orig_h, orig_w = image.shape[:2]
        keypoints[::2] *= orig_w / 224.0
        keypoints[1::2] *= orig_h / 224.0

        return keypoints
    
    def draw_keyP(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
            cv2.circle(image, (x,y), 5, (0,0,255), -1)
        return image
    
    def draw_keyP_video(self, video_frames, keypoints):
        out_vid_frames = []
        for frame in video_frames:
            frame = self.draw_keyP(frame, keypoints)
            out_vid_frames.append(frame)
        return out_vid_frames


class Object_tracker():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.columns = ['x1', 'x2', 'y1', 'y2']
        UtilsObj = Utils()

    def choose_and_filter(self, keypoints, obj_detections):
        obj_detections_frame_one = obj_detections[0]
        chosen_obj = self.choose_obj(keypoints, obj_detections_frame_one)
        filtered_obj_detections = []
        for obj_dict in obj_detections:
            filtered_obj_dict = {track_id: bbox for track_id, bbox in obj_dict.items() if track_id in chosen_obj}
            filtered_obj_detections.append(filtered_obj_dict)
        return filtered_obj_detections
    
    def choose_obj(self, keypoints, obj_dict):
        distances=[]
        for track_id, bbox in obj_dict.items():
            obj_centre = Utils.get_box_centre(bbox)

            distance_min = float('inf')
            for i in range(0, len(keypoints), 2):
                keypoints = (keypoints[i], keypoints[i+1])
                distance = Utils.calc_distance(obj_centre, keypoints)
                if distance < distance_min:
                    distance_min = distance
            distances.append(track_id, distance_min)

        distances.sort(key = lambda x: x[1])
        chosen_obj = [distances[0][0], distances[1][0]]
        return chosen_obj
                    

    def interp_obj_pos(self, obj_pos):
        obj_pos = [x.get(1,[]) for x in obj_pos]
        df_obj_pos = pd.DataFrame(obj_pos, columns=self.columns)
        df_obj_pos = df_obj_pos.interpolate()
        df_obj_pos = df_obj_pos.bfill()

        obj_pos = [{1:x} for x in df_obj_pos.to_numpy().tolist()]

        return obj_pos
    
    def get_obj_shot_frames(self, obj_pos):
        obj_pos = [x.get(1, []) for x in obj_pos]
        df_obj_pos = pd.DataFrame(obj_pos, columns=self.columns)
        df_obj_pos['obj_hit'] = 0
        df_obj_pos['mid_y'] = (df_obj_pos['y1'] + df_obj_pos['y2']) / 2
        df_obj_pos['mid_y_roll_mean'] = df_obj_pos['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_obj_pos['del_y'] = df_obj_pos['mid_y_roll_mean'].diff()

        min_frame_for_change = 30
        for i in range(1, len(df_obj_pos)- int(min_frame_for_change * 1.2)):
            negative_pos_change = df_obj_pos['del_y'].iloc[i] > 0 and df_obj_pos['del_y'].iloc[i+1] < 0
            positive_pos_change = df_obj_pos['del_y'].iloc[i] < 0 and df_obj_pos['del_y'].iloc[i+1] > 0
            if negative_pos_change or positive_pos_change:
                change_counter = 0
                for change_frame in range(i+1, i + int(min_frame_for_change * 1.2)+1):
                    negative_pos_change_next_frame = df_obj_pos['del_y'].iloc[i] > 0 and df_obj_pos['del_y'].iloc[change_frame] < 0
                    positive_pos_change_next_frame = df_obj_pos['del_y'].iloc[i] < 0 and df_obj_pos['del_y'].iloc[change_frame] > 0
                    if negative_pos_change and negative_pos_change_next_frame:
                        change_counter += 1
                    elif positive_pos_change and positive_pos_change_next_frame:
                        change_counter += 1

                    if change_counter > min_frame_for_change-1:
                        df_obj_pos['obj_hit'].iloc[i] = 1

        frame_nums_w_hit = df_obj_pos[df_obj_pos['obj_hit']==1].index.tolist()

        return frame_nums_w_hit
    
    def frames_detect(self, frames, read_from_stub=False, stub_path=None):
        obj_detections= []
        if read_from_stub and stub_path:
            with open(stub_path, 'rb') as f:
                obj_detections = pickle.load(f)
            return obj_detections
        for frame in frames:
            player_dict = self.frame_detect(frame)
            obj_detections = obj_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(obj_detections, f)
        
        return obj_detections


    def frame_detect(self, frame):
        res = self.model.predict(frame, conf=0.15[0])
        obj_dict = {}
        for box in res.boxes:
            res = box.xyxy.tolist()[0]
            obj_dict[1] = res

        return obj_dict
    
    def draw_bbox(self, video_frames, obj_detection):
        out_vid_frames = []
        for frame, obj_dict in zip(video_frames, obj_detection):
            for track_id, bbox in obj_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame,f"Object ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            out_vid_frames.append(frame)
        
        return out_vid_frames


class Utils():
    def __init__(self, video_path=None, out_vid_frames=None, out_vid_path=None):
        self.video_path = video_path
        self.out_vid_frames = out_vid_frames
        self.out_vid_path = out_vid_path

    def read_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        return frames

    def save_video(self):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_vid_path, fourcc, 24, (self.out_vid_frames[0].shape[1], self.out_vid_frames[0].shape[0]))
        for frame in self.out_vid_frames:
            out.write(frame)
        out.release()

    def get_box_centre(self, bbox):
        x1, y1, x2, y2 = bbox
        centre_x = int((x1 + x2) / 2)
        centre_y = int((y1 + y2) / 2)

        return centre_x, centre_y
    
    def calc_distance(self, p1, p2):
        return  ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

class YOLO_main():
    def __init__(self, data_path, image_path, image_number, model, video_path=None, video_capture=False):
        self.data_path = data_path
        self.image_path = image_path
        self.image_number = image_number
        self.video_capture = video_capture
        self.model = model
        self.video_path = video_path
    
    def train(self):
        IMAGES_PATH = os.path.join(str(self.data_path), str(self.image_path)) #/data/images
        labels = ['threat', 'non-threat']
        number_imgs = self.image_number

        if self.video_capture == True:
            cap = cv2.VideoCapture(0)
            # Loop through labels
            for label in labels:
                print('Collecting images for {}'.format(label))
                time.sleep(5)
                
                # Loop through the range of number of images
                for img_num in range(number_imgs):
                    print('Collecting images for {}, image number {}'.format(label, img_num))

                    ret, frame = cap.read() # Camera feed
                    imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
                    cv2.imwrite(imgname, frame)
                    
                    # Render to the screen
                    cv2.imshow('Image Collection', frame)
                    
                    # 2 second delay between captures
                    time.sleep(2)
                    
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()

            for label in labels:
                print('Collecting images for {}'.format(label))
                for img_num in range(number_imgs):
                    print('Collecting images for {}, image number {}'.format(label, img_num))
                    imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
                    print(imgname)   

            os.system("cd yolov5 && python train.py --img 320 --batch 16 --epochs 500 --data dataset.yml --weights yolov5s.pt --workers 2")

    def cam_capture(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Make detections 
            results = self.model(frame)
            
            cv2.imshow('YOLO', np.squeeze(results.render()))
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def video_stream(self):
        if self.video_path is not None:
            model = self.model
            cap = cv2.VideoCapture(self.video_path)
            while cap.isOpened():
                success, frame = cap.read()

                if not success:
                    exit()
                tracks = model.track(frame, show=False, persist=True)
         
                
    def main(self):
        input_video = "inputVideo.mp4"
        UtilObj = Utils(input_video)
        
        video_frames = UtilObj.read_video()
        ObjDetector = Object_tracker(model_path="path_to_model")

        Obj_detections = ObjDetector.frames_detect(video_frames, read_from_stub=True, stub_path="pathToModel")
        obj_shot_frames = ObjDetector.get_obj_shot_frames(Obj_detections)


        for obj_ind in range(len(obj_shot_frames) - 1):
            start_frame = obj_shot_frames[obj_ind]
            end_frame = obj_shot_frames[obj_ind + 1]
            obj_shot_time_seconds = (end_frame - start_frame) / 24

        output_video_frames = ObjDetector.draw_bbox(video_frames, Obj_detections)

        for i, frame in enumerate(output_video_frames):
            cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        
        UtilObj.save_video(output_video_frames, "output_videos/output_video.avi")



        

