import SimpleITK
import numpy as np
import cv2
from pandas import DataFrame
from pathlib import Path
from scipy.ndimage import center_of_mass, label
from pathlib import Path
from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    DataFrameValidator,
)
from typing import (Tuple)
from evalutils.exceptions import ValidationError
import random
from typing import Dict
import json

from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sklearn.metrics
import torchvision
#from dataset import ClassificationTrainDataset, ClassificationValidDataset
#from model import TimeSeriesModel
#import shutil
#import seaborn as sns
import torch.nn as nn
import numpy as np
from scipy import stats

####
# Toggle the variable below to debug locally. The final container would need to have execute_in_docker=True
# Fix fillna
####
execute_in_docker = True

def vectorized_majority_filter(data, window_size=3):
    # Ensure the window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Extend data at both ends to handle boundaries
    extension = window_size // 2
    extended_data = np.pad(data, (extension, extension), mode='edge')
    
    # Create a sliding window view of the data
    shape = (extended_data.shape[0] - window_size + 1, window_size)
    strides = (extended_data.strides[0], extended_data.strides[0])
    sliding_window = np.lib.stride_tricks.as_strided(extended_data, shape=shape, strides=strides)
    
    # Compute the mode within each window
    mode_results = stats.mode(sliding_window, axis=1)[0].flatten()
    
    return mode_results

class VideoLoader():
    def load(self, *, fname):
        path = Path(fname)
        print(path)
        if not path.is_file():
            raise IOError(
                f"Could not load {fname} using {self.__class__.__qualname__}."
            )
            #cap = cv2.VideoCapture(str(fname))
        #return [{"video": cap, "path": fname}]
        return [{"path": fname}]

# only path valid
    def hash_video(self, input_video):
        pass


class UniqueVideoValidator(DataFrameValidator):
    """
    Validates that each video in the set is unique
    """

    def validate(self, *, df: DataFrame):
        try:
            hashes = df["video"]
        except KeyError:
            raise ValidationError("Column `video` not found in DataFrame.")

        if len(set(hashes)) != len(hashes):
            raise ValidationError(
                "The videos are not unique, please submit a unique video for "
                "each case."
            )


class SurgVU_classify(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            index_key='input_video',
            file_loaders={'input_video': VideoLoader()},
            input_path=Path("/input/") if execute_in_docker else Path("./test/"),
            output_file=Path("/output/surgical-step-classification.json") if execute_in_docker else Path(
                "./output/surgical-step-classification.json"),
            validators=dict(
                input_video=(
                    #UniqueVideoValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        #self.num_frames = 16
        

        self.train_transform = A.Compose([
            #A.Resize(256, 320, p=1.0),
            A.ShiftScaleRotate(p=1.0),
            A.RandomCrop(224, 224, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        
            A.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
            ToTensorV2()  # Convert image and mask to torch.Tensor
        ], additional_targets={'image'+'0'*i:'image' for i in range(1, 16)})
    
        
        ###                                                                                                     ###
        ###  TODO: adapt the following part for creating your model and loading weights
        ###                                                                                                     ###
        
        self.step_list = ["range_of_motion",
                          "rectal_artery_vein",
                          "retraction_collision_avoidance",
                          "skills_application",
                          "suspensory_ligaments",
                          "suturing",
                          "uterine_horn",
                          "other"]
        # Comment for docker build
        # Comment for docker built

        print(self.step_list)

    def dummy_step_prediction_model(self):
        random_step_prediction = random.randint(0, len(self.step_list)-1)

        return random_step_prediction
    
    def step_predict_json_sample(self):
        single_output_dict = {"frame_nr": 1, "surgical_step": None}
        return single_output_dict

    def process_case(self, *, idx, case):

        # Input video would return the collection of all frames (cap object)
        input_video_file_path = case #VideoLoader.load(case)
        # Detect and score candidates
        scored_candidates = self.predict(case.path) #video file > load evalutils.py

        # return
        # Write resulting candidates to result.json for this case
        return scored_candidates

    def save(self):
        print('Saving prediction results to ' + str(self._output_file))
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)


    def predict(self, fname) -> Dict:
        """
        Inputs:
        fname -> video file path
        
        Output:
        tools -> list of prediction dictionaries (per frame) in the correct format as described in documentation 
        """
        
        print('Video file to be loaded: ' + str(fname))
        cap = cv2.VideoCapture(str(fname))
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        clip_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (320,256))
            #frame = self.valid_transform(image=frame)['image']
            clip_frames.append(frame)
        cap.release()
        
        print('No. of frames: ', video_length, len(clip_frames))

        stride = 4
        
        if video_length >= 128:
            frame_list = [128, 32, 16]
            weight_list = [0.975, 0.65, 0.2]
        else:
            frame_list = [32, 16]
            weight_list = [0.65, 0.2]

        all_res_array = torch.zeros([8, video_length], dtype=torch.float)

        for num_frames, weight in zip(frame_list, weight_list):
            print(f'using frame {num_frames} weights, ensemble weight: {weight}')
            self.model_dic = {}
            for fold in range(4):
    
                self.model_dic[fold] = torchvision.models.video.swin3d_b(weights=None)
                self.model_dic[fold].head = nn.Linear(in_features=1024, out_features=8, bias=True)
    
                self.model_dic[fold] = self.model_dic[fold].to('cuda')
    
                if execute_in_docker:
                    ddp_sd = torch.load(f'/opt/algorithm/runs/swin_cutmix_3d_1.0_epoch_3200_frame_{num_frames}_repeat_fold_{fold}/best.pth')
                else:
                    ddp_sd = torch.load(f'runs/swin_cutmix_3d_1.0_epoch_3200_frame_{num_frames}_repeat_fold_{fold}/best.pth')
                    
                self.model_dic[fold].load_state_dict({k.replace('_orig_mod.module.', ''):v for k, v in ddp_sd.items()})
                self.model_dic[fold].eval()

                print(f'fold {fold} loaded.')
            
            res_array = torch.zeros([8, video_length], dtype=torch.float)
            overlap_array = torch.zeros([8, video_length], dtype=torch.float)
            
            for start in tqdm(range(video_length-num_frames+1)):
                end = start + num_frames
                if start % stride == 0 or end == video_length:
                    #print(list(range(start,end,(num_frames//16))))
                    vid_images = clip_frames[start:end:(num_frames//16)]
                    vid_images = {'image'+'0'*idx: img for idx, img in enumerate(vid_images)}
                    vid_images = self.train_transform(**vid_images)
                    vid_images = [vid_images['image'+'0'*i] for i in range(len(vid_images))]
                    tensor = torch.stack(vid_images, dim=1).unsqueeze(0).to('cuda')
            
                    with torch.no_grad():
    
                        res = (self.model_dic[0](tensor).squeeze().softmax(0)+self.model_dic[1](tensor).squeeze().softmax(0)+self.model_dic[2](tensor).squeeze().softmax(0)+self.model_dic[3](tensor).squeeze().softmax(0))/4
                        #res = self.model_dic[0](tensor).squeeze().softmax(0)
                        #print(res)
                    
                    res_array[:, start:end] += torch.stack([res.cpu()]*num_frames, dim=-1)#res#
                    overlap_array[:, start:end] += 1
    
            mean_array = res_array / overlap_array
            all_res_array += mean_array * weight
            
        final_pred = np.array(all_res_array.argmax(0))

        if final_pred.shape[0] > 140:
            final_pred = vectorized_majority_filter(final_pred, window_size=140)
        
                #break

        # generate output json
        all_frames_predicted_outputs = []
        for i in range(video_length):
            frame_dict = self.step_predict_json_sample()
            #step_detection = self.dummy_step_prediction_model()

            frame_dict['frame_nr'] = i
            
            frame_dict["surgical_step"] = int(final_pred[i])
            #print(frame_dict)

            all_frames_predicted_outputs.append(frame_dict)

        steps = all_frames_predicted_outputs
        return steps



if __name__ == "__main__":
    SurgVU_classify().process()
