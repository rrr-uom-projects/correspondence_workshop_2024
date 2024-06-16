# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')

chkpt_folder = "/home/ed/postdoc/correspondence_workshop/models/"

def get_timestr():
    now = datetime.now()
    time_stamp = now.strftime("%Y_%m_%d__%H_%M_%S")
    print("Time stamp: ", time_stamp)
    return time_stamp

def save_path(folder_str=None):
    if folder_str is None:
        folder_str = get_timestr()

    folder_path_models = os.path.join(chkpt_folder, folder_str)
    return folder_path_models