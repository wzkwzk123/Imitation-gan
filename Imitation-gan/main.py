from DataProcess import data_save_read
from param import Params
from GAN import Gan
import CartPoleControl
import random


argspid = Params().get_pid_args()
argsmain = Params().get_main_args()

fitness = CartPoleControl.CratpoleControl(argspid) # Generate the data from pid control, Return True

"""
Get data from pid simulite
"""
data = data_save_read.read_data('input_data.xlsx')
random.shuffle(data)
data_obs = []
data_label = []
for i in range(len(data)):
    # Get Train Data And Label
    print('i {}, data {}, label {}'.format(i, data[i][0:-1], data[i][-2:-1]))
    data_obs.append(data[i][0:-1])
    data_label.append(data[i][-2:-1])
# real_obs, real_action, num_steps, log_every, args
model = Gan(data_obs, data_label, 100, 10, argsmain)
model.train()