import torch 
import sys 
from focalnet_pt import FocalNet, load_focalnet as load_focalnetPT
from focalnet.focalnet import FocalNet as FocalNetTF
from focalnet.focalnet_utils import load_focalnet
from tensorflow.keras.layers import Conv2D, Dense, Conv1D, LayerNormalization, Input 
from tensorflow.keras import Sequential, Model 
import tensorflow as tf 



def conversion(ckpt, model_tf):
    for key, value in ckpt.items():
        if key[:6] == "layers":
            key = 'layers_' + key[6:]
        #key = key.replace("layers.", ".layers_.")
        names = key.split(".")
        curr = None 
        #print(names)
        for i, name in enumerate(names):
            if i != len(names) - 1 :
                if not name.isnumeric():
                    curr = getattr(model_tf, name) if curr is None else getattr(curr, name)
                else:
                    curr = curr[int(name)]
                if type(curr) == Sequential:
                    curr = curr.layers
            else:
                if name == "weight":
                    if type(curr) == Dense:
                        curr.kernel.assign(value.cpu().numpy().T)
                    elif type(curr) == Conv2D:
                        #print(key)
                        curr.kernel.assign(value.cpu().numpy().transpose((2, 3, 1, 0)))
                    elif type(curr) == LayerNormalization:
                        curr.gamma.assign(value.cpu().numpy() )
                elif name == "bias":
                    if type(curr) == Dense:
                        curr.bias.assign(value.cpu().numpy())
                    elif type(curr) == Conv2D:
                        curr.bias.assign(value.cpu().numpy())
                    elif type(curr) == LayerNormalization:
                        curr.beta.assign(value.cpu().numpy())
                elif name == "gamma_1":
                    curr.gamma_1.assign(value.cpu().numpy())
                elif name == "gamma_2":
                    curr.gamma_2.assign(value.cpu().numpy())
                else:
                    print(name)
                    print(key)
                    raise("error")
    return model_tf
img_size = 224
import numpy as np 
x = torch.rand(2, 3, img_size, img_size)  
x_tf = x.numpy().transpose((0,2,3,1))


model_name = "focalnet_small_lrf" # #"focalnet_xlarge_lrf_384" #"focalnet_huge_fl3" #"_fl4" 
model_name_tf ="focalnet_small_lrf"  #"focalnet_xlarge_fl3" # #"focalnet_huge_fl3" 

model = load_focalnetPT(model_name=model_name_tf)
model.eval()


weights = torch.load(f"ckpt/{model_name}.pth")["model"]
model.load_state_dict(weights)

with torch.no_grad():
    out = model(x)
print(out[:,:15])


## -- TF -- ##
# 1000 21842
model_tf = load_focalnet(model_name=model_name_tf, input_shape=(img_size, img_size, 3), pretrained=False, return_model=False, num_classes=1000 )#FocalNetTF(depths=[2, 2, 6, 2], embed_dim=96, focal_levels=focal_levels)
model_tf(x_tf)
model_tf = conversion(weights, model_tf)


model_tf.save_weights(f"../focalnet/tmp/{model_name_tf}.h5")
out2 = model_tf(x_tf)
print(out2[:,:15])

assert np.abs(out2.numpy() - out.numpy()).mean() < 1e-2