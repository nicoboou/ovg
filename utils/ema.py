#!/usr/bin/env python
# coding=utf-8

import torch 
import torch .nn as nn 
from typing import Dict ,Any 
import copy 


class ExponentialMovingAverage :
    def __init__ (
    self ,
    model :nn .Module ,
    decay :float =0.9999 ,
    min_decay :float =0.0 ,
    update_after :int =100 ,
    inv_gamma :float =1.0 ,
    power :float =2 /3 ,
    ):

        self .decay =decay 
        self .min_decay =min_decay 
        self .update_after =update_after 
        self .inv_gamma =inv_gamma 
        self .power =power 
        self .step =0 
        self .ema_model =copy .deepcopy (model )
        self .ema_model .eval ()

        for param in self .ema_model .parameters ():
            param .requires_grad_ (False )

    def get_decay (self ,step :int )->float :

        step =max (0 ,step -self .update_after )
        warmup_decay =(1 +step )/(10 +step )
        return max (self .min_decay ,min (self .decay ,warmup_decay ))

    def update (self ,model :nn .Module ,step :int =None ):
        if step is not None :
            self .step =step 
        else :
            self .step +=1 

        if self .step <=self .update_after :
            return 

        decay =self .get_decay (self .step )

        with torch .no_grad ():
            for ema_param ,model_param in zip (self .ema_model .parameters (),model .parameters ()):
                if model_param .requires_grad :
                    ema_param .mul_ (decay ).add_ (model_param ,alpha =1 -decay )

    def copy_to (self ,model :nn .Module ):
        with torch .no_grad ():
            for ema_param ,model_param in zip (self .ema_model .parameters (),model .parameters ()):
                if model_param .requires_grad :
                    model_param .copy_ (ema_param )

    def state_dict (self )->Dict [str ,Any ]:
        return {
        "ema_model":self .ema_model .state_dict (),
        "step":self .step ,
        "decay":self .decay ,
        "min_decay":self .min_decay ,
        "update_after":self .update_after ,
        "inv_gamma":self .inv_gamma ,
        "power":self .power ,
        }

    def load_state_dict (self ,state_dict :Dict [str ,Any ]):
        self .ema_model .load_state_dict (state_dict ["ema_model"])
        self .step =state_dict ["step"]
        self .decay =state_dict ["decay"]
        self .min_decay =state_dict ["min_decay"]
        self .update_after =state_dict ["update_after"]
        self .inv_gamma =state_dict .get ("inv_gamma",1.0 )
        self .power =state_dict .get ("power",2 /3 )

    def to (self ,device ):

        self .ema_model .to (device )
        return self 


class EMAModelWrapper :


    def __init__ (self ,model :nn .Module ,ema :ExponentialMovingAverage ):
        self .model =model 
        self .ema =ema 
        self .original_state =None 

    def __enter__ (self ):
        self .original_state =copy .deepcopy (self .model .state_dict ())
        self .ema .copy_to (self .model )
        return self .model 

    def __exit__ (self ,exc_type ,exc_val ,exc_tb ):
        self .model .load_state_dict (self .original_state )
