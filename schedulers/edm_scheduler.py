from dataclasses import dataclass 
import math 
from typing import Optional 

import torch 
from diffusers .utils .outputs import BaseOutput 


@dataclass 
class EDMSchedulerOutput (BaseOutput ):
    prev_sample :Optional [torch .Tensor ]=None 
    next_sample :Optional [torch .Tensor ]=None 
    pred_original_sample :Optional [torch .Tensor ]=None 


class EDMSchedulerAdapter :
    def __init__ (self ,cfg ,device :torch .device )->None :
        self .config ={
        "edm_sigma_min":cfg .training .get ("edm_sigma_min",0.002 ),
        "edm_sigma_max":cfg .training .get ("edm_sigma_max",80.0 ),
        "edm_sigma_data":cfg .training .get ("edm_sigma_data",0.5 ),
        "prediction_type":"sample",
        "edm_schedule":cfg .training .get ("edm_schedule","karras"),
        "edm_rho":cfg .training .get ("edm_rho",7.0 ),
        }
        self .device =device 
        self .timesteps :Optional [torch .Tensor ]=None 
        self .timesteps_next :Optional [torch .Tensor ]=None 
        self .inversion_timesteps :Optional [torch .Tensor ]=None 

    def set_timesteps (self ,num_steps :int )->None :
        sigma_max =self .config ["edm_sigma_max"]
        sigma_min =self .config ["edm_sigma_min"]
        schedule =self .config .get ("edm_schedule","karras")
        rho =float (self .config .get ("edm_rho",7.0 ))
        self .config ["num_train_timesteps"]=num_steps 

        dtype =torch .get_default_dtype ()
        device =self .device 

        if schedule =="karras":

            ramp =torch .linspace (0 ,1 ,num_steps +1 ,device =device ,dtype =dtype )
            sigmas =(sigma_max **(1.0 /rho )+ramp *(sigma_min **(1.0 /rho )-sigma_max **(1.0 /rho )))**rho 
        else :

            sigmas =torch .linspace (
            math .log (sigma_max ),
            math .log (sigma_min ),
            num_steps +1 ,
            device =device ,
            dtype =dtype ,
            ).exp ()

        self .sigmas =sigmas 

        self .timesteps =self .sigmas [:-1 ]
        self .timesteps_next =self .sigmas [1 :]


        self .inversion_timesteps =self .sigmas .flip (0 )

    def _ode_step (
    self ,
    model_output :torch .Tensor ,
    sigma_cur :torch .Tensor ,
    sigma_next :torch .Tensor ,
    x_t :torch .Tensor ,
    )->torch .Tensor :
        sigma_data =self .config ["edm_sigma_data"]


        sigma_data_t =torch .as_tensor (sigma_data ,device =sigma_cur .device ,dtype =sigma_cur .dtype )
        c_skip =sigma_data_t **2 /(sigma_cur **2 +sigma_data_t **2 )
        c_out =sigma_cur *sigma_data_t /(sigma_cur **2 +sigma_data_t **2 ).sqrt ()
        x_0_pred =c_skip *x_t +c_out *model_output 


        d =(x_t -x_0_pred )/sigma_cur 
        dt =sigma_next -sigma_cur 
        x_prev_or_next =x_t +d *dt 
        return x_prev_or_next 

    def step (
    self ,
    model_output :torch .Tensor ,
    t_curr :torch .Tensor ,
    x_t :torch .Tensor ,
    )->EDMSchedulerOutput :
        t_scalar =t_curr .flatten ()[0 ]
        idx =torch .argmin (torch .abs (self .timesteps -t_scalar ))
        t_next =self .timesteps_next [idx ]

        x_prev =self ._ode_step (model_output ,t_scalar ,t_next ,x_t )
        return EDMSchedulerOutput (prev_sample =x_prev ,pred_original_sample =None )

    def invert_step (
    self ,
    model_output :torch .Tensor ,
    t_curr :torch .Tensor ,
    t_next :torch .Tensor ,
    x_t :torch .Tensor ,
    )->EDMSchedulerOutput :
        x_next =self ._ode_step (model_output ,t_curr ,t_next ,x_t )
        return EDMSchedulerOutput (prev_sample =None ,next_sample =x_next )
