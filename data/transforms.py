import torch 
import torchvision .transforms as T 
import torchvision .transforms .functional as F 
from typing import List ,Union ,Tuple 


def get_interpolation_mode (mode_str :str )->T .InterpolationMode :
    modes ={
    "bicubic":T .InterpolationMode .BICUBIC ,
    "bilinear":T .InterpolationMode .BILINEAR ,
    "nearest":T .InterpolationMode .NEAREST ,
    }
    return modes .get (mode_str .lower (),T .InterpolationMode .BICUBIC )


def create_gaussian_kernel (kernel_size :int ,sigma :float )->torch .Tensor :

    if kernel_size %2 ==0 :
        kernel_size +=1 
    coords =torch .arange (kernel_size ).float ()-(kernel_size -1 )/2 
    x =coords .repeat (kernel_size ,1 )
    y =x .t ()
    gaussian =torch .exp (-(x .pow (2 )+y .pow (2 ))/(2 *sigma **2 ))
    return gaussian /gaussian .sum ()


def get_transforms (
size :int ,
train :bool =True ,
normalize :bool =True ,
mean :List [float ]=[0.5 ,0.5 ,0.5 ],
std :List [float ]=[0.5 ,0.5 ,0.5 ],
custom_transform :T .Compose =None ,
)->T .Compose :

    transform_list =[T .Resize ((size ,size ),antialias =True )]

    if custom_transform is not None :
        if hasattr (custom_transform ,"transforms"):
            custom_transforms =list (custom_transform .transforms )
        else :
            custom_transforms =[custom_transform ]
        transform_list .extend (custom_transforms )

    else :
        if train :
            transform_list .extend (
            [
            T .RandomHorizontalFlip (p =0.5 ),
            T .ColorJitter (brightness =0.1 ,contrast =0.1 ,saturation =0.1 ),
            ]
            )
        transform_list .append (T .ToTensor ())
        if normalize :
            transform_list .append (T .Normalize (mean =mean ,std =std ))

    return T .Compose (transform_list )


class DegradationTransform (torch .nn .Module ):
    def __init__ (
    self ,
    scale_factor :int =4 ,
    hr_size :int =256 ,
    kernel_size :int =7 ,
    sigma :float =1.5 ,
    noise_std :float =0.01 ,
    interpolation_mode :str ="bicubic",
    use_blur :bool =True ,
    use_downsample :bool =True ,
    ):

        super ().__init__ ()
        self .scale_factor =scale_factor 
        self .hr_size =hr_size 
        self .lr_size =hr_size //scale_factor 
        self .noise_std =noise_std 
        self .interpolation =get_interpolation_mode (interpolation_mode )
        self .use_downsample =use_downsample 


        self .use_blur =use_blur and kernel_size >1 and sigma >0 
        if self .use_blur :
            self .register_buffer ("kernel",create_gaussian_kernel (kernel_size ,sigma ))

    def forward (self ,img :Union [torch .Tensor ,any ])->torch .Tensor :
        if not isinstance (img ,torch .Tensor ):
            img =F .to_tensor (img )


        if self .use_blur :
            C =img .shape [0 ]
            kernel_C =self .kernel .expand (C ,1 ,self .kernel .shape [0 ],self .kernel .shape [1 ])
            padding =self .kernel .shape [0 ]//2 
            x_padded =torch .nn .functional .pad (
            img .unsqueeze (0 ),(padding ,padding ,padding ,padding ),mode ="replicate"
            )
            img =torch .nn .functional .conv2d (x_padded ,kernel_C ,groups =C ).squeeze (0 )


        if self .use_downsample :
            img =F .resize (img ,(self .lr_size ,self .lr_size ),interpolation =self .interpolation ,antialias =True )


        if self .noise_std >0 :
            img =img +torch .randn_like (img )*self .noise_std 
        return img 


class UpscaleTransform (torch .nn .Module ):
    def __init__ (
    self ,
    target_size :int =256 ,
    interpolation_mode :str ="bicubic",
    noise_std :float =0.0 ,
    ):
        super ().__init__ ()
        self .target_size =target_size 
        self .mode =interpolation_mode .lower ()
        self .align_corners =self .mode !="nearest"
        self .noise_std =noise_std 

    def forward (self ,img :torch .Tensor )->torch .Tensor :

        img =torch .nn .functional .interpolate (
        img .unsqueeze (0 ),
        size =(self .target_size ,self .target_size ),
        mode =self .mode ,
        align_corners =self .align_corners if self .align_corners else None ,
        ).squeeze (0 )

        if hasattr (self ,"noise_std")and self .noise_std >0 :
            img =img +torch .randn_like (img )*self .noise_std 
        return img 


class Inpainter (torch .nn .Module ):


    def __init__ (self ,mask_ratio :float =0.9 ,mask_type :str ="random"):

        super ().__init__ ()
        if not 0.0 <mask_ratio <1.0 :
            raise ValueError ("mask_ratio doit être compris entre 0 et 1.")
        self .mask_ratio =mask_ratio 
        self .mask_type =mask_type 

    def forward (self ,img :torch .Tensor )->Tuple [torch .Tensor ,torch .Tensor ]:

        if self .mask_type =="random":


            mask =(torch .rand_like (img [0 :1 ])>self .mask_ratio ).float ().to (img .device )
        else :
            raise NotImplementedError (f"Le type de masque '{self.mask_type}' n'est pas supporté.")


        img_masked =img *mask 

        return img_masked ,mask 
