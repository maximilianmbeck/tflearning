from timm.models.vision_transformer import VisionTransformer

from ml_utilities.torch_models.base_model import BaseModel
from ml_utilities.torch_models import register_model


class ViTTimm(BaseModel):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 global_pool='token',
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 class_token=True,
                 no_embed_class=False,
                 pre_norm=False,
                 fc_norm=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
        """
        super().__init__()

        self.config = dict(img_size=img_size,
                           patch_size=patch_size,
                           in_chans=in_chans,
                           num_classes=num_classes,
                           global_pool=global_pool,
                           embed_dim=embed_dim,
                           depth=depth,
                           num_heads=num_heads,
                           mlp_ratio=mlp_ratio,
                           qkv_bias=qkv_bias,
                           class_token=class_token,
                           no_embed_class=no_embed_class,
                           pre_norm=pre_norm,
                           fc_norm=fc_norm,
                           drop_rate=drop_rate,
                           attn_drop_rate=attn_drop_rate,
                           drop_path_rate=drop_path_rate,
                           weight_init=weight_init)
        
        self.vit_model = VisionTransformer(**self.config)

    def reset_parameters(self):
        self.vit_model.init_weights(mode=self.config['weight_init'])

    def forward(self, x):
        return self.vit_model(x)
    
register_model(ViTTimm.__name__, ViTTimm)