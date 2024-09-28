# Copyright (c) OpenMMLab. All rights reserved.
import mmengine
import mmcv
import mmrazor
import mmpretrain
import mmseg
import mmdet
import mmdet3d
import mmdeploy
from mmengine.utils import digit_version

from .version import __version__, version_info

mmengine_minimum_version = '0.6.0'
mmengine_maximum_version = '1.0.0'
mmengine_version = digit_version(mmengine.__version__)

mmcv_minimum_version = '2.0.0rc4'
mmcv_maximum_version = '3.0.0'
mmcv_version = digit_version(mmcv.__version__)

mmrazor_minimum_version = '1.0.0'
mmrazor_maximum_version = '2.0.0'
mmrazor_version = digit_version(mmrazor.__version__)

mmpretrain_minimum_version = '1.0.0rc5'
mmpretrain_maximum_version = '2.0.0'
mmpretrain_version = digit_version(mmpretrain.__version__)

mmdet_minimum_version = '3.0.0rc5'
mmdet_maximum_version = '4.0.0'
mmdet_version = digit_version(mmdet.__version__)

mmseg_minimum_version = '1.0.0rc5'
mmseg_maximum_version = '2.0.0'
mmseg_version = digit_version(mmseg.__version__)

mmdet3d_minimum_version = '1.1.0rc3'
mmdet3d_maximum_version = '2.0.0'
mmdet3d_version = digit_version(mmdet3d.__version__)

mmdeploy_minimum_version = '0.1.0'
mmdeploy_maximum_version = '2.0.0'
mmdeploy_version = digit_version(mmdeploy.__version__)

assert (mmengine_version >= digit_version(mmengine_minimum_version)
        and mmengine_version < digit_version(mmengine_maximum_version)), \
    f'MMEngine=={mmengine.__version__} is used but incompatible. ' \
    f'Please install mmengine>={mmengine_minimum_version}, ' \
    f'<{mmengine_maximum_version}.'

assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version < digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <{mmcv_maximum_version}.'

assert (mmrazor_version >= digit_version(mmrazor_minimum_version)
        and mmrazor_version < digit_version(mmrazor_maximum_version)), \
    f'MMRazor=={mmrazor.__version__} is used but incompatible. ' \
    f'Please install mmrazor>={mmrazor_minimum_version}, <{mmrazor_maximum_version}.'

assert (mmpretrain_version >= digit_version(mmpretrain_minimum_version)
        and mmpretrain_version < digit_version(mmpretrain_maximum_version)), \
    f'MMPRETRAIN=={mmpretrain.__version__} is used but incompatible. ' \
    f'Please install mmpretrain>={mmpretrain_minimum_version}, <{mmpretrain_maximum_version}.'

assert (mmdet_version >= digit_version(mmdet_minimum_version)
        and mmdet_version < digit_version(mmdet_maximum_version)), \
    f'MMDET=={mmdet.__version__} is used but incompatible. ' \
    f'Please install mmdet>={mmdet_minimum_version}, ' \
    f'<{mmdet_maximum_version}.'

assert (mmseg_version >= digit_version(mmseg_minimum_version)
        and mmseg_version < digit_version(mmseg_maximum_version)), \
    f'MMSEG=={mmseg.__version__} is used but incompatible. ' \
    f'Please install mmseg>={mmseg_minimum_version}, <{mmseg_maximum_version}.'

assert (mmdet3d_version >= digit_version(mmdet3d_minimum_version)
        and mmdet3d_version < digit_version(mmdet3d_maximum_version)), \
    f'MMDET3D=={mmdet3d.__version__} is used but incompatible. ' \
    f'Please install mmdet3d>={mmdet3d_minimum_version}, ' \
    f'<{mmdet3d_maximum_version}.'

assert (mmdeploy_version >= digit_version(mmdeploy_minimum_version)
        and mmdeploy_version < digit_version(mmdeploy_maximum_version)), \
    f'MMDeploy=={mmdeploy.__version__} is used but incompatible. ' \
    f'Please install mmdeploy>={mmdeploy_minimum_version}, ' \
    f'<{mmdeploy_maximum_version}.'

__all__ = ['__version__', 'version_info', 'digit_version']
