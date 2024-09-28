from typing import List, Union, Optional

from mmengine.config import ConfigDict
from mmengine.structures import InstanceData


# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]

InstanceList = List[InstanceData]
OptInstanceList = Optional[InstanceList]
