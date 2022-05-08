from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .object_pose import ObjectPoseTrainer
from .knowledge_distillation import KDTrainer

train_factory = {
    'object_pose': ObjectPoseTrainer,
    'kd': KDTrainer
}
