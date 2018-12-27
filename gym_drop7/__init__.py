import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Drop7-v0',
    entry_point='gym_drop7.envs:Drop7Env',
)

register(
    id='Drop7-v1',
    entry_point='gym_drop7.envs:Drop7Env',
    kwargs={'mode' : "sequence"}
)