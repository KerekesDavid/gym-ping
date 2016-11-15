from gym.envs.registration import register

register(
    id='PingPongImg-v0',
    entry_point='gym.envs.ping:PingEnvImg',
    timestep_limit=200,
)

# register(
#     id='PingPongPos-v0',
#     entry_point='gym.envs.ping:PingEnvPos',
#     timestep_limit=200,
# )