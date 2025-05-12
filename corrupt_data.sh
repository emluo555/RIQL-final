python test_corruption.py \
    --env_name ant-medium-replay-v2 \
    --save_path ./ant-datasets/ant-medium-replay-v2-clean.hdf5

python test_corruption.py \
    --env_name ant-medium-replay-v2 \
    --save_path ./ant-datasets/ant-medium-replay-v2-corrupt-acts.hdf5 \
    --corrupt_acts

python test_corruption.py \
    --env_name ant-medium-replay-v2 \
    --save_path ./ant-datasets/ant-medium-replay-v2-corrupt-reward.hdf5 \
    --corrupt_reward

python test_corruption.py \
    --env_name ant-medium-replay-v2 \
    --save_path ./ant-datasets/ant-medium-replay-v2-corrupt-dyns.hdf5 \
    --corrupt_dynamics

python test_corruption.py \
    --env_name ant-medium-replay-v2 \
    --save_path ./ant-datasets/ant-medium-replay-v2-corrupt-obs.hdf5 \
    --corrupt_obs
