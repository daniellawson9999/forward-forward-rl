# Cartpole

## Simple REINFORCE Cartpole

python reinforce.py --h=16 --gamma=.99 --layers=3 --lr=1e-2 --break_at_threshold
python reinforce.py --h=64 --gamma=.99 --layers=3 --lr=1e-3 --break_at_threshold

## ff
bc working:
python forward_learning_bc.py --gamma=1 --layers=3 --inner_updates=10 --device=cuda --threshold=2 --batch_size=64 --data=data-t_24567-i_18979.pkl

testing:
python forward_learning.py --gamma=1 --layers=3  --inner_updates=10 --device=cpu --threshold=2

or forward-forward-upside-down.py