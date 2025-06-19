# SO101
python lerobot/find_port.py

## calibration

Commande pour calibrer robot (erreur dans la doc à cause des commentaires)
sudo chmod 666 /dev/ttyACM1
sudo chmod 666 /dev/ttyACM0


```python -m lerobot.calibrate     --robot.type=so101_follower     --robot.port=/dev/ttyACM0 --robot.id=follower_arm```

pour le leader: 

```python -m lerobot.calibrate     --teleop.type=so101_leader     --teleop.port=/dev/ttyACM1     --teleop.id=leader_arm```


pour le robot attention à :

- ne pas inverser les alim (robot correspondant indiqué dessus)

- en cas de besoin de connecter un moteur directement à la carte (pour reconfigurer son ID par exemple): attention aux cables qui ont fragiles. Pour les retirer on doit quelque fois tirer directmeent sur les cables, dans ce cas il faut ensuite bien faire attention que les bouts du cables ne sont pas sortis du connecteur


## téléopération

pour lancer la téléopération:

```[bash]
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm
```

pour la calibration, bien suivre la vidéo (le mettre en position intermédiaire avant de lancer la calibration)

https://huggingface.co/docs/lerobot/so101


## camera

voir https://huggingface.co/docs/lerobot/main/en/cameras#setup-cameras 


```
python lerobot/find_cameras.py opencv
```

test cam : 

```
cheese -d /dev/video2
```

pytest cam : 
```
python tests/cameras/test_my_cam.py
```

## record video

```
python -m lerobot.record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.num_episodes=2 \
    --dataset.single_task="Grab the black cube" \
    --dataset.push_to_hub=False
```


## replay video

```
python -m lerobot.replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.episode=0
```

## train


python lerobot/scripts/train.py   --dataset.repo_id=${HF_USER}/record-maceo-good   --policy.type=act   --output_dir=outputs/train/act_so101_test   --job_name=act_so101_test   --policy.device=cuda   --wandb.enable=False