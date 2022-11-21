#!/bin/bash
# download docker image if it doesn't exist yet
wget --directory-prefix=../docker -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XkVpyYyYqQQ4FcgLIDUxg-GR1WI89-XC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XkVpyYyYqQQ4FcgLIDUxg-GR1WI89-XC" -O deployment_image.tar && rm -rf /tmp/cookies.txt

#rsync -av -e ssh --exclude=*.pt --exclude=*.mp4 $PWD/../../go1_gym_deploy $PWD/../../runs $PWD/../../setup.py pi@192.168.12.1:/home/pi/go1_gym
rsync -av -e ssh --exclude=*.pt --exclude=*.mp4 $PWD/../../go1_gym_deploy $PWD/../../runs $PWD/../setup.py unitree@192.168.123.15:/home/unitree/go1_gym
#scp -r $PWD/../../runs pi@192.168.12.1:/home/pi/go1_gym
#scp -r $PWD/../../setup.py pi@192.168.12.1:/home/pi/go1_gym
