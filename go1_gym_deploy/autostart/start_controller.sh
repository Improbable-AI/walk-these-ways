#!/bin/bash
sudo docker stop foxy_controller || true
sudo docker rm foxy_controller || true
cd ~/go1_gym/go1_gym_deploy/docker/
sudo make autostart