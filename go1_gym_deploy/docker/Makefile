default: build
build:
	docker build -t jetson-model-deployment .
clean-build:
	docker build -t jetson-model-deployment . --no-cache=true
run:
	docker stop foxy_controller || true
	docker rm foxy_controller || true
	docker run -it \
		--env="DISPLAY" \
		--env="QT_X11_NO_MITSHM=1" \
		--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		--env="XAUTHORITY=${XAUTH}" \
		--volume="${XAUTH}:${XAUTH}" \
		--volume="/home/unitree/go1_gym:/home/isaac/go1_gym" \
		--privileged \
		--runtime=nvidia \
		--net=host \
		--workdir="/home/isaac/go1_gym" \
		--name="foxy_controller" \
		jetson-model-deployment bash
autostart:
	docker stop foxy_controller || true
	docker rm foxy_controller || true
	docker run -d\
		--env="DISPLAY" \
		--env="QT_X11_NO_MITSHM=1" \
		--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		--env="XAUTHORITY=${XAUTH}" \
		--volume="${XAUTH}:${XAUTH}" \
		--volume="/home/unitree/go1_gym:/home/isaac/go1_gym" \
		--privileged \
		--runtime=nvidia \
		--net=host \
		--workdir="/home/isaac/go1_gym" \
		--name="foxy_controller" \
		jetson-model-deployment tail -f /dev/null
		docker start foxy_controller
		docker exec foxy_controller bash -c 'cd /home/isaac/go1_gym/ && python3 setup.py install && cd go1_gym_deploy/scripts && ls && python3 deploy_policy.py'
