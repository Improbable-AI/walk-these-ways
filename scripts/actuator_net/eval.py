from utils import train_actuator_network_and_plot_predictions
from glob import glob


log_dir_root = "../../logs/"
log_dir = "example_experiment/2022/11_01/16_01_50_0"

# Evaluates the existing actuator network by default
load_pretrained_model = True
actuator_network_path = "../../resources/actuator_nets/unitree_go1.pt"

log_dirs = glob(f"{log_dir_root}{log_dir}/", recursive=True)

if len(log_dirs) == 0: raise FileNotFoundError(f"No log files found in {log_dir_root}{log_dir}/")

for log_dir in log_dirs:
    try:
        train_actuator_network_and_plot_predictions(log_dir[:11], log_dir[11:], actuator_network_path=actuator_network_path, load_pretrained_model=load_pretrained_model)
    except FileNotFoundError:
        print(f"Couldn't find log.pkl in {log_dir}")
    except EOFError:
        print(f"Incomplete log.pkl in {log_dir}")