import wandb
import json
import os

os.system('clear')
wandb.login()
sweep_config = {
    "name": "Trilateration_Swarm_Optim",
    "metric":{
        "name":"Reward",
        "goal":"maximize"
    },
    "controller":{
        "type":"local"
    },
    "method":"bayes",
    "parameters":{
        "num_episodes":{
            "distribution": "int_uniform",
            "min": 20,
            "max": 500
        },
        "num_anchors":{
            "distribution": "int_uniform",
            "min": 3,
            "max": 10
        },
        "hidden_size":{
            "distribution": "int_uniform",
            "min": 50,
            "max": 1000
        },
        "num_layers":{
            "distribution": "int_uniform",
            "min": 1,
            "max": 5
        },
        "learning_rate":{
            "distribution": "uniform",
            "min": 0,
            "max": 1e-2
        },
        "entropy":{
            "distribution": "uniform",
            "min": 0,
            "max": 1
        }

    }
}

sweep_id = wandb.sweep(sweep_config, project="Swarm_Optim")
data = {
    "ID": sweep_id
}

with open('sweep_data.json', 'w') as f:
    json.dump(data, f, indent=4)

sweep = wandb.controller(sweep_id)
sweep.run()
