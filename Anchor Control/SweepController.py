import wandb
import json
import os

os.system('clear')
wandb.login()
sweep_config = {
    "name": "Reinforcement-Learning-Sweep",
    "metric":{
        "name":"Reward",
        "goal":"maximize"
    },
    "controller":{
        "type":"local"
    },
    "method":"bayes",
    "parameters":{
        "optimizer":{
            "values":["adam", "radam", "nadam","RMSprop"]
        },
        "fc_layer_size":{
            "distribution": "int_uniform",
            "min": 1000,
            "max": 3500
        },
        "layers":{
            "distribution": "int_uniform",
            "min": 1,
            "max": 5
        },
        "episodes":{
            "distribution":"int_uniform",
            "min":400,
            "max": 3000
        },
        "lr":{
            "distribution":"uniform",
            "min":0,
            "max":0.0001
        },
        "gamma":{
            "distribution":"uniform",
            "min":0,
            "max":0.99999
        }

    }
}

sweep_id = wandb.sweep(sweep_config, project="Cooperative Localization NN")
data = {
    "ID": sweep_id
}

with open('sweep_data.json', 'w') as f:
    json.dump(data, f, indent=4)

sweep = wandb.controller(sweep_id)
sweep.run()
