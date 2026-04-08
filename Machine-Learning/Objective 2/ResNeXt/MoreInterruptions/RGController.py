import wandb
import json
import os

os.system('clear')
wandb.login()
sweep_config = {
    "name": "RGNet-Sweep",
    "metric":{
        "name":"Average Validation Loss",
        "goal":"minimize"
    },
    "controller":{
        "type":"local"
    },
    "method":"random",
    "parameters":{
        "optimizer":{
            "values":["adam", "radam", "nadam"]
        },
        "batch_size":{
            "distribution": "int_uniform",
            "min": 5000,
            "max": 10000
        },
        "fc_layer_size":{
            "distribution": "int_uniform",
            "min": 100,
            "max": 300
        },
        "layers":{
            "distribution": "int_uniform",
            "min": 1,
            "max": 3
        },
        "nodes":{
            "distribution":"int_uniform",
            "min":1,
            "max":15
        },
        "epochs":{
            "distribution":"int_uniform",
            "min":100,
            "max": 500
        },
        "nodal_layers":{
            "distribution":"int_uniform",
            "min":1,
            "max":5
        },
        "drop":{
            "distribution":"uniform",
            "min":0,
            "max":0.15
        },
        "lr":{
            "distribution":"uniform",
            "min":0,
            "max":0.0002
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
