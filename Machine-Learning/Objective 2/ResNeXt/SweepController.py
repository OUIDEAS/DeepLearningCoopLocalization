import wandb
import json
import os

os.system('clear')
wandb.login()
sweep_config = {
    "name": "Machine-Learning-Cooperative-Localization-Sweep",
    "metric":{
        "name":"Average Validation Loss",
        "goal":"minimize"
    },
    "controller":{
        "type":"local"
    },
    "method":"bayes",
    "parameters":{
        "optimizer":{
            "values":["adam", "radam", "nadam"]
        },
        "batch_size":{
            "distribution": "int_uniform",
            "min": 150,
            "max": 1000
        },
        "fc_layer_size":{
            "distribution": "int_uniform",
            "min": 750,
            "max": 2500
        },
        "layers":{
            "distribution": "int_uniform",
            "min": 1,
            "max": 3
        },
        "epochs":{
            "distribution":"int_uniform",
            "min":35,
            "max": 75
        },
        "residuals":{
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
            "max":0.0005
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
