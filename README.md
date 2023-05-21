<div id="top"></div>
<!-- from https://github.com/othneildrew/Best-README-Template -->


<!-- PROJECT LOGO -->
<br />
<div align="center">
<h1 align="center">TDIL (Imitation Learning via Transition Discriminator)</h1>
</div>


### Score of Expert Data
* Hopper-v3:4114
* Walker-v3:6123
* Ant-v3:6561
* HalfCheetah-v3:15251
* Humanoid-v3:5855


## Scripts
### Normal neighborhood IL
* Hopper-v3:
    ```
    python main.py --main_stage neighborhood_il --main_task neighborhood_sac --env Hopper-v3 --wrapper basic --episode 5000 --data_name sac/episode_num1
    ``` 
    * fix alpha
        ```
        add
        --no_update_alpha --log_alpha_init -4.6
        ```
* Walker-v3:
    ```
    python main.py --main_stage neighborhood_il --main_task neighborhood_sac --env Walker2d-v3 --wrapper basic --episode 5000 --data_name sac/episode_num1
    ```
    * fix alpha
        ```
        add
        --no_update_alpha --log_alpha_init -1.2
        ```
* Ant-v3:
    ```
    python main.py --main_stage neighborhood_il --main_task neighborhood_sac --env Ant-v3 --wrapper basic --episode 10000 --data_name sac/episode_num1 --terminate_when_unhealthy
    ```
    * fix alpha
        ```
        add
        --no_update_alpha --log_alpha_init -1.9
        ```
* HalfCheetah-v3:
    ```
   python main.py --main_stage neighborhood_il --main_task neighborhood_sac --env HalfCheetah-v3 --wrapper basic --episode 5000 --data_name sac/episode_num1
    ``` 
    * fix alpha
        ```
        add
        --no_update_alpha --log_alpha_init 0.4
        ```
* Humanoid-v3:
    ```
   python main.py --main_stage neighborhood_il --main_task neighborhood_sac --env Humanoid-v3 --wrapper basic --episode 25000 --data_name sac/episode_num1 --terminate_when_unhealthy
    ``` 
    * fix alpha
        ```
        add
        --no_update_alpha --log_alpha_init -0.6
        ```
### Run without BC loss
    ```
    add
    --no_bc
    ```
### Run without hard negative sampling
    ```
    add
    --no_hard_negative_sampling
    ```
