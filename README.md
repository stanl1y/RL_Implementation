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
### Run neighborhood IL without BC loss
```
add
--no_bc
```
### Run neighborhood IL without hard negative sampling
```
add
--no_hard_negative_sampling
```


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
<!-- ## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- LICENSE -->
<!-- ## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- CONTACT -->
<!-- ## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- ACKNOWLEDGMENTS -->
<!-- ## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
