<!-- PROJECT SHIELDS -->
<!-- REF: https://github.com/othneildrew/Best-README-Template -->
[![Python][python-shield]][python-url]
<!-- TODO: gradio shield and url -->

<!-- GITHUB SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache-2.0 License][license-shield]][license-url]
<!-- TODO: github tag version shield and url https://shields.io/badges/git-hub-tag -->

<!-- SOCIAL SHIELDS -->
[![LinkedIn][linkedin-shield]][linkedin-url]
<!-- TODO: x.com shield and url https://shields.io/badges/x-formerly-twitter-url -->
<!-- TODO: instagram shield and url ? -->
<!-- TODO: discord server shield and url https://shields.io/badges/discord -->

<!-- TODO: add Zenerator banner -->

# Zenerator
Zenerator is a collection of ComfyUI workflows by Nilor Studio that we commonly use to produce art, conveniently packaged up in an easy-to-use Gradio application. The application was made to be extensible, so that others can produce Gradio UIs for [their own custom workflows via JSON](#making-custom-workflow-tabs).  

<!-- TODO: brag about how Zenerator has already been used to make production-quality work -->

## Dependencies
- 61 GB of available storage
- [Python 3.12.7](https://www.python.org/downloads/release/python-3127/)
- [CUDA 12.11](https://developer.nvidia.com/cuda-12-1-1-download-archive)
- SSH connection to the [Nilor Corp organization on HuggingFace](https://huggingface.co/nilor-corp)
- Git and LFS


## Installation
> [!WARNING]
> These setup instructions have only been tested on Windows using Powershell with Administrative privileges.

### Set up the Python Environment
* In a directory of your choosing, enter the following into your terminal:
    ```console
    mkdir nilor-corp
    cd nilor-corp
    python -m venv venv
    .\venv\Scripts\activate
    ```

### Install comfy-cli
* In the `.\nilor-corp\` directory:
    ```console
    python.exe -m pip install --upgrade pip
    pip install comfy-cli
    pip install ultralytics     # should prevent an error when installing ComfyUI-Impact-Pack later
    ```

### Install ComfyUI
1. In the `.\nilor-corp\` directory:
    ```console
    comfy --workspace=ComfyUI install
    ```
2. Running the above command will cause the terminal to prompt the user for input in the terminal. Select either "nvidia", "amd", or "intel_arc" for the first question (depending on which GPU you have), and type "y" to agree to the second question about which repository to clone from.
    
> [!WARNING]
> Zenerator has only been tested on Nvidia GPUs.

3. 
    ```console
    comfy set-default ComfyUI
    ```

### Install Models
* In the `.\nilor-corp\` directory:
    ```console
    cd ComfyUI
    rm -rf models   #if unix
    rmdir models    #if win. Answer yes if prompted
    git clone git@hf.co:nilor-corp/zenerator-models models
    cd ..           # should put you in .\nilor-corp\ directory
    ```

### Clone Zenerator
* In the `.\nilor-corp\` directory:
    ```console
    git clone https://github.com/nilor-corp/zenerator.git
    ```

### Set up Zenerator Env Variables
> [!NOTE]
> This step is only necessary if you intend to use the [nilor.cool](https://nilor.cool) platform to generate images and use Collections of generated images as input to Zenerator. This is a highly recommended and very convenient feature, but it requires a personal Nilor API key which can currently only be attained by request from the Nilor Corp development team. In the near future it will be possible to attain this API key directly through the [nilor.cool](https://nilor.cool) platform.

* In the `.\nilor-corp\zenerator\` directory, create a `.env` file and add the following to it:
    ``` 
    NILOR_API_KEY=<API KEY>
    NILOR_API_URI=https://api.nilor.cool/api
    ```

### Provision ComfyUI
1. In the `.\nilor-corp\` directory, still within the `venv` virtual environment:
    ```console
    cd zenerator
    cp .\ComfyUI-Manager-Snapshots\zenerator-snapshot.json ..\ComfyUI\custom_nodes\ComfyUI-Manager\snapshots\
    cd ..           # should put you in .\nilor-corp\ directory
    comfy launch
    ```
2. Once launched, navigate to ComfyUI in your browser: http://127.0.0.1:8188
3. In the top right corner of the screen click the `🧩 Manager` button.
4. A new window will appear. In the bottom left corner under the `Experimental` section, click `Snapshot Manager`.
5. Click `Restore` on the snapshot, then press the `Restart` button that appears in order to restart ComfyUI. This will download a lot packages which you should see in terminal.

> [!TIP]
> You may see a `ModuleNotFoundError` about `"tensorrt_bindings"` during ComfyUI startup. In our tests, this can be safely ignored because it is resolved by the next step.

### Install Zenerator dependencies
1. Once ComfyUI has fished installing packages, quit ComfyUI by pressing `Ctrl + C` (Win) or `Cmd + C` (Unix) in the terminal.

2. In the `.\nilor-corp\` directory, enter the following:
    ```console
    .\venv\Scripts\activate
    cd zenerator
    python -m pip install --no-cache-dir -r requirements.txt
    ```
3. Then run Zenerator to install the last remaining TensorRT dependencies by entering the following:
    ```console
    gradio app.py
    ```

Congratulations, you are finished installing Zenerator!

### Directory Structure
* If you have installed Zenerator correctly, your directory structure should look like this:
    ```
    nilor-corp
    ├── ComfyUI
    ├── venv  
    └── zenerator
    ```

### Test your Installation
* Following the [Usage](#usage) instructions below, with both ComfyUI and Zenerator initialized please run the "Test" workflow tab to ensure that your installation is functioning properly.

> [!NOTE]
> The first time you run a workflow on a machine, expect it to take longer to begin generating because some custom nodes will need to download models. Subsequent runs of the same workflow will take less time to start generating.


## Usage
You will need to run ComfyUI and Zenerator in seperate instances of Powershell.

### Run ComfyUI
* In the first instance of Powershell, from the `.\nilor-corp\` directory:
    ```console
    .\venv\scripts\activate
    comfy launch
    ```

> [!TIP]
> If you run into a "import torch" error when trying to launch ComfyUI for the first time, try [the potential fix here](https://github.com/Comfy-Org/comfy-cli/issues/150).

### Run Zenerator
* In the second instance of Powershell, from the `.\nilor-corp\` directory:
    ```console
    .\venv\scripts\activate
    cd zenerator
    gradio ./app.py
    ```

> [!NOTE]
> The first time you launch Zenerator, expect the startup to be delayed because it needs to build a TensorRT engine in order for the "Upscale Video (TensorRT)" workflow tab to work. This shouldn't take more than a few minutes and should only occur once.

> [!TIP]
> Generated outputs can be found in: `.\nilor-corp\ComfyUI\output\Zenerator\` directory.


## Advanced Usage

### Making a Custom Workflow Tab
> [!IMPORTANT]
> This section is under construction and we will add more detailed instructions soon.

In order to make new custom workflow tabs in the Gradio app, follow these instructions:

1. Within ComfyUI, click `Workflow` > `Export (API)` to create a new workflow JSON file and place it in the [workflows](./workflows/) folder. 

2. Then, edit the [workflow_definitions.json](workflow_definitions.json) file to add another definition for your new workflow. Be sure to edit the "filename" field to the filename of your newly-exported workflow JSON file, and follow the conventions of the existing pre-defined workflows in [workflow_definitions.json](workflow_definitions.json) to write a new workflow definition. 

3. You should populate your new workflow definition with the input components required to control the workflow, as well as the worklow node IDs that those input components should point to. The workflow node IDs can be found within your newly-exported worklow JSON file.



<!-- MARKDOWN LINKS & IMAGES -->
<!-- REF: https://github.com/othneildrew/Best-README-Template -->
[contributors-shield]: https://img.shields.io/github/contributors/nilor-corp/zenerator.svg?style=for-the-badge
[contributors-url]: https://github.com/nilor-corp/zenerator/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/nilor-corp/zenerator.svg?style=for-the-badge
[forks-url]: https://github.com/nilor-corp/zenerator/network/members
[stars-shield]: https://img.shields.io/github/stars/nilor-corp/zenerator.svg?style=for-the-badge
[stars-url]: https://github.com/nilor-corp/zenerator/stargazers
[issues-shield]: https://img.shields.io/github/issues/nilor-corp/zenerator.svg?style=for-the-badge
[issues-url]: https://github.com/nilor-corp/zenerator/issues
[license-shield]: https://img.shields.io/github/license/nilor-corp/zenerator.svg?style=for-the-badge
[license-url]: https://github.com/nilor-corp/zenerator/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/company/nilor-corp/
<!-- TODO: github tag version shield and url https://shields.io/badges/git-hub-tag -->
<!-- TODO: x.com shield and url https://shields.io/badges/x-formerly-twitter-url -->
<!-- TODO: discord server shield and url https://shields.io/badges/discord -->
[python-shield]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/
<!-- TODO: gradio shield and url -->