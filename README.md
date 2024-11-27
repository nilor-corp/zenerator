---
title: Zenerator 
emoji: ☯️
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 5.6.0
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# Zenerator

## Dependencies
- Python
- Cuda version 12.11
- SSH connection to Nilor Corp HuggingFace organization: https://huggingface.co/nilor-corp
- Git and LFS


## Installation
**Disclaimer:** These commands have only been tested using Powershell with Administrative privileges.

### Setup the Python environment
```
mkdir nilor-corp
cd nilor-corp
python -m venv venv
.\venv\Scripts\activate
```

### Install comfy-cli
In the `.\nilor-corp\` dir:
```
pip install comfy-cli
```

### Install ComfyUI
In the `.\nilor-corp\` dir:
```
comfy --workspace=ComfyUI install
comfy set-default ComfyUI
```

### Install Models
In the `.\nilor-corp\` dir:
```
cd ComfyUI
rm -rf models   #if unix
rmdir models    #if win. Answer yes if prompted
git clone git@hf.co:nilor-corp/zenerator-models models
cd ..   # should put you in .\nilor-corp\ dir
```

### Install Workflower
In the `.\nilor-corp\` dir:
```
git clone git@hf.co:spaces/nilor-corp/zenerator
cd zenerator
------------------
cp .\ComfyUI-Manager-Snapshots\2024-11-22_18-23-34_snapshot.json ..\ComfyUI\custom_nodes\ComfyUI-Manager\snapshots\
-------------------
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
python -m pip install -r requirements.txt
```

### Env Variables
In the `.\nilor-corp\zenerator\` dir, create a `.env` file and add the following to it:
``` 
NILOR_API_KEY=<API KEY>
NILOR_API_URI=https://api.nilor.cool/api
```

### Provision ComfyUI
In the `.\nilor-corp\zenerator\` dir:
```
cd ..   # should put you in .\nilor-corp\ dir
.\venv\Scripts\activate
comfy launch
```
- once launched, navigate to comfyUI in browser  http://127.0.0.1:8188
- in the top right corner of the screen click the "🧩 Manager" button
- a new window will appear, in the bottom left corner, under the "Experimental" section click "Snapshot Manager"
- click "Restore" on the snapshot and then press the "Restart" button that will appear to restart ComfyUI. This will download a lot packages which you should see in terminal 

### Directory Structure
After finishing installation, your directory structure should look like this:
- nilor-corp
    - ComfyUI
    - venv
    - zenerator

## Usage
You will need to run ComfyUI and Zenerator in seperate instances of Powershell.

### Run ComfyUI
In the first instance of Powershell, from the `.\nilor-corp\` dir:
```
.\venv\scripts\activate
comfy launch
```
**Note:** If you run into "import torch" error when trying to launch comfy for the first time, [see potential fix here](https://github.com/Comfy-Org/comfy-cli/issues/150) 

### Run Zenerator
In the second instance of Powershell, from the `.\nilor-corp\` dir:
```
.\venv\scripts\activate
cd zenerator
gradio ./app.py
```

**Note:** The first time you launch Zenerator, expect the startup to be delayed because it needs to build a TensorRT engine in order for the "Upscale Video (TensorRT)" workflow tab to work. This shouldn't take more than a few minutes.

### Output
Generated output can be found in: `.\nilor-corp\ComfyUI\output\Zenerator\` dir.
