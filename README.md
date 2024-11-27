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
In the `.\nilor-corp\` directory:
```
python.exe -m pip install --upgrade pip
pip install comfy-cli
```

### Install ComfyUI
In the `.\nilor-corp\` directory:
```
comfy --workspace=ComfyUI install
```
**Note:** Before proceeding, select either "nvidia", "amd", or "intel_arc" for the first question (depending on which GPU you have), and type "y" to agree to the second question about which repository to clone from.

**Disclaimer:** Zenerator has only been tested with Nvidia GPUs.
```
comfy set-default ComfyUI
```

### Install Models
In the `.\nilor-corp\` directory:
```
cd ComfyUI
rm -rf models   #if unix
rmdir models    #if win. Answer yes if prompted
git clone git@hf.co:nilor-corp/zenerator-models models
cd ..           # should put you in .\nilor-corp\ directory
```

### Clone Zenerator
In the `.\nilor-corp\` directory:
```
git clone git@hf.co:spaces/nilor-corp/zenerator
```

### Env Variables
In the `.\nilor-corp\zenerator\` directory, create a `.env` file and add the following to it:
``` 
NILOR_API_KEY=<API KEY>
NILOR_API_URI=https://api.nilor.cool/api
```

### Provision ComfyUI
In the `.\nilor-corp\` directory, still within the `venv` virtual environment:
```
cd zenerator
cp .\ComfyUI-Manager-Snapshots\2024-11-27_19-29-44_snapshot.json ..\ComfyUI\custom_nodes\ComfyUI-Manager\snapshots\
cd ..           # should put you in .\nilor-corp\ directory
comfy launch
```
Once launched, navigate to ComfyUI in your browser: http://127.0.0.1:8188

In the top right corner of the screen click the `🧩 Manager` button.

A new window will appear. In the bottom left corner under the `Experimental` section, click `Snapshot Manager`.

Click `Restore` on the snapshot, then press the `Restart` button that appears in order to restart ComfyUI. This will download a lot packages which you should see in terminal.

### Install Zenerator dependencies
Once ComfyUI has fished installing packages, quit ComfyUI by pressing `Ctrl + C` (Win) or `Cmd + C` (Unix) in the terminal.

In the `.\nilor-corp\` directory, enter the following:
```
.\venv\Scripts\activate
cd zenerator
python -m pip install --no-cache-dir -r requirements.txt
```
Then run Zenerator to install the last remaining TensorRT dependencies by entering the following:
```
gradio app.py
```

You are finished installing Zenerator!

### Directory Structure
If you have installed Zenerator correctly, your directory structure should look like this:
```
nilor-corp
├── ComfyUI
├── venv  
└── zenerator
```

## Usage
You will need to run ComfyUI and Zenerator in seperate instances of Powershell.

### Run ComfyUI
In the first instance of Powershell, from the `.\nilor-corp\` directory:
```
.\venv\scripts\activate
comfy launch
```

**Note:** If you run into "import torch" error when trying to launch comfy for the first time, [see potential fix here](https://github.com/Comfy-Org/comfy-cli/issues/150) 

### Run Zenerator
In the second instance of Powershell, from the `.\nilor-corp\` directory:
```
.\venv\scripts\activate
cd zenerator
gradio ./app.py
```

**Note:** The first time you launch Zenerator, expect the startup to be delayed because it needs to build a TensorRT engine in order for the "Upscale Video (TensorRT)" workflow tab to work. This shouldn't take more than a few minutes.

### Output
Generated output can be found in: `.\nilor-corp\ComfyUI\output\Zenerator\` directory.
