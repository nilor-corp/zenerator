---
title: WorkFlower
emoji: 📉
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 4.20.0
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# Installing nilor industrial

## Dependencies:
- python
- cuda version 12.11
- ssh connection to nilor corp HF organization: https://huggingface.co/nilor-corp
- git lfs


## Installation
### setup python environment
```
mkdir nilor-corp
cd nilor-corp
python -m venv venv
.\venv\Scripts\activate
```

### Install comfy-cli
git clone https://github.com/Comfy-Org/comfy-cli
cd comfy-cli
git checkout e7d5e8a4dab2e32289d32ecf6d01a7aa0705d2de
python -m pip install .
cd ..

### Install ComfyUI
```
comfy --workspace=. install --commit=eecd69b53a896343775bcb02a4f8349e7442ffd1
comfy set-default .
mkdir .\ComfyUI\output\WorkFlower
```

### Install Models
```py
cd ComfyUI
rm -rf models #if unix
rmdir models #if win. Answer yes if prompted
git clone -b version-0  git@hf.co:datasets/nilor-corp/models
```

### Install Workflower
```
cd ..
git clone -b version-0 git@hf.co:spaces/nilor-corp/nilor-industrial
cp .\nilor-industrial\ComfyUI-Manager-Snapshots\2024-04-30_18-41-53_snapshot.json .\ComfyUI\custom_nodes\ComfyUI-Manager\snapshots\
cd nilor-industrial
python -m pip install -r requirements.txt
```

### Env Variables
- (in nilor-industrial root, where you should be if you've been following the above commands) create a .env file and add:
``` 
NILOR_API_KEY=<API KEY>
NILOR_API_URI=https://api.nilor.cool/api
```

### Provision comfy UI

```py
cd .. # should be in nilor-corp dir root
comfy launch
```
- once launched, navigate to comfyUI in browser  http://127.0.0.1:8188
- in the bottom right corner of the secret on the floating modal click the "manager" button
- a new window will appear, in the bottom left corner, under the "expiremental", section on the bottom left click, "snapshot manager"
- Click "restore" on the snapshot and then press the  "restart" button that will appear to restart comfyUI. This will download a lot packages which you should see in terminal 

### Directory Structure
After finishing installation, your directory structure should look like this:
- nilor-corp
    - comfy-cli
    - ComfyUI
    - nilor-industrial
    - venv

## Run
You will need to run ComfyUI and nilor industrial in seperate shells

### Run ComfyUI
From nilor-corp root:
```
.\venv\scripts\activate
comfy launch
```

### Run nilor industrial 
From nilor-corp root:
```
.\venv\scripts\activate
cd nilor-industrial
gradio ./app.py
```

### Output
Generated output can be found in:
```
nilor-corp\ComfyUI\output\WorkFlower
```

