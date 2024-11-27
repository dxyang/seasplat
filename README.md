# SeaSplat: Representing Underwater Scenes with 3D Gaussian Splatting and a Physically Grounded Image Formation Model
Daniel Yang, John J. Leonard, Yogesh Girdhar

## Abstract
We introduce SeaSplat, a method to enable real-time rendering of underwater scenes leveraging recent advances in 3D radiance fields. Underwater scenes are challenging visual environments, as rendering through a medium such as water introduces both range and color dependent effects on image capture. We constrain 3D Gaussian Splatting (3DGS), a recent advance in radiance fields enabling rapid training and real-time rendering of full 3D scenes, with a physically grounded underwater image formation model. Applying SeaSplat to the real-world scenes from SeaThru-NeRF dataset, a scene collected by an underwater vehicle in the US Virgin Islands, and simulation-degraded real-world scenes, not only do we see increased quantitative performance on rendering novel viewpoints from the scene with the medium present, but are also able to recover the underlying true color of the scene and restore renders to be without the presence of the intervening medium. We show that the underwater image formation helps learn scene structure, with better depth maps, as well as show that our improvements maintain the significant computational improvements afforded by leveraging a 3D Gaussian representation.

## Setup

```
# clone the repo
git clone git@github.com:dxyang/seasplat.git
git submodule init
git submodule update

# create a virtual environment
pyenv virtualenv 3.10 seasplat-py310
pip install --upgrade pip

# install pytorch (system dependent)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# standard pip dependencies
pip install -r requirements.txt

# install submodule dependencies
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

Alternatively, the `Dockerfile` specifies a Docker container

```
docker build --tag seasplat --build-arg USER_ID=$(id -u) -f Dockerfile .

docker run \
-u $(id -u) \
--gpus '"device=0"' \
-v ~/.cache/:/home/user/.cache/ \
-v ~/localdata:/home/user/localdata \
-v ~/localdata/code:/home/user/code \
--rm -it --shm-size=64gb \
-p 7010:7010 seasplat:latest
```

## Data

* [SaltPond dataset](https://drive.google.com/file/d/1gItZkfEFmXZzIRh5b6wXeWD6GappX-QN/view?usp=sharing)
* [SeaThru-NeRF dataset](https://drive.google.com/uc?export=download&id=1RzojBFvBWjUUhuJb95xJPSNP3nJwZWaT) from [https://sea-thru-nerf.github.io/](https://sea-thru-nerf.github.io/)

## Train a model
The codebase is built upon the original INRIA Gaussian Splatting [implementation](https://github.com/graphdeco-inria/gaussian-splatting). `DATASET_PATH` is expected to contain a directory of `images` (undistorted images) and `sparse/0` (typical COLMAP outputs). Additinoal arguments can be modified as defined in `arguments/__init__.py`.

```
python train.py \
-s DATASET_PATH \
--exp EXPERIMENT_NAME \
--do_seathru --seathru_from_iter 10000
```

## Bibtex
```
@article{yang2024seasplat,
  author    = {Yang, Daniel and Leonard, John J. and Girdhar, Yogesh},
  title     = {SeaSplat: Representing Underwater Scenes with 3D Gaussian Splatting and a Physically Grounded Image Formation Model},
  journal   = {arxiv},
  year      = {2024},
}
```