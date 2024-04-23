# cs5342-final-project

Use `environment.yml` to create a conda environment to be able to run this repository.

## Dataset preparation

To prepare the dataset, download the animal kingdom dataset from this repo: https://sutdcv.github.io/Animal-Kingdom/

You'll have to extract the videos in this directory such that you have a new directory named `video`, and it contains the list of videos like so:
```
./video/AACXZTV.mp4
./video/AAAUILHH.mp4
...
```

The labels are already pre-generated, but if you want to make some modifications, please refer to `./tools/prepare_dataset.py` to see how we generated all the required files.

## Running

Running only requires running the following:
```bash
python main.py --config <CONFIG_FILE>
```

### Training

To train, use the config file `etc/config/train_107.yml`.

```bash
python main.py --config etc/config/train_107.yml
```

To do validations, use any config file in the following list:

```bash
python main.py --config etc/config/test_7_trained.yml
python main.py --config etc/config/test_33_trained.yml
python main.py --config etc/config/test_107_trained.yml
python main.py --config etc/config/test_958_trained.yml
python main.py --config etc/config/test_7_zeroshot.yml
python main.py --config etc/config/test_33_zeroshot.yml
python main.py --config etc/config/test_107_zeroshot.yml
python main.py --config etc/config/test_958_zeroshot.yml
```