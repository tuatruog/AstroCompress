# AstroCompress Benchmark

### About
This is a repository for benchmarking neural compression algorithms on the [AstroCompress Corpus](https://huggingface.co/AstroCompress).
These neural compression algorithms are adapted from the original publication. For more details, please refer to the main paper [here]().

### Local setup
To run the benchmark locally, clone the dataset of interest from the AstroCompress corpus to your local device. Ex:

```
git lfs install
git clone https://huggingface.co/datasets/AstroCompress/GBI-16-2D-Legacy
```

Create new environment and install required dependencies
```
conda create --name astrocompress python=3.8
pip install requirements.txt
```

### Benchmarking models
#### Dataset name
We define the dataset name alias for our experiments as follows:

| Experiment  | Alias    |
|-------------|----------|
| LCO         | lco      |
| Keck        | keck     |
| Hubble      | hst      |
| JWST-2D     | jwst     |
| SDSS-2D     | sdss     |
| JWST-2D-RES | jwst-res |
| SDSS-3Dλ    | sdss-3d  |
| SDSS-3DT    | sdss-3t  |

These alias are used as the name for the argument `--dataset` in our training and evaluating scripts.
Note that these dataset will need to be downloaded locally and update the `data_configs.py` to the 
root of your local the data directory before using the benchmark commands.

<br>

#### Integer Discrete Flows (IDF) [[Paper]](https://arxiv.org/pdf/1905.07376) [[Github]](https://github.com/jornpeters/integer_discrete_flows)
For training IDF, use `train_idf.py`. Example command:
```
python3 train_idf.py --dataset <alias> --out_dir <out_dir> --input_size 32,32 --random_crop --flip_horizontal 0.5 --batch_size 256 --lr_decay_epoch 250 --epochs 30000 --evaluate_interval_epochs 2000 --epoch_log_interval 300
```
For evaluating IDF, use `evaluate_idf.py`. Example command:
```
python3 evaluate_idf.py --dataset <alias> --out_dir <out_dir> --epoch <epoch_step>
```
For more details on command flags, run:
`python train_idf.py --help`
`python evaluate_idf.py --help`

<br>

#### L3C [[Paper]](https://arxiv.org/pdf/1811.12817) [[Github]](https://github.com/fab-jul/L3C-PyTorch)
For training l3c, use `train_l3c.py`. Example command:
```
python3 train_l3c.py --ms_config_p <config_path> --log_dir_root <out_dir> --log_train 350 --log_val 3500 --dataset <alias> --input_size 32,32 --split_bits --random_crop --flip_horizontal 0.5 --batch_size 256
```
For evaluating l3c, use `evaluate_l3c.py`. Example command:
```
python3 evaluate_l3c.py --ms_config_p <config_path_root> --log_dir <out_dir> --log_dates <log_date_id> --dataset <alias> --split_bits --model_input_size 32,32 --restore_itr <itr>
```
For more details on command flags, run:
`python train_l3c.py --help`
`python evaluate_l3c.py --help`

<br>

#### PixelCNN++ [[Paper]](https://arxiv.org/pdf/1701.05517) [[Github]](https://github.com/openai/pixel-cnn)
For training PixelCNN++, use `main_compression.py`. Example command:
```
cd ./pixelcnn
python main_compression.py --dataset <alias>
```
For evaluating PixelCNN++, use `evaluate_pixelcnn.py`. Example command:
```
cd ./pixelcnn
python evaluate_pixelcnn.py --dataset <alias> --model <model_path>
```
For more details on command flags, run:
`python main_compression.py --help`
`python evaluate_pixelcnn.py --help`