This repository contains the code for our paper: Training with One2MultiSeq: CopyBART for Social Media Keyphrase Generation.
## Dataset

The datasets can be downloaded from [here](https://drive.google.com/file/d/12f2HOl6uOvsnCfiofuoB19vxWEEiGy00/view?usp=share_link)

For more details about the Twitter dataset, please reference [here](https://github.com/yuewang-cuhk/TAKG) or contact us at `gaochunyang@mail.hfut.edu.cn`

### Prepocessing
To preprocess the source data, run:
`python One2MultiSeq_dataprocess.py`

### Training
To preprocess the source data, run:
`python train_One2MultiSeq.py`
After the training, you can change `model_name` in line 707 to the path of the trained model(for example, `model_name = 'models/temp_model/CMKP/CopyBART_One2MultiSeq_base_epochs-10_learning_rate-5e-05_batch_size-32_seed-100'`) and set `is_train = False` in `train_One2MultiSeq.py`.

**Note:** 

* Please download and unzip the datasets in the `./data` directory first. 
