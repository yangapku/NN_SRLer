# 程序运行说明

## 代码文件说明

+ `src/`文件夹下存放了模型源代码，由以下代码文件组成：
  + `src/data_utils.py`用于数据载入，读取词典文件、训练数据和预测数据
  + `src/model.py`定义了BiLSTM-CRF模型
  + `src/config.py`存储训练和预测时的各项超参数和配置
  + `src/nnsrler.py`主程序所在位置
  + `src/calc_f1.py`用于计算F值


## 执行训练过程
+ `models/`文件夹用于存放训练的模型，`log/`文件夹记录训练日志
+ 执行训练过程时，先修改`src/config.py`中的配置
  + 将`do_train`设为`True`，`do_predict`设为`False`
  + `batch_size`指定batch大小
  + `num_epoch`指定训练epoch数
  + `optimizer`指定优化器，可选`sgd`、`adam`、`adagrad`、`momentum`
  + `lrate`指定学习率
  + `save_path`指定模型存储路径
  + `log_dir`指定日志存储路径
  + `use_crf`指定是否使用CRF
+ 配置好`config.py`文件后，在`src`文件夹下输入`python nnsrler.py`即可，训练和每个epoch的验证loss会被输出至控制台
+ `models/171222_ver04/171222_ver04-9`是已经训练好的BiLSTM-CRF模型，可直接用于预测，在开发集得到0.658的F值

## 执行预测过程
+ 修改`src/config.py`中的配置
  + `do_train`设为`False`，将`do_predict`设为`True`
  + `load_path`指定模型所在路径
  + `output_path`指明预测文件输出路径
  + `testing_data_path`指明处理后的输入文件所在位置（开发集`../dat/devIn.txt`测试集`../dat/testIn.txt`）
  + `testing_sourcefile_path`指明原始输入文件所在位置（开发集`../dat/cpbdev.txt`测试集`../dat/cpbtest.txt`）
  + 其他超参数设置需要与载入的模型训练时相同
+ 配置好`config.py`文件后，在`src`文件夹下输入`python nnsrler.py`即可
+ `outputs/dev_171222_ver04_9.txt`和`outputs/test_171222_ver04_9.txt`分别是用`models/171222_ver04/171222_ver04-9`预测的开发集和测试集输出
