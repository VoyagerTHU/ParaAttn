## `Hunyuan`代码详解

### 结构综述

我们的`attention_type`可以大概分为两类，获取信息型和生成视频型，具体而言：
获取信息型重点不在于生成视频，而是预先计算好一定的信息，目前用到的信息分为两类，一个是不同head的`attention pattern(flags)`，一个是分块计算的`block_avg`，目前这两种需要预先计算；
生成视频型对于目前的`Hunyuan`来说主要有`decay`，`repeat mask`，`sink`这三类，在这基础上可能用到上述的`flags`和`block_avg`，用`flags`可以只对特定`pattern`操作，用`block_avg`可以通过指定`ref_function`来获取`block_bias`，从而保护小值

### 代码解析

#### `pattern`：获取`flags`

按照以上的综述理解代码，首先在获取信息型的`attention_type`中亦有不同，获取`flags`的`attention_type`名为`pattern`，所有与获取`pattern`相关的参数在字典`threshold_attn_args`中，主要是传入了掩码`mask`和阈值`threshold`，`maks`在当前应用中暂时放弃了原来`narrow`和`wide`的三级划分，改成两级，故现在只有`attention_masks_narrow`有作用，而`threshold`根据`window`和`token`的比值直接规定，定义为`config['narrow_width'] * 2 * config['pattern_threshold'] / ((num_frames - 1) // 4 + 1)`，其中`narrow_width`是`window`的半径（以frame为单位），通过这个参数在获取`mask`和获取`threshold`中的一致来保证正确性

目前固定的分`pattern`方法的`method`参数为`proportional`，`mask`内`value`占总体的比重小于`token`占总体比重的被划为全局型，对应取值为1，否则为0

注意上述式子中的`config['pattern_threshold']`是可以调节的超参，该值越高，阈值越高，被分为全局的pattern越少，结果的pattern中得1的应该更少

#### `get_block_avg`：获取`block_avg`

获取`block_avg`的代码，需要设定`save_one_timestep`参数，为`True`时只存储第一个`timestep`，存完一个`timestep`后raise一个error并退出，因为只有一步，所以这种情况很快；为`False`时存储所有的`timestep`，谨慎使用，所用时间和存储空间都很大

#### `sink`：生成视频

目前大部分逻辑被集成在了`sink`这一个`attention_type`中，可以选择衰减参数（alpha_xpos_xi, beta_xpos_xi），是否`repeat mask`（repeat_mask_in_sink，True则mask）以及`attention sink`（sink_width，单位为frame，0则无sink）大小和`sliding window`（window_width，单位为frame，默认33，训练长度，这里是直径）大小，前述括号内部名称即为`sink_args`的参数

此外还可以加入对之前获取信息的选择性使用，

对于`flags`，两个参数：`use_pattern_flags`，为True代表区分`pattern`，需要加载分`pattern`的`flags`；在此基础上，如果`use_one_time_flags`为True，那么代表直接加载训练长度上获取的`pattern`，**需要预先在num_frames=129的时候跑一遍`attention_type='pattern'`的代码**，否则就在线计算，但外推的时候基于的是外推后的`pattern`，可能有失偏颇

对于`block_avg`，和`flags`类似，两个参数：`use_decay_mask`，为True代表需要用到`block_avg`来计算`block_bias`保护小值；在此基础上，如果`use_one_timestep_block_avg`为True，则**对应于`get_block_avg`中`save_one_timestep`为True的情形**，同一block的所有timestep获取的是同一`block_avg`，如果`use_one_timestep_block_avg`是False，那么相应地需要之前跑过一次`save_one_timestep`为False的`get_block_avg`


## Usage

- 是否分为全局和局部 pattern
    - 生成时是否分治：use_pattern_flags=True
    - 若生成时需要分治，则又有判断 pattern 对不同方式：
        - 基于训练长度还是对外推长度判断 pattern：use_one_time_flags=True 是基于训练长度
        - 若基于训练长度，则需预处理。预处理时，`'attention_type': 'pattern'`，需要指定另外两个参数：
            - `pattern_threshold`：默认为1，代表阈值是标准比例的几倍
            - `narrow_width`：默认为1，计算mask，分pattern时的窗口半径，单位是frame

- 是否对小值进行保护
    - 生成时是否启动小值保护：use_decay_mask=True
    - 如需要，分为两种获取block_avg方式，从所有timestep均一样的block_avg获取：use_one_timestep_block_avg=True，否则从每个timestep都不一样的block_avg获取，两种均需预处理。预处理时，`'attention_type': 'get_block_avg'`，需要指定一个参数：
        - `save_one_timestep`：True时只存一个timestep，然后退出；False时所有timestep都存

- sink
    - 示例配置：
    ```python
        'attention_type': 'sink',
        'prompt': prompts[0],
        'num_frames': 393,

        # --- decay part ---
        'alpha': 1.00,
        'beta': 0.95,

        # --- sink part ---
        'sink_width': 4,

        # --- repeat mask part ---
        'repeat_mask_in_sink': True,

        # --- sliding window part ---
        'window_width': 33,  # 这里是直径

        # --- online flags part ---
        'use_pattern_flags': True,
        'use_one_time_flags': True,
        'narrow_width': 1,
        'pattern_threshold': 1,  

        # --- block avg part ---
        'use_one_timestep_block_avg': False,

        # --- block bias part ---
        'use_block_bias': True,
        'ref_function_name': 'uniform',

        # --- decay mask part ---
        'use_decay_mask': False,
        'decay_mask_threshold': 0.0000,
        
    ```

    - 衰减参数（方法目前固定为双指数插值）
        - `alpha` `beta`
    - 是否 attention sink
        - `'sink_width'`，单位为frame，0时退化为无sink
    - 是否 引入repeat mask
        - `'repeat_mask_in_sink'`
    - 衰减类型：是否有 window protection
        - `'window_width'`，单位为frame，0时退化为无window protection，衰减和普通情形相同
    - 分pattern - `use_pattern_flags`
        - 如果false，无需其他参数
        - 如果true，看`use_one_time_flags`
            - 如果true，无需其他参数
            - 如果false，指定`narrow_width`，`pattern_threshold`，语义见上文Usage第一部分
    - 与`block_avg`相关的 - `use_one_timestep_block_avg`
        - true则用one_timestep版本的block_avg，否则用不同timestep区分开的版本
        - 利用到`block_avg`的有下面两个方法：
            - block_bias：`use_block_bias=True`，使用block_bias，需要指定`ref_function_name`作为阈值函数
            - decay_mask：`use_decay_mask=True`使用，需要设定阈值`decay_mask_threshold`


> TODO: 看hunyuan的cfg