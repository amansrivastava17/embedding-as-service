class Flags:

    # Model
    model_config_path: str = None  # Model config path
    dropout: float = 0.1  # Dropout rate
    dropatt: float = 0.1  # Attention dropout rate.
    clamp_len: int = -1  # Clamp length
    summary_type: str = "last"  # Method used to summarize a sequence into a compact vector.
    use_summ_proj: bool = True  # Whether to use projection for summarizing sequences.
    use_bfloat16: bool = False  # Whether to use bfloat16.

    # Parameter initialization
    init: str = "normal"
    init_std: float = 0.2   # Initialization std when init is normal.
    init_range: float = 0.1  # Initialization std when init is uniform.

    # I/O paths
    overwrite_data: bool = False  # If False, will use cached data if available."
    init_checkpoint: str = None  # checkpoint path for initializing the model.
    output_dir: str = ""  # Output dir for TF records.
    spiece_model_file: str = ""  # Sentence Piece model path.
    model_dir: str = ""  # Directory for saving the finetuned model.
    data_dir: str = ""  # Directory for input data.

    # TPUs and machines
    use_tpu: bool = False  # whether to use TPU.
    num_hosts: int = 1  # How many TPU hosts.
    num_core_per_host: int = 8  # 8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context of GPU training,
    # it refers to the number of GPUs used.")

    tpu_job_name: str = None  # TPU worker job name.
    tpu: str = ""  # TPU name.
    tpu_zone: str = ""  # TPU zone.
    gcp_project: str = ""  # gcp project.
    master: str = None  # master
    iterations: int = 1000  # number of iterations per TPU training loop.

    # training
    do_train: bool = False  # whether to do training
    train_steps: int = 1000  # Number of training steps
    warmup_steps: int = 0  # number of warmup steps
    learning_rate: float = 1e-5  # initial learning rate
    lr_layer_decay_rate: float = 1.0  # Top layer: lr[L] = FLAGS.learning_rate.
    # Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.

    min_lr_ratio: float = 0.0  # min lr ratio for cos decay.
    clip: float = 1.0   # Gradient clipping
    max_save: int = 0  # Max number of checkpoints to save. Use 0 to save all.
    save_steps: int = None  # Save the model for every save_steps. If None, not to save any model.
    train_batch_size: int = 8  # Batch size for training
    weight_decay: float = 0.00  # Weight decay rate
    adam_epsilon: float = 1e-8  # Adam epsilon
    decay_method: str = "poly"  # poly or cos

    # evaluation
    do_eval: bool = False  # whether to do eval
    do_predict: bool = False  # whether to do prediction
    predict_threshold: float = 0  # Threshold for binary prediction.
    eval_split: str = "dev"  # could be dev or test
    eval_batch_size: int = 128  # batch size for evaluation
    predict_batch_size: int = 128  # batch size for prediction.
    predict_dir: str = None  # Dir for saving prediction files.
    eval_all_ckpt: bool = False  # Eval all ckpts. If False, only evaluate the last one.
    predict_ckpt: str = ""  # Ckpt path for do_predict. If None, use the last one.

    # task specific
    task_name: str = None  # Task name
    max_seq_length: int = 128  # Max sequence length
    shuffle_buffer: int = 2048  # Buffer size used for shuffle.
    num_passes: int = 1  # Num passes for processing training data. This is use to batch data without loss for TPUs.
    uncased: bool = False  # Use uncased.
    cls_scope: str = ""  # Classifier layer scope.
    is_regression: bool = False  # Whether it's a regression task.
