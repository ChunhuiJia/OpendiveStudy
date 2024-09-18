watch -n 2 nvidia-smi  每隔2s刷新一次gpu占用率情况
tensorboard --logdir runs --bind_all  # 运行此命令可以看opendive的tensorboard的训练过程
python main.py  # 运行主训练程序