






class VinConfig():
    No = 3  # the number of object
    img_folder = "./vin/img/"  # image folder
    data_folder = "./vin/data/"  # data folder
    frame_num = 50  # The Number of Saved Frame per each simul
    frame_step = 1  # The difference between saved frames
    roll_num = 20  # The Number of Rollout
    set_num = 10  # The Number of set
    weight=32
    height=32
    col_dim=4
    batch_size=4
    checkpoint_dir="./checkpoint/"
    load=False
    log_dir="./log"