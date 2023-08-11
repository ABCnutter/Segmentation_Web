import logging  
import os

def set_logger(logging_level, logging_save_dir, logger_work_id):
  # 配置日志级别
    # 创建一个将日志消息输出到终端的处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_level)

    if not os.path.exists(logging_save_dir):
        os.makedirs(logging_save_dir)
    logging_save_path = os.path.join(logging_save_dir, f"work-{logger_work_id}.log")

    file_handler = logging.FileHandler(logging_save_path, mode='a', encoding='utf-8', delay=False)
    file_handler.setLevel(logging_level)

    # 创建一个日志格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 设置处理器的格式器
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 获取根日志记录器
    logger = logging.getLogger(f'{logger_work_id}_logger')
    logger.setLevel(logging_level)
    # 将处理器添加到日志记录器
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger