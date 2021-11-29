from datetime import datetime
import logging
import datetime

now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger = logging
log_file = f"./logs/teacher-student.log"
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)   
