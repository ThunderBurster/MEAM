import time

class Utils:
    @staticmethod
    def get_train_name_with_time(name: str) -> str:
        timestr = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        return name + '_' + timestr