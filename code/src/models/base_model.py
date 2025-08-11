import time;

class BaseModel:
    def __init__(self, model_name: str):
        self.model_name = model_name;
        self._X_label = "wave_obs_AA";
        self._y_label = "residual";
    
    def __time_data_processing(self, func: callable):
        start_time = time.time()
        result = func()
        end_time = time.time()
        return {
            'function': func.__name__,
            'model_name': self.model_name,
            'result': result,
            'time_taken': end_time - start_time
        } 

    def time_train(self, data):
        return self.__time_data_processing(lambda: self.train(data))

    def time_predict(self, data):
        return self.__time_data_processing(lambda: self.predict(data))
