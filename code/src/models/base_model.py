import time;

class BaseModel:
    def __init__(self, model_name: str):
        self.model_name = model_name;

        self._trained = False;

        self._max_iterations = 1000;
        self._eval_every = 10;
        self._tolerance = 1e-6;
        self._max_stalls = 10;
        self._stall_count = 0;

    def __time_data_processing(self, func: callable, func_name: str, data_X):
        start_time = time.time()
        result = func()
        end_time = time.time()
        return {
            'function': func_name,
            'model_name': self.model_name,
            'X': data_X,
            'result': result,
            'time_taken': end_time - start_time
        } 

    def time_train(self, data_X, data_y):
        return self.__time_data_processing(lambda: self.train(data_X, data_y), "train", data_X)

    def time_predict(self, data_X, data_y):
        return self.__time_data_processing(lambda: self.predict(data_X, data_y), "predict", data_X)
