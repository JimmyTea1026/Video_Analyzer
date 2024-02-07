import queue

class StabilizationAlgorithm:
    def __init__(self, q_size) -> None:
        self.queue = queue.Queue(maxsize=q_size)
    
    def get_algorithm_result(self, model_result):
        if not self.queue.full():
            self.queue.put(model_result)
            return model_result
            
        self.queue.get()
        self.queue.put(model_result)
        # sum(1 for _ in iter(q.queue) if _ == target_element)
        true_count = sum(1 for _ in iter(self.queue.queue) if _ == True)
        false_count = sum(1 for _ in iter(self.queue.queue) if _ == False)
        # true_count = self.queue.count(True)
        # false_count = self.queue.count(False)
        
        return true_count > false_count