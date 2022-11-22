"""
    ids:{
        "1":{
            "state":,
            "distance":,
            "count":
        },
        "2":{
            "state":,
            "distance":,
            "count":
        },
    }
    state       -> (string) "tracked","entry", "exit"
    distance    -> (float)
    count       ->
"""
import reid

class IDAssigner():

    def __init__(self, reid_model):
        self.last_id = 0
        self.ids = []
        self.reid = reid_model
        # Inisialisasi model


    def update_ids(ids, datas):
        self.ids.append(datas)

    def next_id():
        self.last_id += 1
        return self.last_id

    def is_id_registered(track):
        patch = reid_inference(track.tlbr)
        

        if registered:
            return id
        else:
            return None

        

    # def say_hello(self):
    #     return "hello"