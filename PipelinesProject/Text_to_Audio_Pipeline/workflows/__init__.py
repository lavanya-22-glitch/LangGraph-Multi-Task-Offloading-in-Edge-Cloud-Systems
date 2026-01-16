import pickle, os

class Fake: # a fake object
    def __init__(self, **attributes):
        for name,value in attributes.items(): setattr(self, name, value)

def Pack(task, buffer=None):
    if buffer is None:
        from io import BytesIO
        buffer = BytesIO()
        pickle.dump(task, buffer)
    else:
        buffer = os.path.abspath(buffer)
        with open(buffer, 'wb') as f: pickle.dump(task, f)
    
    return buffer

def Unpack(buffer):
    if not isinstance(buffer, str):
        buffer.seek(0)
        task = pickle.load(buffer)
    else:
        buffer = os.path.abspath(buffer)
        with open(buffer, 'rb') as f: task = pickle.load(f)
    return task