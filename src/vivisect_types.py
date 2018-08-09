from uuid import uuid4


class ModelInfo(object):
    """
    model_name = specified directly to the probe() command
    state = (train|dev|test|none)
    epoch = positive integer
    framework = name of framework this model is defined in
    """
    __slots__ = ("model_name", "state", "epoch", "framework", "misc")
    def __init__(self, model_name=None, state=None, epoch=0, framework=None, misc={}):
        self.model_name = (model_name if isinstance(model_name, str) else str(uuid4()))
        self.state = (state if isinstance(state, str) else "train")
        self.epoch = epoch
        self.misc = misc
        self.framework = framework

    def __str__(self):
        return "{} ({})".format(self.model_name, self.framework)

class OperationInfo(object):
    """
    operation_name = typically framework-specific, e.g. "model.encoder"
    operation_type = typically framework-specific, e.g. "RNN" or "Dense"
    """

    __slots__ = ("operation_name", "operation_type", "misc")
    
    def __init__(self, operation_name, operation_type, misc={}):
        self.operation_name = operation_name
        self.operation_type = operation_type
        self.misc = misc
        
    def __str__(self):
        return "{} ({})".format(self.operation_name, self.operation_type)
        
    
class ArrayInfo(object):
    """
    array_name = sequential number or interpretable name
    type = (activation_input|activation_output|gradient_input|gradient_output|parameter)
    """
    __slots__ = ("array_name", "array_type", "misc")
    def __init__(self, array_name, array_type, misc={}):
        self.array_name = array_name
        self.array_type = array_type
        self.misc = misc
