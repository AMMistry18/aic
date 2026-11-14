"""Development-only inference thread guard; checkpoints/geometry stay unchanged.

Ultralytics CPU initialization resets Torch's thread count. Its prediction
callback runs after backend initialization and before actual image inference,
so it can restore the explicit budget for both full-frame and crop passes.
"""


def bind_inference_threads(model, *, cpu_threads=1):
    if not isinstance(cpu_threads,int) or not 1<=cpu_threads<=8:
        raise ValueError('Explicit inference budget must be 1-8 threads')
    def before_prediction(predictor):
        import torch
        if torch.get_num_threads()!=cpu_threads:
            torch.set_num_threads(cpu_threads)
    model.add_callback('on_predict_start',before_prediction)
    return model
