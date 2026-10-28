"""Posttrain method selection; importing never initializes workers or tracking."""


def build_posttrain(config, runtime, logger, *, parent=None, resume=True):
    method = config['stages']['posttrain'].get('method', 'truncated')
    if method == 'truncated':
        from ..engine import Training
        return Training(config, 'posttrain', runtime, logger, parent=parent, resume=resume)
    if method == 'rl':
        from .rl import RLPosttrain
        return RLPosttrain(config, runtime, logger, parent=parent, resume=resume)
    raise ValueError(f'Unknown posttrain method: {method}')
