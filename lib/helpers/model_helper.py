from lib.models.MonoSGC import MonoSGC


def build_model(cfg,mean_size):
    if cfg['type'] == 'MonoSGC':
        return MonoSGC(backbone=cfg['backbone'], neck=cfg['neck'], mean_size=mean_size)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])
