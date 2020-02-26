from maskrcnn_benchmark.modeling import registry


def build_one_stage_head(cfg, in_channels):
    assert cfg.MODEL.ONE_STAGE_HEAD in registry.ONE_STAGE_HEADS, \
        "cfg.MODEL.ONE_STAGE_HEAD: {} are not registered in registry".format(
            cfg.MODEL.ONE_STAGE_HEAD)
    return registry.ONE_STAGE_HEADS[cfg.MODEL.ONE_STAGE_HEAD](cfg, in_channels)
