def setup(P, *others):

    if P.mode == "Main_Trainer":
        from .trainer import train
    else:
        raise NotImplementedError(f"Mode {P.mode} not implemented")

    dir_name = P.dataset+'_'+P.source
    fname = dir_name
    if P.suffix != "":
        fname += f"_{P.suffix}"

    fname += f"_do_{P.dataorder}"

    return train, fname
