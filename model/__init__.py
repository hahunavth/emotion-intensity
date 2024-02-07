from model.rank_model import RankModel

__factory_model = {
    'rank': RankModel,
}

__args_dict_model = {
    'rank': {
        "fft_dim": 128,
        "n_emotion": 5,
        "stats_path": "datasets/esd_processed/stats.json",
    },
}


def create_model(name: str, **kwargs):
    assert name in __factory_model, f'invalid model_name: {name}'
    _kwargs = {k: v for k, v in kwargs.items() if k in __args_dict_model[name]}
    default_kwargs = __args_dict_model[name]
    new_kwargs = {**default_kwargs, **_kwargs}
    model = __factory_model[name](**new_kwargs)
    return model, new_kwargs


if __name__ == "__main__":
    model = create_model('rank')
    print(model)