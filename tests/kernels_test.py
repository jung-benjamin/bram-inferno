#! /usr/bin/env python3

"""Tests for the kernels module"""

import json
from newbie import kernels

def model_dict():
    """Dictionary of model attributes"""
    d = {
    'Params': [33.90938176807465, 8.014395700596914, 13.602488333804923, 1e-10],
    'LAMBDA': [[0.12477547120932896, 0.0], [0.0, 0.07351596086392508]],
    'alpha_': [9495.266495455951, 899.6204311144278,
               -4344.025299121975, 9857.440586445668,
               1399.307539288503, 12619.751314661358,
               -16391.488934019377, -2434.233532291585,
               -5138.175102070208, -5963.440700278297],
    'x_train': [[1318.463134765625, 1981.8115234375],
                [368.463134765625, 6981.8115234375],
                [843.463134765625, 9481.8115234375],
                [1793.463134765625, 4481.8115234375],
                [130.963134765625, 3231.8115234375],
                [1080.963134765625, 8231.8115234375],
                [1555.963134765625, 5731.8115234375],
                [605.963134765625, 731.8115234375],
                [724.713134765625, 6356.8115234375],
                [1674.713134765625, 1356.8115234375]],
    'y_train': [8038470000000000.0, 2078810000000000.0, 4963040000000000.0,
                1.12809e+16, 721153000000000.0, 6478440000000000.0,
                9640250000000000.0, 3495480000000000.5, 4223540000000000.0,
                1.0455700000000002e+16],
    'y_trafo': ['StandardNormalize', [6137578300000000.0, 3635217363511814.5]],
    'x_trafo': ['Normalize', [1793.463134765625, 9481.8115234375]],
    'kernel': 'AnisotropicSquaredExponential'
    }
    return d

def model_dict_2():
    """Dictionary of model attributes with different dimensions."""
    d = {
        'Params': [0.48913262996451534, 0.040561695584638846, 0.166614138050176, 0.22683997611062934, 1e-10],
        'LAMBDA': [
            [24.65380171086119, 0.0, 0.0],
            [0.0, 6.001891626380765, 0.0],
            [0.0, 0.0, 4.408394045643447]
        ],
        'alpha_': [-7.19769972937485, 31.498281483019504, -44.86648129597428,
                    14.01477959257388, 13.23626773606846, 7.437602099215877, 
                    -4.790259976132744, -13.015160939190217, 5.172611481312183, 
                    -32.82451940917667],
        'x_train': [[30.0164794921875, 85.0579833984375, 2183.53271484375],
                    [45.0164794921875, 150.0579833984375, 2683.53271484375],
                    [52.5164794921875, 52.5579833984375, 2433.53271484375],
                    [37.5164794921875, 117.5579833984375, 2933.53271484375],
                    [41.2664794921875, 36.3079833984375, 2558.53271484375],
                    [56.2664794921875, 101.3079833984375, 2058.53271484375],
                    [48.7664794921875, 68.8079833984375, 2808.53271484375],
                    [33.7664794921875, 133.8079833984375, 2308.53271484375],
                    [35.6414794921875, 60.6829833984375, 2871.03271484375],
                    [50.6414794921875, 125.6829833984375, 2371.03271484375]],
        'y_train': [
            4.28468e+18, 9.5619e+18, 1.3683e+19, 7.03437e+18, 8.62215e+18, 
            1.44569e+19, 1.18291e+19, 5.33259e+18, 6.31171e+18, 1.17968e+19
        ],
        'y_trafo': [
            'StandardNormalize', [1.00975247e+19, 3.778837987330014e+18]
        ],
        'x_trafo': [
            'Normalize',
            [59.8992919921875, 159.7064208984375, 2996.03271484375]
        ],
        'kernel': 'AnisotropicSquaredExponential'
    }
    return d

def store_model(temp):
    """Store model_dict to a json file"""
    with open(temp, 'w') as f:
        json.dump(model_dict(), f)

def store_model_2(temp):
    """Store model_dict_2 to a json file"""
    with open(temp, 'w') as f:
        json.dump(model_dict_2(), f)

def test_asqe_predictor_from_file(tmp_path):
    """Test for the from_file method of the ASQEKernelPredictor"""
    p = tmp_path / 't.json'
    store_model(p)
    k = kernels.ASQEKernelPredictor.from_file(p)
    assert True

def test_asqe_predict(tmp_path):
    p = tmp_path / 't.json'
    store_model(p)
    k = kernels.ASQEKernelPredictor.from_file(p)
    k.predict([700, 5000])
    assert True

def test_asqe_predict_many(tmp_path):
    p = tmp_path / 't.json'
    store_model(p)
    k = kernels.ASQEKernelPredictor.from_file(p)
    k.predict_many([[700, 5000], [500, 1000], [600, 3000]])
    assert True

def test_predict_sum2(tmp_path):
    """Test for the PredictSum2 class"""
    p1 = tmp_path / 't1.json'
    p2 = tmp_path / 't2.json'
    store_model(p1)
    store_model(p2)
    k = kernels.PredictorSum2.from_file(p1, kernels.ASQEKernelPredictor,
                                        p2, kernels.ASQEKernelPredictor)
    k.predict([2, *[700, 5000], 3, *[600, 4000]])
    assert True

def test_predict_quotient(tmp_path):
    """Test for the PredictorQuotient class"""
    p1 = tmp_path / 't1.json'
    p2 = tmp_path / 't2.json'
    store_model(p1)
    store_model(p2)
    k = kernels.PredictorQuotient.from_file(p1, kernels.ASQEKernelPredictor,
                                            p2, kernels.ASQEKernelPredictor)
    k.predict([700, 5000])
    assert True

def test_linear_combination(tmp_path):
    """Test for the LinearCombination class"""
    p1 = tmp_path / 't1.json'
    p2 = tmp_path / 't2.json'
    store_model(p1)
    store_model_2(p2)
    k = kernels.LinearCombination.from_file(
        (p1, kernels.ASQEKernelPredictor), (p2, kernels.ASQEKernelPredictor)
    )
    k.predict([2, *[700, 5000], 3, *[600, 4000, 200]])
    assert True

def test_quotient(tmp_path):
    """Test for the Quotien class"""
    p1 = tmp_path / 't1.json'
    p2 = tmp_path / 't2.json'
    store_model(p1)
    store_model(p2)
    k = kernels.Quotient.from_file(
        (p1, kernels.ASQEKernelPredictor), (p2, kernels.ASQEKernelPredictor)
    )
    k.predict([700, 5000])
    assert True
