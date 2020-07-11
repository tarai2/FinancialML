import requests
from finacialml.metrics import *
from finacialml.bar import *
from finacialml.cv import *
from finacialml.labeling import *


def sample():
    try:
        # ここを Mock したい
        res = requests.get("http://checkip.amazonaws.com/")

    except requests.RequestException as e:
        print(e)
        raise e

    return {
        "statusCode": res.status_code,
        "ip": res.text,
    }


def test_sample(mocker):
    # 嘘のレスポンスを作成
    responseMock = mocker.Mock()
    responseMock.status_code = 404
    responseMock.text = '127.0.0.1'
    # requests.get の戻り値を patch する
    mocker.patch('requests.get').return_value = responseMock

    actual = sample()
    assert actual['statusCode'] == 404
    assert actual['ip'] == '127.0.0.1'
