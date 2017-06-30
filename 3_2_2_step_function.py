def step_function(x):
    # Numpy(行列)に対応できない
    # if x > 0:
    #     return 1
    # else:
    #     return 0

    # 引数のxが行列の場合、boolの配列を返す
    y = x > 0
    # astypeで型変換、true -> 1, false -> 0
    return y.astype(np.int)
