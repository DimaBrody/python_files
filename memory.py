import numpy as np


def minimize_memory(data, verbose=True):
    start_mem = data.memory_usage().sum() / 1024 ** 2

    for col in list(data):
        col_type = data[col].dtype

        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    if c_min >= 0:
                        data[col] = data[col].astype(np.uint8)
                    else:
                        data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    if c_min >= 0:
                        data[col] = data[col].astype(np.uint16)
                    else:
                        data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    if c_min >= 0:
                        data[col] = data[col].astype(np.uint32)
                    else:
                        data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    if c_min >= 0:
                        data[col] = data[col].astype(np.uint64)
                    else:
                        data[col] = data[col].astype(np.int64)
            else:
                #             if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #                 data[col] = data[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

    end_mem = data.memory_usage().sum() / 1024 ** 2

    if verbose:
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return data
