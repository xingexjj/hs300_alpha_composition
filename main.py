from run import run

def main():
    config = {
        'type':        'all',
        'DATA_PATH':   './data',
        'MODEL_PATH':  './models',
        'train_start': '20130701',
        'train_end':   '20201231',
        'valid_start': '20210101',
        'valid_end':   '20211231',
        'test_start':  '20211201',
        'test_end':    '20230930',
        'period':        20,
        'batch_size':    1024,
        'learning_rate': 1e-4,
        'n_epochs':      30,
        'early_stop':    5,
        'seed':          42
    }

    run(config)

if __name__ == '__main__':
    main()