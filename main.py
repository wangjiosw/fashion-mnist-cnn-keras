# coding: utf-8

import sys
import setting
import cnn_model


def main(argv):
    if len(sys.argv) == 2:
        setting.mode = argv[1]
    
    model = cnn_model.cnn_model(setting)

    if setting.mode == '-test':
        model.test()
    elif setting.mode == '-train':
        model.train()
    else:
        print('Input Mode Error !')
    

    


if __name__ == "__main__":
    main(sys.argv)