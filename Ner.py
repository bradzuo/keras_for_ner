# -*-coding:utf-8-*-
from keras.models import Sequential
from keras.layers import LSTM,Embedding,Bidirectional,Dropout
import pickle,os,sys
curPath = os.path.abspath(os.path.dirname(__file__))
print('curPath',curPath)
sys.path.append(curPath)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from keras_contrib.layers.crf import CRF
from Proc_data import process_data

import logging
# logging.basicConfig(filename=os.path.join(os.getcwd(), 'log.txt'), level=logging.INFO)
#os.getcwd()获取当前工作路径，'log.txt'为保存的文件名称，logging.DEBUG为log的优先级别 logging.debug(model)



def ner_model(Train=True):

    if Train==True:
        (train_x, train_y), (test_x, test_y), (num_words, entity_tags,embedding_dim,embedding_matrics,maxlen) = process_data().load_data()
        print('训练数据数量',train_x.shape,'训练数据标签数量',train_y.shape)
        train_x[train_x>=num_words]=0
        test_x[test_x>=num_words]=0
    else:
        with open(curPath+'/Pickle/config.pkl', 'rb') as cnf:
            (num_words, entity_tags,embedding_dim,embedding_matrics,maxlen) = pickle.load(cnf)

    model = Sequential()

    model.add(Embedding(input_dim=num_words,
                        output_dim=embedding_dim,
                        # weights=[ ]
                        weights=[embedding_matrics],
                        mask_zero=True,
                        input_length = maxlen,
                        trainable = False))

    # model.add(Dropout(0.5))
    # model.add(Bidirectional(LSTM(units=256, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))

    crf = CRF(len(entity_tags), sparse_target=True)
    model.add(crf)
    model.summary()
    # 优化器采用adam
    optimizer = Adam(lr=1e-3)
    model.compile(optimizer=optimizer, loss=crf.loss_function, metrics=[crf.accuracy])

    if Train == True:

        path_checkpoint = 'Model/ner_checkpoint.h5'
        checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1, save_weights_only=True,
                                     save_best_only=True)

        # early_stopping如果3个epoch内validation_loss没有改善则停止训练
        earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        # 自动降低learning_rate
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-5, patience=0, verbose=1)
        # tensorboard可视化
        tensorboard = TensorBoard(log_dir='Tensorboard')
        callbacks = [earlystopping, lr_reduction, checkpoint, tensorboard]

        logging.debug(model)

        try:
            model.load_weights('Model/crf.h5')
        except:
            pass
        # 训练
        model.fit(train_x, train_y,
                  batch_size=64,
                  epochs=100,
                  validation_split=0.1,
                  shuffle=True,
                  callbacks=callbacks)
        result = model.evaluate(test_x, test_y)
        print('Accuracy:{}'.format(result[1]))

        model.save('Model/crfApril.h5')

    else:
        return model,num_words

if __name__=='__main__':

    ner_model()