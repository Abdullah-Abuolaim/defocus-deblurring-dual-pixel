"""
This is the main module for linking different components of the CNN-based model
proposed for the task of image defocus deblurring based on dual-pixel data. 

Copyright (c) 2020-present, Abdullah Abuolaim
This source code is licensed under the license found in the LICENSE file in
the root directory of this source tree.

This code imports the modules and starts the implementation based on the
configurations in config.py module.

Note: this code is the implementation of the "Defocus Deblurring Using Dual-
Pixel Data" paper accepted to ECCV 2020. Link to GitHub repository:
https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel

Email: abuolaim@eecs.yorku.ca
"""

from model import *
from config import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

check_dir(path_write)

if op_phase=='train':
    data_random_shuffling('train')
    data_random_shuffling('val')
    
    in_data = Input(batch_shape=(None, patch_h, patch_w, nb_ch_all))
    
    model = Model(inputs=in_data, outputs=unet(in_data))
    model.summary()
    model.compile(optimizer = Adam(lr = lr_[0]), loss = 'mean_squared_error')
    
    # training callbacks
    model_checkpoint = ModelCheckpoint(path_save_model, monitor='loss',
                            verbose=1, save_best_only=True)

    l_r_scheduler_callback = LearningRateScheduler(schedule=schedule_learning_rate)
    
    history = model.fit_generator(generator('train'), nb_train, nb_epoch,
                        validation_data=generator('val'),
                        validation_steps=nb_val,callbacks=[model_checkpoint,
                        l_r_scheduler_callback])
    
    np.save(path_write+'train_loss_arr',history.history['loss'])
    np.save(path_write+'val_loss_arr',history.history['val_loss'])

elif op_phase=='test':
    data_random_shuffling('test')
    model = load_model(path_save_model, compile=False)
    # fix input layer size
    model.layers.pop(0)
    input_size = (img_h, img_w, nb_ch_all)
    input_test = Input(input_size)
    output_test=model(input_test)
    model = Model(input = input_test, output = output_test)
    
    img_mini_b = 1

    test_imgaes, gt_images = test_generator(total_nb_test)
    predictions = model.predict(test_imgaes,img_mini_b,verbose=1)
                            
    save_eval_predictions(path_write,test_imgaes,predictions,gt_images)
    
    np.save(path_write+'mse_arr',np.asarray(mse_list))
    np.save(path_write+'psnr_arr',np.asarray(psnr_list))
    np.save(path_write+'ssim_arr',np.asarray(ssim_list))
    np.save(path_write+'mae_arr',np.asarray(mae_list))
    np.save(path_write+'final_eval_arr',[np.mean(np.asarray(mse_list)),
                                          np.mean(np.asarray(psnr_list)),
                                          np.mean(np.asarray(ssim_list)),
                                          np.mean(np.asarray(mae_list))])