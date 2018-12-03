import os, time
import numpy as np
from random import shuffle
import torch

import utils
import dataset.datasets
import batch_gen
#import optimizers

import models.model
import gan_loss
import training
import signal


utils.flags.DEFINE_string("action", 'train', "action to do [train]")
utils.flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
utils.flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
utils.flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
utils.flags.DEFINE_integer("train_size", int(1e+6), "The size of train images [np.inf]")
utils.flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
utils.flags.DEFINE_integer("batch_per_update", 1, "How many batches for one parameters update. [1]")
utils.flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
utils.flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
utils.flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
utils.flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
utils.flags.DEFINE_integer("save_step", 500, "The interval of saveing checkpoints. [500]")
utils.flags.DEFINE_string("dataset", "celeba", "The name of dataset [celebA, mnist, lsun]")
utils.flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
utils.flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
utils.flags.DEFINE_string("log_dir", "log", "Directory name to save the logs [log]")
utils.flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
utils.flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
utils.flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
utils.flags.DEFINE_integer("z_dim", 100, "Dimensions of generator input [100]")
FLAGS = utils.flags.FLAGS()



def main(_):
    utils.exists_or_mkdir(FLAGS.checkpoint_dir)
    utils.exists_or_mkdir(FLAGS.sample_dir)
    utils.exists_or_mkdir(FLAGS.log_dir)

    batch_size = FLAGS.batch_size
    if FLAGS.action == 'sample':
        batch_size = FLAGS.sample_size

    #dataset
    ds = dataset.datasets.from_name(FLAGS.dataset, batch_size = batch_size, output_size=(FLAGS.output_size, FLAGS.output_size))

    with batch_gen.ThreadedBatch(batch_gen.BatchWithNoise(ds, z_dim = FLAGS.z_dim)) as batch:
        #initialize device
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            cur_device = torch.cuda.current_device()
            device = torch.device('cuda:'+str(cur_device))
            print ('CUDA device: ' + torch.cuda.get_device_name(cur_device))
        else:
            device = torch.device('cpu:0')
            print ('CUDA not available')

        #model
        nn_model = models.model.DeepResidualModel(device=device, batch = batch, g_tanh = True, g_act = 'LeakyReLU', d_act = 'LeakyReLU', batch_norm = True)

        trainer = training.Trainer(model = nn_model, batch = batch, loss = gan_loss.js_loss, lr = FLAGS.learning_rate, reg = 'gp', lambd = 10.)
        trainer.sub_batches = FLAGS.batch_per_update

        model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)
        save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)

        # load the latest checkpoints
        if nn_model.load_checkpoint(save_dir):
            print ('[*] checkpoint loaded')
        else:
            print ('[*] checkpoint not found')

        if FLAGS.action == 'sample':
            batch_z, batch_images = batch.get_batch()
            img = trainer.sample(batch_z)

            n = int(np.sqrt(FLAGS.sample_size))

            n_file = np.random.randint(10000)
            utils.save_images(img[:n*n], [n, n], './{}/sample_{:06d}.png'.format(FLAGS.sample_dir, n_file))
            print("Images saved")

        elif FLAGS.action == 'train':
            sample_seed, sample_images = batch.get_samples(FLAGS.sample_size)

            ##========================= TRAIN MODELS ================================##
            iter_counter = 0
            batches_per_epoch = 10000
            total_time = 0

            d_loss_array = []
            time_array = []

            score_array = []
            score_time_array = []

            #print('Pretraining discriminator')
            #n_pretrain_steps = 100
            #for i in range(n_pretrain_steps):
            #    batch_z, batch_images = batch.get_batch()
            #    errD, errS, _ = sess.run([d_loss_orig, s, d_optim], feed_dict={z: batch_z, real_images: batch_images })
            #    if i % 10 == 0:
            #        print ("[%2d/%2d]" % (i, n_pretrain_steps))
            #print('Done')

            for epoch in range(FLAGS.epoch):
                for b in range(batches_per_epoch):
                    start_time = time.time()

                    # updates the discriminator
                    #if iter_counter < 1000:
                    #    d_iter = 10
                    #else:
                    #    d_iter = 2
                    d_iter = 2

                    errD, s, errG = trainer.update(d_iter, 1)

                    end_time = time.time()

                    iter_time = end_time - start_time
                    total_time += iter_time

                    print("[%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, s: %.4f, g_loss: %.8f" % (epoch, FLAGS.epoch, b+1, batches_per_epoch, iter_time, errD, s, errG))

                    time_array.append(total_time)
                    d_loss_array.append(errD)

                    iter_counter += 1
                    #if np.mod(iter_counter, 1000) == 0:
                    #    sess.run(lambd.assign(tf.maximum(lambd/2., 1e-2)))

                    if np.mod(iter_counter, FLAGS.sample_step) == 0 or iter_counter == 1:
                        img = trainer.sample(sample_seed)

                        n = int(np.sqrt(FLAGS.sample_size))
                        utils.save_images(img, [n, n], './{}/train_{:02d}_{:04d}.png'.format(FLAGS.sample_dir, epoch, b+1))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (errD, errG))

                        #m_score = evaluation.evaluation.evaluate(sess, trainer, batch, 'is')
                        #score_array.append(m_score)
                        #score_time_array.append(total_time)

                        np.savez(os.path.join(FLAGS.log_dir, 'log.npz'), loss = d_loss_array, time = time_array)

                    if np.mod(iter_counter, FLAGS.save_step) == 0:
                        # save current network parameters
                        print("[*] Saving checkpoints...")
                        nn_model.save_checkpoint(save_dir)
                        print("[*] Saving checkpoints SUCCESS!")

            print("[*] Saving checkpoints...")
            nn_model.save_checkpoint(save_dir)
            print("[*] Saving checkpoints SUCCESS!")

        else:
            raise ValueError('unknown action')

if __name__ == '__main__':
    main('')
