import os, time
import numpy as np

import traceback
import logger

import utils
import dataset.datasets
import batch_gen

import models.model_factory
import gan_loss
from trainer import Trainer
import lambda_scheduler
from checkpoint import Checkpoint


utils.flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
utils.flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
utils.flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
utils.flags.DEFINE_integer("batch_per_update", 1, "How many batches for one parameters update. [1]")
utils.flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
utils.flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
utils.flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
utils.flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
utils.flags.DEFINE_integer("save_step", 100, "The interval of saveing checkpoints. [500]")
utils.flags.DEFINE_string("dataset", "celeba", "The name of dataset [celebA, mnist, lsun]")
utils.flags.DEFINE_string("data_folder", "./data", "The path to data folder [./data]")
utils.flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
utils.flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
utils.flags.DEFINE_string("log_dir", "log", "Directory name to save the logs [log]")
utils.flags.DEFINE_integer("z_dim", 100, "Dimensions of generator input [100]")
utils.flags.DEFINE_string("model_name", "Model", "Name of the model [Model]")
utils.flags.DEFINE_integer("lambda_switch_steps", 100, "Name of steps to wait before annealing lambda")
utils.flags.DEFINE_boolean('use_averaged_gen', False, 'If use averaged generator for sampling')
utils.flags.DEFINE_integer('n_discriminator', 1, 'Number of discriminator updates per generator')
FLAGS = utils.flags.FLAGS()


def main(_):
    checkpoint_file = 'checkpoint.npz'
    checkpoint = Checkpoint(FLAGS.checkpoint_dir)
    utils.exists_or_mkdir(FLAGS.sample_dir)
    utils.exists_or_mkdir(FLAGS.log_dir)

    logger.info('[Params] lr:%f, size:%d, dataset:%s, av_gen:%d'%(FLAGS.learning_rate, FLAGS.output_size, FLAGS.dataset, int(FLAGS.use_averaged_gen)))

    #dataset
    z_shape = (FLAGS.z_dim,)
    image_size = (FLAGS.output_size, FLAGS.output_size)
    image_shape = (3,) + image_size

    ds = dataset.datasets.from_name(name=FLAGS.dataset, data_folder=FLAGS.data_folder,
                                    output_size=image_size)

    batch = batch_gen.BatchWithNoise(ds, batch_size=FLAGS.batch_size, z_shape=z_shape,num_workers=10)

    #initialize device
    device = utils.get_torch_device()

    #model
    nn_model = models.model_factory.create_model(FLAGS.model_name, device=device, image_shape=image_shape,z_shape=z_shape, use_av_gen=FLAGS.use_averaged_gen)
    nn_model.register_checkpoint(checkpoint)

    loss = gan_loss.js_loss()
    #lambd = lambda_scheduler.Constant(0.1)
    lambd = lambda_scheduler.ThresholdAnnealing(1000., threshold=loss.lambda_switch_level, min_switch_step=FLAGS.lambda_switch_steps, verbose=True)
    checkpoint.register('lambda', lambd, True)

    trainer = Trainer(model=nn_model, batch=batch, loss=loss, lr=FLAGS.learning_rate,
                      reg='gp', lambd=lambd)
    trainer.sub_batches = FLAGS.batch_per_update
    trainer.register_checkpoint(checkpoint)

    it_start = checkpoint.load(checkpoint_file)

    ##========================= LOAD CONTEXT ================================##
    context_path = os.path.join(FLAGS.checkpoint_dir, 'context.npz')
    
    sample_seed = None
    if os.path.exists(context_path):
        sample_seed = np.load(context_path)['z']
        if sample_seed.shape[0] != FLAGS.sample_size or sample_seed.shape[1] != FLAGS.z_dim:
            sample_seed = None
            logger.info('Invalid sample seed')
        else:
            logger.info('Sample seed loaded')
    
    if sample_seed is None:
        sample_seed = batch.sample_z(FLAGS.sample_size).data.numpy()
        np.savez(context_path, z = sample_seed)

    ##========================= TRAIN MODELS ================================##
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

    bLambdaSwitched = (it_start == 0)
    n_too_good_d = []

    number_of_iterations = FLAGS.epoch*batches_per_epoch
    for it in range(number_of_iterations):
        start_time = time.time()
        iter_counter = it + it_start

        # updates the discriminator
        #if iter_counter < 25 or iter_counter % 500 == 0:
        #    d_iter = 20
        #else:
        #    d_iter = 5
        if bLambdaSwitched:
            #if lambda was switched we want to keep discriminator optimal
            logger.info('[!] Warming up discriminator')
            d_iter = 25
        else:
            d_iter = FLAGS.n_discriminator
#
        errD, s, errG, b_too_good_D = trainer.update(d_iter, 1)

        #updating lambda
        n_too_good_d.append(b_too_good_D)
        if len(n_too_good_d) > 20:
            del n_too_good_d[0]               
                
        bLambdaSwitched = lambd.update(errD)
        if not bLambdaSwitched and sum(n_too_good_d) > 10:
            bLambdaSwitched = lambd.switch()

        end_time = time.time()

        iter_time = end_time - start_time
        total_time += iter_time

        logger.info("[%2d/%2d] time: %4.4f, d_loss: %.8f, s: %.4f, g_loss: %.8f" % (iter_counter, it_start + number_of_iterations, iter_time, errD, s, errG))

        time_array.append(total_time)
        d_loss_array.append(errD)

        if np.mod(iter_counter, FLAGS.sample_step) == 0 and it > 0:
            img = trainer.sample(sample_seed)

            n = int(np.sqrt(FLAGS.sample_size))
            utils.save_images(img, [n, n], './{}/train_{:02d}.png'.format(FLAGS.sample_dir, iter_counter))
            logger.info("[Sample] d_loss: %.8f, g_loss: %.8f" % (errD, errG))

            np.savez(os.path.join(FLAGS.log_dir, 'log.npz'), loss = d_loss_array, time = time_array)

        if np.mod(iter_counter, FLAGS.save_step) == 0 and it > 0:
            logger.info("[*] Saving checkpoints...")
            checkpoint.save(iter_counter, checkpoint_file)
            logger.info("[*] Saving checkpoints SUCCESS!")

    logger.info("[*] Saving checkpoints...")
    checkpoint.save(iter_counter, checkpoint_file)
    logger.info("[*] Saving checkpoints SUCCESS!")

import threading
if __name__ == '__main__':
    try:
        main('')
    except Exception as e:
        logger.error(traceback.format_exc())

