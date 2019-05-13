import numpy as np
import torch
import datetime
import logger
import utils
import models.model_factory
from checkpoint import Checkpoint

utils.flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
utils.flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
utils.flags.DEFINE_integer("z_dim", 100, "Dimensions of generator input [100]")
utils.flags.DEFINE_string("model_name", "Model", "Name of the model [Model]")
utils.flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
utils.flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
utils.flags.DEFINE_boolean('use_averaged_gen', False, 'If use averaged generator for sampling')
utils.flags.DEFINE_integer('n_samples', 1, 'Number of batches')
utils.flags.DEFINE_integer("checkpoint_it_to_load", -1, "Iteration to restore [-1]")

FLAGS = utils.flags.FLAGS()

checkpoint = Checkpoint(FLAGS.checkpoint_dir)

utils.exists_or_mkdir(FLAGS.sample_dir)

z_shape = (FLAGS.z_dim,)
image_size = (FLAGS.output_size, FLAGS.output_size)
image_shape = (3,) + image_size

device = utils.get_torch_device()
nn_model = models.model_factory.create_model(FLAGS.model_name,
                                             device=device,
                                             image_shape=image_shape,
                                             z_shape=z_shape,
                                             use_av_gen=FLAGS.use_averaged_gen)
nn_model.register_checkpoint(checkpoint)

if not checkpoint.load(FLAGS.checkpoint_it_to_load):
    raise RuntimeError('Cannot load checkpoint')

now = datetime.datetime.now()
for i in range(FLAGS.n_samples):
    z = np.random.randn(FLAGS.sample_size, FLAGS.z_dim).astype(np.float32)
    z = torch.tensor(z, device=device)

    with torch.no_grad():
        if hasattr(nn_model, 'av_g_model'):
            nn_model.av_g_model.eval()
            gen_samples = nn_model.av_g_model(z)
        else:
            nn_model.g_model.eval()
            gen_samples = nn_model.g_model(z)
            nn_model.g_model.train()

        gen_samples = torch.clamp(gen_samples, -1., 1.)

    gen_samples = gen_samples.data.cpu().numpy()

    n = int(np.sqrt(FLAGS.sample_size))
    utils.save_images(gen_samples, [n, n], './{}/sample_{:02d}_{:02d}_{:02d}:{:02d}:{:02d}__{:d}.png'.format(FLAGS.sample_dir, now.month, now.day, now.hour, now.minute, now.second, i))

logger.info("Sample done")