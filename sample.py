import numpy as np
import datetime
from trainer import Trainer
import logger
import utils
import models.model_factory

utils.flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
utils.flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
utils.flags.DEFINE_integer("z_dim", 100, "Dimensions of generator input [100]")
utils.flags.DEFINE_string("model_name", "Model", "Name of the model [Model]")
utils.flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
utils.flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
FLAGS = utils.flags.FLAGS()

utils.exists_or_mkdir(FLAGS.sample_dir)

z_shape = (FLAGS.z_dim,)
image_size = (FLAGS.output_size, FLAGS.output_size)
image_shape = (3,) + image_size

device = utils.get_torch_device()
nn_model = models.model_factory.create_model(FLAGS.model_name, device=device, image_shape=image_shape,z_shape=z_shape)
if not nn_model.load_checkpoint(FLAGS.checkpoint_dir):
    raise RuntimeError('Checkpoint not found')

trainer = Trainer(model=nn_model, batch=None, loss=None, lr=0., reg=None, lambd=None)

z = np.random.randn(FLAGS.sample_size, FLAGS.z_dim).astype(np.float32)
img = trainer.sample(z)

n = int(np.sqrt(FLAGS.sample_size))
now = datetime.datetime.now()
utils.save_images(img, [n, n], './{}/sample_{:02d}_{:02d}_{:02d}:{:02d}:{:02d}.png'.format(FLAGS.sample_dir, now.month, now.day, now.hour, now.minute, now.second))
logger.info("Sample done")