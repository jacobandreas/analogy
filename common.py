from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("device", "cuda:0", "device to use")

def _device():
    return FLAGS.device
