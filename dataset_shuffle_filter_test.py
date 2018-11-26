"""Tests for dataset shuffle and filter."""

import os
import numpy as np
import tensorflow as tf
from collections import namedtuple
import functools

Config=namedtuple('Config',
                  ['num_readers', 'shuffle', 'filenames_shuffle_buffer_size',
                  'num_epochs', 'read_block_length', 'shuffle_buffer_size', 'filter'])
Config.__new__.__defaults__ = (4, False, 100, 4, 32, 2048, False)

def read_dataset(file_read_func, input_files, config):
  """Reads a dataset, and handles repetition and shuffling.

  Args:
    file_read_func: Function to use in tf.contrib.data.parallel_interleave, to
      read every individual file into a tf.data.Dataset.
    input_files: A list of file paths to read.
    config: A input_reader_builder.InputReader object.

  Returns:
    A tf.data.Dataset of (undecoded) tf-records based on config.
  """
  # Shard, shuffle, and read files.
  filenames = tf.gfile.Glob(input_files)
  filenames.sort()  # add by liyanan12
  num_readers = config.num_readers
  if num_readers > len(filenames):
    num_readers = len(filenames)
    tf.logging.warning('num_readers has been reduced to %d to match input file '
                       'shards.' % num_readers)
  filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
  if config.shuffle:
    filename_dataset = filename_dataset.shuffle(
        config.filenames_shuffle_buffer_size)
  elif num_readers > 1:
    tf.logging.warning('`shuffle` is false, but the input data stream is '
                       'still slightly shuffled since `num_readers` > 1.')
  filename_dataset = filename_dataset.repeat(config.num_epochs or None)
  records_dataset = filename_dataset.apply(
      tf.contrib.data.parallel_interleave(
          file_read_func,
          cycle_length=num_readers,
          block_length=config.read_block_length,
          sloppy=config.shuffle))
  if config.shuffle:
    records_dataset = records_dataset.shuffle(config.shuffle_buffer_size)
  return records_dataset


class ReadDatasetTest(tf.test.TestCase):
  def setUp(self):
    self._shuffle_path_template = os.path.join(self.get_temp_dir(),'shuffle_%s.record')
    for i in range(4):
      path = self._shuffle_path_template % i
      writer=tf.python_io.TFRecordWriter(path)
      for _ in range(5):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
                }))
        writer.write(example.SerializeToString())
      writer.close()

  def _get_dataset_next(self, files, config, batch_size):

    def decode_func(value):
      features = tf.parse_single_example(value,
              features={'id': tf.FixedLenFeature([], tf.int64)})
      return features

    def filter_func(value):
        #return tf.equal(value['id'], 0)
        return tf.not_equal(value['id'], 0)

    dataset = read_dataset(
        functools.partial(tf.data.TFRecordDataset,
                          buffer_size=8 * 1000 * 1000),
        files,
        config)
    dataset = dataset.map(decode_func)
    if config.filter:
        dataset = dataset.filter(filter_func)
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

  def test_enable_shuffle(self):
    config = Config(shuffle=True, num_epochs=1)
    tf.set_random_seed(1)  # Set graph level seed.
    data = self._get_dataset_next(
        [self._shuffle_path_template % '*'], config, batch_size=20)
    expected_non_shuffle_output = [0, 0, 0, 0, 0,
                                   1, 1, 1, 1, 1,
                                   2, 2, 2, 2, 2,
                                   3, 3, 3, 3, 3]

    with self.test_session() as sess:
      data_out = sess.run(data)
      self.assertTrue(
          np.any(np.not_equal(data_out['id'], expected_non_shuffle_output)))
      #print('liyanan12_data:%s' % data_out)

  def test_enable_filter(self):
    config = Config(num_epochs=1, filter=True)
    tf.set_random_seed(1)  # Set graph level seed.
    data = self._get_dataset_next(
        [self._shuffle_path_template % '*'], config, batch_size=10)
    expected_non_shuffle_output = [0, 0, 0, 0, 0,
                                   1, 1, 1, 1, 1,
                                   2, 2, 2, 2, 2,
                                   3, 3, 3, 3, 3]

    with self.test_session() as sess:
      data_out = sess.run(data)
      print('liyanan12_data:%s' % data_out)
      #self.assertTrue(
      #    np.any(np.not_equal(data_out['id'], expected_non_shuffle_output)))


if __name__ == '__main__':
  tf.test.main()
