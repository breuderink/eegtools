import re, datetime, operator, logging
import numpy as np
from collections import namedtuple
from edfplus import tal, BaseEDFReader, EDFEndOfData

log = logging.getLogger(__name__)

EVENT_CHANNEL = 'BDF Annotations'

def bdf_header(f):
  h = {}
  assert f.tell() == 0  # check file position
  assert f.read(8) == '\xffBIOSEMI'

  # recording info)
  h['local_subject_id'] = f.read(80).strip()
  h['local_recording_id'] = f.read(80).strip()

  # parse timestamp
  (day, month, year) = [int(x) for x in re.findall('(\d+)', f.read(8))]
  (hour, minute, sec)= [int(x) for x in re.findall('(\d+)', f.read(8))]
  h['date_time'] = str(datetime.datetime(year + 2000, month, day, 
    hour, minute, sec))

  # misc
  header_nbytes = int(f.read(8))
  subtype = f.read(44)[:5]
  h['BDF+'] = subtype in ['BDF+C', 'BDF+D']
  h['contiguous'] = subtype != 'BDF+D'
  h['n_records'] = int(f.read(8))
  h['record_length'] = float(f.read(8))  # in seconds
  nchannels = h['n_channels'] = int(f.read(4))

  # read channel info
  channels = range(h['n_channels'])
  h['label'] = [f.read(16).strip() for n in channels]
  h['transducer_type'] = [f.read(80).strip() for n in channels]
  h['units'] = [f.read(8).strip() for n in channels]
  h['physical_min'] = np.asarray([float(f.read(8)) for n in channels])
  h['physical_max'] = np.asarray([float(f.read(8)) for n in channels])
  h['digital_min'] = np.asarray([float(f.read(8)) for n in channels])
  h['digital_max'] = np.asarray([float(f.read(8)) for n in channels])
  h['prefiltering'] = [f.read(80).strip() for n in channels]
  h['n_samples_per_record'] = [int(f.read(8)) for n in channels]
  f.read(32 * nchannels)  # reserved
  
  assert f.tell() == header_nbytes
  return h 


class BaseBDFReader(BaseEDFReader):
  def read_header(self):
    self.header = h = bdf_header(self.file)

    # calculate ranges for rescaling
    self.dig_min = h['digital_min']
    self.phys_min = h['physical_min']
    phys_range = h['physical_max'] - h['physical_min']
    dig_range = h['digital_max'] - h['digital_min']
    assert np.all(phys_range > 0)
    assert np.all(dig_range > 0)
    self.gain = phys_range / dig_range

  
  def read_raw_record(self):
    '''Read a record with data and return a list containing arrays with raw
    bytes.
    '''
    result = []
    for nsamp in self.header['n_samples_per_record']:
      samples = self.file.read(nsamp * 3)
      if len(samples) != nsamp * 3:
        raise EDFEndOfData
      result.append(samples)
    return result

    
  def convert_record(self, raw_record):
    '''Convert a raw record to a (time, signals, events) tuple based on
    information in the header.
    '''
    h = self.header
    dig_min, phys_min, gain = self.dig_min, self.phys_min, self.gain
    time = float('nan')
    signals = []
    events = []
    for (i, samples) in enumerate(raw_record):
      if h['label'][i] == EVENT_CHANNEL:
        ann = tal(samples)
        time = ann[0][0]
        events.extend(ann[1:])
      else:
        # collect 3-byte little-endian integers
        dig = np.fromstring(samples, dtype=np.uint8).reshape(-1, 3)

        # combine the bytes in integers
        dig = np.dot(dig.astype(np.uint32), [1<<0, 1<<8, 1<<16])
        
        # add sign bit.
        dig = (dig << 8).astype(np.int32) >> 8

        phys = (dig - dig_min[i]) * gain[i] + phys_min[i]
        signals.append(phys.astype(np.float32))

    return time, signals, events


def load_bdf(bdffile):
  '''Load an BDF+ file.

  Very basic reader for BDF and BDF+ files. While BaseBDFReader does support
  exotic features like non-homogeneous sample rates and loading only parts of
  the stream, load_bdf expects a single fixed sample rate for all channels and
  tries to load the whole file.

  Parameters
  ----------
  bdffile : file-like object or string

  Returns
  -------
  Named tuple with the fields:
    X : NumPy array with shape p by n.
      Raw recording of n samples in p dimensions.
    sample_rate : float
      The sample rate of the recording. Note that mixed sample-rates are not
      supported.
    sens_lab : list of length p with strings
      The labels of the sensors used to record X.
    time : NumPy array with length n
      The time offset in the recording for each sample.
    annotations : a list with tuples
      BDF+ annotations are stored in (start, duration, description) tuples.
      start : float
        Indicates the start of the event in seconds.
      duration : float
        Indicates the duration of the event in seconds.
      description : list with strings
        Contains (multiple?) descriptions of the annotation event.
  '''
  if isinstance(bdffile, basestring):
    with open(bdffile, 'rb') as f:
      return load_bdf(f)  # convert filename to file

  reader = BaseBDFReader(bdffile)
  reader.read_header()

  h = reader.header
  log.debug('BDF header: %s' % h)

  # get sample rate info
  nsamp = np.unique(
    [n for (l, n) in zip(h['label'], h['n_samples_per_record'])
    if l != EVENT_CHANNEL])
  assert nsamp.size == 1, 'Multiple sample rates not supported!'
  sample_rate = float(nsamp[0]) / h['record_length']

  rectime, X, annotations = zip(*reader.records())
  X = np.hstack(X)
  annotations = reduce(operator.add, annotations)
  chan_lab = [lab for lab in reader.header['label'] if lab != EVENT_CHANNEL]

  # create timestamps
  if reader.header['contiguous']:
    time = np.arange(X.shape[1]) / sample_rate
  else:
    reclen = reader.header['record_length']
    within_rec_time = np.linspace(0, reclen, nsamp, endpoint=False)
    time = np.hstack([t + within_rec_time for t in rectime])

  tup = namedtuple('BDF', 'X sample_rate chan_lab time annotations')
  return tup(X, sample_rate, chan_lab, time, annotations)
