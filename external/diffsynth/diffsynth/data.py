import os, glob, functools
import librosa
import torch
from torch.utils.data import Subset, Dataset, DataLoader, random_split, ConcatDataset, SubsetRandomSampler, BatchSampler
import pytorch_lightning as pl
import numpy as np
from diffsynth.f0 import process_f0

def mix_iterable(dl_a, dl_b):
    for i, j in zip(dl_a, dl_b):
        yield i
        yield j

class ReiteratableWrapper():
    def __init__(self, f, length):
        self._f = f
        self.length = length

    def __iter__(self):
        # make generator
        return self._f()

    def __len__(self):
        return self.length

def _check_pairing(audio_files, other_files, base_dir, kind):
    """Audio and its targets are paired by SORTED INDEX and by nothing else.

    __getitem__ reads raw_files[idx] and param_files[idx] from two independent
    globs. Nothing ties a clip to its own parameters, so a directory whose two
    halves disagree -- a merge that renumbered one side, a partial generation,
    an interrupted copy -- trains every clip against another clip's targets.

    IT DOES NOT RAISE ANYWHERE. The loss still falls, just to a worse floor,
    which reads as a hard task rather than as a broken dataset. That is the
    whole reason for this check: the failure has no symptom of its own.

    Compares stems rather than counts, because equal counts is the case that
    gets through: two shards merged with colliding numbering have exactly as
    many .wav as .pt and are still wrong.
    """
    a = [os.path.splitext(os.path.basename(p))[0] for p in audio_files]
    b = [os.path.splitext(os.path.basename(p))[0] for p in other_files]
    if a == b:
        return
    if len(a) != len(b):
        raise AssertionError(
            f'{base_dir}: {len(a)} audio file(s) but {len(b)} {kind} file(s). '
            f'They are paired by sorted index, so this cannot be resolved by '
            f'ignoring the extras -- regenerate, or fix the directory.')
    i = next(j for j in range(len(a)) if a[j] != b[j])
    raise AssertionError(
        f'{base_dir}: audio and {kind} stems diverge at index {i} '
        f'({a[i]!r} vs {b[i]!r}). They are paired by sorted index, so every '
        f'clip from here on would be trained against another clip\'s {kind}.')


class WaveParamDataset(Dataset):
    def __init__(self, base_dir, sample_rate=16000, length=4.0, params=True, f0=False):
        self.base_dir = base_dir
        self.audio_dir = os.path.join(base_dir, 'audio')
        self.raw_files = sorted(glob.glob(os.path.join(self.audio_dir, '*.wav')))
        print('loaded {0} files'.format(len(self.raw_files)))
        self.length = length
        self.sample_rate = sample_rate
        self.params = params
        self.f0 = f0
        if f0:
            self.f0_dir = os.path.join(base_dir, 'f0')
            assert os.path.exists(self.f0_dir)
            # all the f0 files should already be written
            # with the same name as the audio
            self.f0_files = sorted(glob.glob(os.path.join(self.f0_dir, '*.pt')))
            _check_pairing(self.raw_files, self.f0_files, base_dir, 'f0')
        if params:
            self.param_dir = os.path.join(base_dir, 'param')
            assert os.path.exists(self.param_dir)
            # all the files should already be written
            self.param_files = sorted(glob.glob(os.path.join(self.param_dir, '*.pt')))
            _check_pairing(self.raw_files, self.param_files, base_dir, 'param')
    
    def __getitem__(self, idx):
        raw_path = self.raw_files[idx]
        audio, _sr = librosa.load(raw_path, sr=self.sample_rate, duration=self.length)
        assert audio.shape[0] == self.length * self.sample_rate
        data = {'audio': audio}
        if self.f0:
            f0, periodicity = torch.load(self.f0_files[idx])
            f0_hz = process_f0(f0, periodicity)
            data['BFRQ'] = f0_hz.unsqueeze(-1)
        if self.params:
            params = torch.load(self.param_files[idx])
            data['params'] = params
        return data

    def __len__(self):
        return len(self.raw_files)

class IdOodDataModule(pl.LightningDataModule):
    def __init__(self, id_dir, ood_dir, train_type, batch_size, sample_rate=16000, length=4.0, num_workers=8, splits=[.8, .1, .1], f0=False):
        """ood_dir may be None, which drops the out-of-domain half entirely.

        The paper's procedures need it -- Real trains on it and every run logs
        val_ood -- but an experiment that supplies the oscillator pitches as
        conditioning cannot use it: the OOD set is loaded with params=False, so
        those batches carry no targets to take the pitches from. Rather than
        invent a fundamental for acoustic notes that have nothing to do with
        the comparison, drop the half. Validation then logs val_id only, and
        train.py monitors that instead.
        """
        super().__init__()
        self.id_dir = id_dir
        self.ood_dir = ood_dir
        self.use_ood = ood_dir is not None
        assert train_type in ['id', 'ood', 'mixed']
        assert self.use_ood or train_type == 'id', (
            'train_type={0!r} needs an ood_dir'.format(train_type))
        self.train_type = train_type
        self.splits = splits
        self.sr = sample_rate
        self.l = length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.f0 = f0
    
    def create_split(self, dataset):
        dset_l = len(dataset)
        split_sizes = [int(dset_l*self.splits[0]), int(dset_l*self.splits[1])]
        split_sizes.append(dset_l - split_sizes[0] - split_sizes[1])
        # should be seeded fine but probably better to split test set in some other way
        dset_train, dset_valid, dset_test = random_split(dataset, lengths=split_sizes)
        return {'train': dset_train, 'valid': dset_valid, 'test': dset_test}

    def setup(self, stage):
        id_dat = WaveParamDataset(self.id_dir, self.sr, self.l, True, self.f0)
        id_datasets = self.create_split(id_dat)
        if not self.use_ood:
            self.id_datasets = id_datasets
            self.ood_datasets = None
            return
        # ood should be the same size as in-domain
        ood_dat = WaveParamDataset(self.ood_dir, self.sr, self.l, False, self.f0)
        indices = np.random.choice(len(ood_dat), len(id_dat), replace=False)
        ood_dat = Subset(ood_dat, indices)
        ood_datasets = self.create_split(ood_dat)
        self.id_datasets = id_datasets
        self.ood_datasets = ood_datasets
        assert len(id_datasets['train']) == len(ood_datasets['train'])
        if self.train_type == 'mixed':
            dat_len = len(id_datasets['train'])
            indices = np.random.choice(dat_len, dat_len//2, replace=False)
            self.train_set = ConcatDataset([Subset(id_datasets['train'], indices), Subset(ood_datasets['train'], indices)])

    def train_dataloader(self):
        if self.train_type=='id':
            return DataLoader(self.id_datasets['train'], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)
        elif self.train_type=='ood':
            return DataLoader(self.ood_datasets['train'], batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=True)
        elif self.train_type=='mixed':
            id_indices = list(range(len(self.train_set)//2))
            ood_indices = list(range(len(self.train_set)//2, len(self.train_set)))
            id_samp = SubsetRandomSampler(id_indices)
            ood_samp = SubsetRandomSampler(ood_indices)
            id_batch_samp = BatchSampler(id_samp, batch_size=self.batch_size, drop_last=False)
            ood_batch_samp = BatchSampler(ood_samp, batch_size=self.batch_size, drop_last=False)
            generator = functools.partial(mix_iterable, id_batch_samp, ood_batch_samp)
            b_sampler = ReiteratableWrapper(generator, len(id_batch_samp)+len(ood_batch_samp))
            return DataLoader(self.train_set, batch_sampler=b_sampler, num_workers=self.num_workers)

    def _loaders(self, split):
        out = [DataLoader(self.id_datasets[split], batch_size=self.batch_size, num_workers=self.num_workers)]
        if self.use_ood:
            out.append(DataLoader(self.ood_datasets[split], batch_size=self.batch_size, num_workers=self.num_workers))
        return out

    def val_dataloader(self):
        return self._loaders("valid")

    def test_dataloader(self):
        return self._loaders("test")