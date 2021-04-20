import random
import torch
from torch.utils.data import Dataset
import numpy as np
from IPython import embed
import utils 
import h5py

class SpeakerDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, emb_file, meta_file=None, meta_is_dur_only=False, device=None):
        """
        Args:
            emb_file (string):  File with embeddings and sample ids in npz format
            meta_file (string): File with metadata for each id we want to store from the file above. 
                                Should contain: sample_id speaker_id session_id domain_id
                                *  The speaker id is a unique string identifying the speaker
                                *  The session id is a unique string identifying the recording session from which
                                   the audio sample was extracted (ie, if a waveform is split into chunks as a pre-
                                   processing step, the chunks all belong to the same session, or if several mics 
                                   were used to record a person speaking all these recordings belong to the same
                                   session). This information is used by the loader to avoid creating same-session
                                   trials which would mess up calibration.
                                *  The domain id is a unique string identifying the domain. Domains should correspond 
                                   to disjoint speaker sets. This information is also used by the loader. Only same-
                                   domain trials are created since cross-domain trials would never include target 
                                   trials and would likely result in very easy impostor trials.
        """
        if meta_file is not None:
            print("Loading data from %s\n  with metadata file %s"%(emb_file,meta_file))
        else:   
            print("Loading data from %s without metadata"%emb_file)

        if emb_file.endswith(".npz"):
            data_dict = np.load(emb_file)
        elif emb_file.endswith(".h5"):
            with h5py.File(emb_file, 'r') as f:
                data_dict = {'ids': f['ids'][()], 'data': f['data'][()]}
        else:
            raise Exception("Unrecognized format for embeddings file %s"%emb_file)

        embeddings_all = data_dict['data']
        if type(data_dict['ids'][0]) == np.bytes_:
            ids_all = [i.decode('UTF-8') for i in data_dict['ids']]
        elif type(data_dict['ids'][0]) == np.str_:
            ids_all = data_dict['ids']
        else:
            raise Exception("Bad format for ids in embeddings file %s (should be strings)"%emb_file)

        self.idx_to_str = dict()        
        self.meta = dict()
            
        if meta_file is None:
            fields = ('sample_id',)
            formats = ('O',)
            self.meta_raw = np.array(ids_all, np.dtype({'names': fields, 'formats': ('O',)}))
        else:
            if meta_is_dur_only:
                fields, formats = zip(*[('sample_id', 'O'), ('duration', 'float32')])
            else:
                fields, formats = zip(*[('sample_id', 'O'), ('speaker_id', 'O'), ('session_id', 'O'), ('domain_id', 'O'), ('duration', 'float32')])
            self.meta_raw = np.loadtxt(meta_file, np.dtype({'names': fields, 'formats': formats}))
        
        # Convert the metadata strings into indices
        print("  Converting metadata strings into indices")
        for field, fmt in zip(fields, formats):
            if fmt == 'O':
                # Convert all the string fields into indices
                names, nmap = np.unique(self.meta_raw[field], return_inverse=True)
                if field == 'sample_id' and len(names)!=len(self.meta_raw):
                    raise Exception("Metadata file %s has repeated sample ids"%meta_file)

                # Index to string and string to index maps                
                self.idx_to_str[field]        = dict(zip(np.arange(len(names)),names))
                self.idx_to_str[field+"_inv"] = dict(zip(names, np.arange(len(names))))
                self.meta[field] = np.array(nmap, dtype=np.int32)
            else:
                self.meta[field] = self.meta_raw[field]

        self.meta = utils.AttrDict(self.meta)

        # Subset the embeddings to only those in the metadata file
        name_to_idx = dict(zip(ids_all, np.arange(len(ids_all))))
        keep_idxs = np.array([name_to_idx.get(n, -1) for n in self.meta_raw['sample_id']])
        if np.any(keep_idxs == -1):
            raise Exception("There are %d sample ids (out of %d in the metadata file %s) that are missing from the embeddings file %s.\nPlease, remove those files from the metadata file and try again"%
                (np.sum(keep_idxs==-1), len(self.meta_raw), meta_file, emb_file))
        self.embeddings = embeddings_all[keep_idxs]
        self.ids = np.array(ids_all)[keep_idxs]

        if device is not None:
            # Move the embeddings and the durations to the device
            self.embeddings = utils.np_to_torch(self.embeddings, device)
            if 'duration' in self.meta:
                self.meta['duration'] = utils.np_to_torch(self.meta['duration'], device)

        print("Done. Loaded %d embeddings from %s"%(len(self.embeddings), emb_file), flush=True)



    def get_data_and_meta(self, subset=None):
        if subset is None:
            return self.embeddings, self.meta, self.idx_to_str

        else:
            (found, indx) = utils.ismember(subset, list(self.ids))
            if np.sum(found) != len(subset):
                print("Some of the files (%d) in the subset are missing. Ignoring them"%(len(subset)-np.sum(found)))
            indx = indx[found]

            # Subset the embeddings and the metadata to the required samples
            embeddings = self.embeddings[indx]
            meta = dict()
            for k, v in self.meta.items():
                meta[k] = v[indx]
            meta = utils.AttrDict(meta)

            return embeddings, meta, self.idx_to_str

    def get_data(self):
        return self.embeddings

    def get_ids(self):
        return [self.idx_to_str['sample_id'][i] for i in self.meta['sample_id']]

    def get_durs(self):
        return self.meta['duration'] if 'duration' in self.meta else None

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):

        return {'emb': self.embeddings[idx].flatten(), 'speaker': self.meta_raw[idx]}



class TrialLoader(object):

    def __init__(self, dataset, device, batch_size=256, num_batches=10, balance_by_domain=True, seed=0, num_samples_per_spk=2, subset=None):
        """
        Args:
            dataset: an object of class SpeakerDataset 
        """
        print("Initializing trial loader (this might take a while for big datasets but this process saves time during training).")
        self.embeddings, self.metadata, self.metamaps = dataset.get_data_and_meta(subset=subset)
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.rng = np.random.RandomState(seed)
        self.balance_by_domain = balance_by_domain
        self.num_samples_per_spk = num_samples_per_spk

        if batch_size%num_samples_per_spk != 0:
            raise Exception("Batch size has to be a multiple of the number of samples per speaker requested (%d)"%num_samples_per_spk)

        if balance_by_domain:
            dom_col = 'domain_id'
        else:
            # Add a dummy domain column in self.metadata with all domains = 0 so that the
            # domain is ignored for trial selection
            dom_col = 'dummy_domain_id'
            self.metadata[dom_col] = np.zeros_like(self.metadata['domain_id'])
        
        # The following dictionaries are made to expedite the generation of batches
        self.spkrs_for_dom,            self.spki    = self._init_index_and_list([dom_col], 'speaker_id')
        self.sess_for_spk_dom,         self.sessi   = self._init_index_and_list(['speaker_id',dom_col], 'session_id', min_len=2)
        self.samples_for_sess_spk_dom, self.samplei = self._init_index_and_list(['session_id','speaker_id',dom_col], 'sample_id')

        self.domains = list(self.spki.keys())
        
        self.num_spkrs_per_dom = dict()
        for dom in self.domains:
            if balance_by_domain == 'num_classes_per_dom_prop_to_its_total_num_classes':
                # In this case, each domain will have a number of samples that is proporcional to
                # the number of classes that domain has
                self.num_spkrs_per_dom[dom] = len(self.spkrs_for_dom[dom])
            elif balance_by_domain is True or balance_by_domain == 'same_num_classes_per_dom':
                # In this case, each domain has the same number of speakers
                self.num_spkrs_per_batch    = int(batch_size/num_samples_per_spk)
                self.num_spkrs_per_dom[dom] = int(np.ceil(self.num_spkrs_per_batch/len(self.domains)))
            elif balance_by_domain is not False or balance_by_domain is not None:
                raise Exception("Option for balance_by_domain (%s) not implemented"%balance_by_domain)

        self.sample_to_idx = dict(np.c_[self.metadata['sample_id'], np.arange(len(self.metadata['sample_id']))])

        self.device = device
        print("Done Initializing trial loader")
        print("Will create %d batches per epoch using %s samples"%(num_batches, self.embeddings.shape[0]), flush=True)

    def _init_index_and_list(self, key_fields, list_field, min_len = 1):
        list_dict = dict()
        index_dict = dict()
        print("  Creating dictionary with a list of %s for each %s"%(list_field, str(key_fields)))
       
        # Create a new array with the concatenation of the values for the fields in key_fields
        keys = np.array([self.metadata[f] for f in key_fields]).T
        values = self.metadata[list_field]
        for i, k in enumerate(keys):
            kl = tuple(k)
            if kl in list_dict:
                list_dict[kl].append(values[i])
            else:
                list_dict[kl] = [values[i]]

        for kl in list_dict.keys():
            list_dict[kl] = list(set(list_dict[kl]))
            index_dict[kl] = 0
            if len(list_dict[kl]) < min_len:
                raise Exception("Not enough %ss for some combination of %s (there should be at least %d %s per %s)"%
                    (list_field, str(key_fields), min_len, list_field, str(key_fields)))
            self.rng.shuffle(list_dict[kl])

        return list_dict, index_dict

    def __iter__(self):

        # Yield num_batches containing batch_size/num_samples_per_spk speakers each, with
        # num_samples_per_spk samples per speaker, each sample from a different session (if possible).

        # Keep track of how many rounds through the speakers we do for each domain, just for
        # logging and debugging purposes
        num_rewinds = dict([(d, 0) for d in self.domains])

        for batch_num in np.arange(self.num_batches):

            # Shuffle the domains before generating each batch because if the batch size is not multiple of the
            # number of domains, the last domain will have fewer samples than the others
            self.rng.shuffle(self.domains)
        
            sel_idxs = []

            for dom in self.domains:
                for numsp in np.arange(self.num_spkrs_per_dom[dom]):

                    # Select a speaker from this domain
                    spk_dom, r = self._find_value(self.spki, self.spkrs_for_dom, dom)
                    num_rewinds[dom] += r

                    # For the selected speaker, select num_samples_per_spk samples all from different sessions
                    for numsample in np.arange(self.num_samples_per_spk):
                        # Only shuffle at rewind for the first sample, else we risk getting the same session again
                        sess_spk_dom, _  = self._find_value(self.sessi, self.sess_for_spk_dom, spk_dom, shuffle_at_rewind=(numsample==0))
                        # For the selected session, select one sample 
                        sample_sess_spk_dom, _ = self._find_value(self.samplei, self.samples_for_sess_spk_dom, sess_spk_dom)
                        sel_idxs += [self.sample_to_idx[sample_sess_spk_dom[0]]]

            sel_idxs = np.array(sel_idxs)[:self.batch_size]

            metadata_for_batch = dict([(f, self._np_to_torch(v[sel_idxs])) for f,v in self.metadata.items()])

            yield self._np_to_torch(self.embeddings[sel_idxs]), metadata_for_batch

        if self.balance_by_domain:
            print("Dumped %d batches with the following number of resets of the speaker lists per domain:"%self.num_batches)
            for dom in self.domains:
                print("  dom %s: %d resets, %d speakers"%(self.metamaps['domain_id'][dom[0]], num_rewinds[dom], len(self.spkrs_for_dom[dom])))
        else:
            print("Dumped %d batches with %d resets, %d speakers."%(self.num_batches, num_rewinds[self.domains[0]], len(self.spkrs_for_dom[dom])))


    def _np_to_torch(self, x):
        return utils.np_to_torch(x, self.device)


    def _find_value(self, index_dict, list_dict, key, shuffle_at_rewind=True):
        # Return the value in list_dict corresponding to index_dict unless
        # that index is bigger than the length of the list. In that case,
        # reshuffle the list and restart the index.
        # Note that this method modifies the values in the input dictionaries.
        rewound = 0
        if index_dict[key] >= len(list_dict[key]):
            if shuffle_at_rewind:
                self.rng.shuffle(list_dict[key])
            index_dict[key] = 0
            rewound = 1
        
        val = list_dict[key][index_dict[key]]
        index_dict[key] += 1

        return (val,)+key, rewound








