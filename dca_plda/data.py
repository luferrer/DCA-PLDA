import random
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from dca_plda import utils
from numpy.lib import recfunctions as rfn

class LabelledDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, emb_file, meta_file=None, meta_is_dur_only=False, device=None, cluster_ids=None, skip_missing=False):
        """
        Args:
            emb_file (string):  File with embeddings and sample ids in npz format
            meta_file (string): File with metadata for each id we want to store from the file above. 
                                Should contain: sample_id class_id session_id domain_id [cluster_id]
                                *  The class id is a unique string identifying the class of the sample (speaker, 
                                   language, etc)
                                *  The session id is a unique string identifying the recording session from which
                                   the audio sample was extracted (ie, if a waveform is split into chunks as a pre-
                                   processing step, the chunks all belong to the same session, or if several mics 
                                   were used to record a person speaking all these recordings belong to the same
                                   session). This information is used by the loader to avoid creating same-session
                                   trials which would mess up calibration.
                                *  The domain id is a unique string identifying the domain of the sample. The domain
                                   is assumed to be the dataset name. The domain id can be used to balance out
                                   the samples per batch by domain. 
                                   Further, only same-domain trials will be created since cross-domain trials would   
                                   not include target trials and would likely result in very easy impostor trials.
                                *  The cluster id is optional and only used for Hierarchical_DCA_PLDA.
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

        print("  Done loading embeddings")
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
            self.meta_raw = np.array(ids_all, np.dtype({'names': fields, 'formats': formats}))
        else:
            if meta_is_dur_only:
                fields, formats = zip(*[('sample_id', 'O'), ('duration', 'float32')])
            else:
                fields, formats = zip(*[('sample_id', 'O'), ('class_id', 'O'), ('session_id', 'O'), ('domain_id', 'O'), ('duration', 'float32')])
            self.meta_raw = np.loadtxt(meta_file, np.dtype({'names': fields, 'formats': formats}))

            if cluster_ids is not None:
                cids = np.array([cluster_ids[c] for c in self.meta_raw['class_id']])
                fields = fields + ('cluster_id',)
                formats = formats + ('O',)
                self.meta_raw = rfn.append_fields(self.meta_raw, '', cids, usemask=False).astype(np.dtype({'names': fields, 'formats': formats}))
        
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
            print("There are %d sample ids (out of %d in the metadata file %s) that are missing from the embeddings file %s"%(np.sum(keep_idxs==-1), len(self.meta_raw), meta_file, emb_file))
            if not skip_missing:
                raise Exception("Please, remove missing files from the metadata file and try again")

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
            idx_to_str = dict()
            for k, v in self.meta.items():
                id_map, meta[k] = utils.map_to_consecutive_ids(v[indx])
                if k in self.idx_to_str:
                    idx_to_str[k] = dict([(id_map[i],n) for i, n in self.idx_to_str[k].items() if i in id_map])
                    idx_to_str[k+"_inv"] = dict([(n,i) for i, n in idx_to_str[k].items()])
            meta = utils.AttrDict(meta)

            return embeddings, meta, idx_to_str

    def get_data(self):
        return self.embeddings

    def get_ids(self):
        return [self.idx_to_str['sample_id'][i] for i in self.meta['sample_id']]

    def get_durs(self):
        return self.meta['duration'] if 'duration' in self.meta else None

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):

        return {'emb': self.embeddings[idx].flatten(), 'class': self.meta_raw[idx]}



class TrialLoader(object):

    def __init__(self, dataset, metadata, metamaps, device, batch_size=256, num_batches=10, balance_method='same_num_classes_per_dom_then_same_num_samples_per_class', seed=0, num_samples_per_class=2, check_count_per_sess=True):
        """
        Args:
            embeddings: list of embeddings, size NxD, where N is the number of samples and D is the embedding dimension
            metadata: metadata dictionary for the embeddings (as generated by LabelledDataset.get_data_and_meta) above
            metamaps: dictionary with the string names for each id in metadata (also generated by LabDataset.get_data_and_meta)
            device: device where to move the created batches
            batch_size: size of batches to create
            num_batches: number of batches to create
            balance_method: which method to use for balancing the batches. There are four options, the first two are 
                recommended for speaker verification data, the last two for language detection.
                same_num_samples_per_class or none: each batch is composed of the same number of samples for each class
                        The number of samples for each class is determined by num_samples_per_class.
                same_num_classes_per_dom_then_same_num_samples_per_class: for each domain, the same number of classes
                        appear in each batch. For each class, the same number of samples are selected (as indicated by
                        num_samples_per_class)
                same_num_samples_per_class_and_dom: each batch is composed of the same number of samples for each class
                        and domain. Classes that appear in several domains, are overrepresented.
                same_num_samples_per_class_then_same_num_samples_per_dom: each batch is composed of the same number of
                        samples for each class, and within each class, each domain is represented by the same number of 
                        samples.
            seed: seed for randomization of lists
            num_samples_per_class: minimum number of samples per class per batch. If the available number of classes is small, classes
               might repeat more than this number within each batch in order to fill in the batch.
            subset: subset of samples in dataset to use for batch generation
        """

        print("Initializing trial loader (this might take a while for big datasets but this process saves time during training).")
        self.embeddings, self.metadata, self.metamaps = dataset, metadata, metamaps
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.rng = np.random.RandomState(seed)
        self.balance_method = balance_method

        if balance_method not in ['none', 'same_num_samples_per_class_and_dom', 'same_num_samples_per_class_then_same_num_samples_per_dom', 'same_num_classes_per_dom_then_same_num_samples_per_class']:
            raise Exception("Balance method %s not implemented"%balance_method)

        if batch_size%num_samples_per_class != 0:
            raise Exception("Batch size has to be a multiple of the number of samples per class requested (%d)"%num_samples_per_class)

        if balance_method == 'same_num_samples_per_class' or balance_method == 'none':
            # Add a dummy domain column in self.metadata with all domains = 0 so that the
            # domain is ignored for trial selection
            dom_col = 'dummy_domain_id'
            self.metadata[dom_col] = np.zeros_like(self.metadata['domain_id'])
            self.metamaps[dom_col] = {0: 'all'}
        else:
            # All other balancing methods use the domain information
            dom_col = 'domain_id'
        
        # The following dictionaries are made to expedite the generation of batches
        self.classes_for_dom,            self.classi  = self._init_index_and_list([dom_col], 'class_id')
        self.sessions_for_class_dom,     self.sessi   = self._init_index_and_list(['class_id',dom_col], 'session_id', min_len=2 if check_count_per_sess else 1)
        self.samples_for_sess_class_dom, self.samplei = self._init_index_and_list(['session_id','class_id',dom_col], 'sample_id')
        self.doms_for_classes,           self.domi    = self._init_index_and_list(['class_id'], dom_col)

        self.domains = list(self.classi.keys())
        self.sel_num_classes_per_dom = dict()
        self.sel_num_samples_per_class = dict()
        for dom in self.domains:
            dom_name = self.metamaps[dom_col][dom[0]]
            if balance_method == 'same_num_samples_per_class_and_dom':
                # In this case, each domain will have a number of selected classes proportional to
                # the number of classes that domain has. 
                nclasses = len(self.classes_for_dom[dom])
                self.sel_num_classes_per_dom[dom] = nclasses * batch_size / len(self.sessions_for_class_dom)
                # This is not used because the sel_num_classes_per_dom is in number of samples
                self.sel_num_samples_per_class[dom] = 1
                print("Selecting batches with %4d samples for domain %s divided equally for each of the %d classes (~ %.2f samples per class)"%
                    (self.sel_num_classes_per_dom[dom], dom_name, nclasses, self.sel_num_classes_per_dom[dom] / nclasses))

            elif balance_method == 'same_num_samples_per_class_then_same_num_samples_per_dom':
                class_count = len(np.unique(np.concatenate(list(self.classes_for_dom.values()))))
                self.sel_num_classes_per_dom[dom] = len(self.classes_for_dom[dom]) 
                # Number of samples for each class in the domain is such that, over all domains
                # each class gets the same number of samples 
                num_doms_per_class = dict([(c, len(doms)) for c, doms in self.doms_for_classes.items()])
                self.sel_num_samples_per_class[dom] = dict([((c,)+dom,  batch_size / class_count / num_doms_per_class[(c,)]) for c in self.classes_for_dom[dom]])

                print("Selecting batches with %d classes for domain %s, with a number of samples for each class proportional to the inverse of the number of domains available for the class"%
                    (self.sel_num_classes_per_dom[dom], dom_name))

            elif balance_method in ['none', 'same_num_samples_per_class', 'same_num_classes_per_dom_then_same_num_samples_per_class']:
                # In this case, each domain has the same number of classes. For none and same_num_samples_per_class,
                # the domain has been set to a dummy value, so this still applies.
                self.sel_num_samples_per_class[dom] = num_samples_per_class
                sel_num_classes_per_batch = int(batch_size/num_samples_per_class)
                self.sel_num_classes_per_dom[dom] = int(np.ceil(sel_num_classes_per_batch/len(self.domains)))
                print("Selecting %d classes for domain %s, and %d samples per selected class"%(self.sel_num_classes_per_dom[dom], dom_name, self.sel_num_samples_per_class[dom]))
            else:
                raise Exception("Balance method %s not implemented"%balance_method)

        self.sample_to_idx = dict(np.c_[self.metadata['sample_id'], np.arange(len(self.metadata['sample_id']))])

        self.device = device
        print("Done Initializing trial loader")
        print("Will create %d batches of size %d per epoch using %s samples"%(num_batches, batch_size, self.embeddings.shape[0]), flush=True)

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

        # Yield num_batches batches.
        # Keep track of how many rounds through the classes we do for each domain, just for
        # logging and debugging purposes
        num_rewinds = dict([(d, 0) for d in self.domains])

        for batch_num in np.arange(self.num_batches):

            # Shuffle the domains before generating each batch because if the batch size is not multiple of the
            # number of domains, the last domain will have fewer samples than the others
            self.rng.shuffle(self.domains)
        
            sel_idxs = []

            while len(sel_idxs) < self.batch_size:
                # Normally this should not need to loop more than once, but sometimes,
                # due to rounding issues, we need to restart the loop to fill up the batch at the end.
                
                for dom in self.domains:
                    for numcl in np.arange(self._random_round(self.sel_num_classes_per_dom[dom])):
                        if len(sel_idxs) > self.batch_size:
                            break

                        # Select a class from this domain
                        class_dom, r = self._find_value(self.classi, self.classes_for_dom, dom)
                        num_rewinds[dom] += r

                        if self.balance_method == 'same_num_samples_per_class_then_same_num_samples_per_dom':
                            sel_num_samples = self.sel_num_samples_per_class[dom][class_dom]
                        else:
                            sel_num_samples = self.sel_num_samples_per_class[dom]

                        # For the selected class, select num_samples_per_class samples all from different sessions
                        for numsample in np.arange(self._random_round(sel_num_samples)):
                            # Only shuffle at rewind for the first sample, else we risk getting the same session again
                            sess_class_dom, _  = self._find_value(self.sessi, self.sessions_for_class_dom, class_dom, shuffle_at_rewind=(numsample==0))
                            # For the selected session, select one sample 
                            sample_sess_class_dom, _ = self._find_value(self.samplei, self.samples_for_sess_class_dom, sess_class_dom)
                            sel_idxs += [self.sample_to_idx[sample_sess_class_dom[0]]]


            sel_idxs = np.array(sel_idxs)[:self.batch_size]

            metadata_for_batch = dict([(f, self._np_to_torch(v[sel_idxs])) for f,v in self.metadata.items()])

            assert len(sel_idxs) == self.batch_size
            yield self._np_to_torch(self.embeddings[sel_idxs]), metadata_for_batch

        if self.balance_method != 'none':
            print("Created %d batches with the following number of resets of the class lists per domain:"%self.num_batches)
            for dom in self.domains:
                print("  dom %s: %d resets, %d classes"%(self.metamaps['domain_id'][dom[0]], num_rewinds[dom], len(self.classes_for_dom[dom])))
        else:
            print("Created %d batches with %d resets, %d classes."%(self.num_batches, num_rewinds[self.domains[0]], len(self.classes_for_dom[dom])))

    @staticmethod
    def _random_round(x):
        # Round to an int by doing floor or ceil depending on the decimal part
        round_func = np.ceil if random.random() < x-int(x) else np.floor
        return int(round_func(x))


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








