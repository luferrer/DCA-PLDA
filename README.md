# DCA-PLDA

This repository implements the Discriminative Condition-Aware Backend described in the paper:

*L. Ferrer, M. McLaren, and N. Brümmer, ["A Speaker Verification Backend with Robust Performance across Conditions"](https://arxiv.org/pdf/2102.01760), in Computer Speech and Language, volume 71, 2021*

This backend has the same functional form as the usual probabilistic discriminant analysis (PLDA) backend which is commonly used for speaker verification, including the preprocessing stages. It also integrates the calibration stage as part of the backend, where the calibration parameters depend on an estimated condition for the signal. The condition is internally represented by a very low dimensional vector. See the paper for more details on the mathematical formulation of the backend.

We have found this system to provide great out-of-the-box performance across a very wide range of conditions, when training the backend with a variety of data including Voxceleb, SRE (from the NIST speaker recognition evaluations), Switchboard, Mixer 6, RATS and FVC Australian datasets, as described in the above paper. 

The code can also be used to train and evaluate a standard PLDA pipeline. Basically, the initial model before any training epochs is identical to a PLDA system, with an option for weighting the samples during training to compensate for imbalance across training domains.

Further, the current version of the code can also be used to do language detection. In this case, we have not yet explored the use of condition-awereness, but rather focused on a novel hierachical approach, which is described in the following paper:

*L. Ferrer, D. Castan, M. McLaren, and A. Lawson, ["A Hierarchical Model for Spoken Language Recognition"](https://arxiv.org/abs/2201.01364), arXiv:2201.01364, 2021*

Example scripts and configuration files to do both speaker verification and language detection are provided in the examples directory.

This code was written by Luciana Ferrer. We thank Niko Brummer for his help with the calibration code in the calibration.py file and for providing the code to do heavy-tail PLDA. The pre-computed embeddings provided to run the example were computed using SRI's software and infrastructure.

We will appreciate any feedback about the code or the approaches. Also, please let us know if you find bugs.


## How to install

1. Clone this repository:  

   ```git clone https://github.com/luferrer/DCA-PLDA.git```

2. Install the requirements:  
   
   ```pip install -r requirements.txt```

3. If you want to run the example code, download the pre-computed embeddings for the task you want to run from:  

   [```https://sftp.speech.sri.com/forms/DCA-DPLDA```](https://sftp.speech.sri.com/forms/DCA-DPLDA)
   
   Untar the file and move (or link) the resulting data/ dir inside the example dir for the task you want to run. 

4. You can then run the run_all script which runs several experiments using different configuration files and training sets. You can edit it to just try a single configuration, if you want. Please, see the top of that script for an explanation on what is run and where the output results end up. The run_all scripts will take a few hours to run (on a GPU) if all configurations are run. A RESULTS file is also provided for comparison. The run_all script should generate similar numbers to those in that file if all goes well.

## About the examples

The example dir contains two example recipes, one for speaker verification and one for language detection.

### Speaker Verification

The example provided with the repository includes the Voxceleb and FVC Australian subsets of the training data used in the paper, since the other datasets are not freely available. As such, the resulting system will only work well on conditions similar to those present in that data. For this reason, we test the resulting model on SITW and Voxceleb2 test dataset, which are very similar in nature to the Voxceleb data used for training. We also test on a set of FVC speakers which are held-out from training.

### Language Detection

The example uses the Voxlingua107 dataset which contains a large number of languages. 

## How to change the examples to use your own data and embeddings

The example scripts run using embeddings for each task extracted at **SRI International** using standard x-vector architectures. See the papers cited above for a description of the characteristics of the corresponding embedding extractors. Unfortunately, we are unable to release the embedding extractors, but you should be able to replace these embeddings with any type of speaker or language embeddings (eg, those that can be extracted with Kaldi).

The audio files corresponding to the databases used in the speaker verification example above can be obtained for free:

* Voxceleb1 and Voxceleb2 data can be obtained from http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
* FVC Australian dataset can be requested in http://databases.forensic-voice-comparison.net/
* The SITW database can be requested at sitw_poc@speech.sri.com

For the language detection example, the Voxlingua107 audio samples can be obtained from http://bark.phon.ioc.ee/voxlingua107/.

Once you have extracted embeddings for all that data using your own procedure, you can set up all the lists and embeddings in the same way and with the same format (hdf5 or npz in the case of embeddings) as in the example data dir for your task of interest and use the run_all script. 

The training script train.py takes several inputs:

* A metadata file with 5 columns:
	* file id: a name that identifies a certain speech segment. This name is then used to find the embeddings in the embeddings file described below.
   * speaker id: a unique name for the speaker in the file.
   * session id: a unique name for the session from which the file was extracted. This information is used to avoid creating same-session trials. The session should be the same the two sides of a telephone conversation, for a raw and a degraded file or for chunks extracted from that file. It is important to have this information be accurate since otherwise same-session trials may be created and those would degrade performance of the resulting model.
	* domain id: an identifier for the domain from which the file comes from. This can be the dataset name. This information is used to avoid having cross-domain impostor trials that would be extremely easy to classify and, if the config sets balance_batches_by_domain to True, to create the batches with equal number of samples per domain.
   * duration: this should be the duration of the speech within the file (as accurately extracted as possible, ie, probably an energy-based speech activity detector would not be enough) in seconds, with at least two digits after the decimal point.

* An optional metadata file which should be a subset of the file above to be used during initialization. While you can use the same set of files to initialize and run optimization, it is generally unnecessary to use all the available training data for initialization.

* A file with the embeddings for each of file id. The formats allowed by the code are npz and hdf5. In both cases, the code expects a dictionary with a data field containing the matrix of embeddings (one row per sample), and an ids field, containing the list of file ids, in the same order as the embeddings in the data field.

* A table with one line for each of the development sets to be used to report performance after each epoch, which is then used to select the best epoch, containing the following information
	* Dataset name
	* Path to the embeddings file for that dataset (in the same format as for training)
	* A file with the key for this dataset. The formats allowed for the key can be found in the Key class in dca_plda/scores.py. Basically, you can provide it in ascii format: enrollment_id test_id label (imp or tgt), or in h5 format as a dictionary with enroll_ids, test_ids and trial_mask with 1 for target, -1 for impostor and 0 for non-scored trials.
	* A file with two columns: enrollment_id (as in the key above) and sample_id as in the embeddings file. For multi-sample enrollment, list several lines with the same enrollment_id.
	* A file as the one above but for the test ids. This file will generally just have two columns identical to each other with the test sample ids 
	* A file with the duration for each sample used for enrollment and testing 

* A config file describing the architecture of the model. You can reuse the ones provided in the examples. 

* A config file for each stage of training. You can also probably reuse the ones provided in the examples.

The code will generate log files in the output dirs which you should inspect to see if things look as expected (ie, number of read samples, number of embeddings, etc).

When done, train.py will have generated a lot of models, one per epoch. The best model per run can be found in the folder corresponding to the last training stage, with the name best_*.

Then, the script eval.py takes the same info that goes in the table for the devsets above.

You can run train.py and eval.py with the —help flag to see what the expected inputs are.


## Note on scoring multi-sample enrollment models

For now, for speaker verification, the DCA-PLDA model only knows how to calibrate trials that are given by a comparison of two individual speech waveforms since that is the way we create trials during training. The code in this repo can still score trials with multi-file enrollment models, but it does it in a hacky way. Basically, it scores each enrollment sample against the test sample for the trial and then averages the scores. This works reasonably well but it is not ideal. A generalization to scoring multi-sample enrollment trials within the model is left as future work. 


