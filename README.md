# DCA-PLDA

This repository implements the Discriminative Condition-Aware Backend described in the paper:

*Ferrer, L. and McLaren, M., ["A Speaker Verification Backend for Improved Calibration Performance across Varying Conditions"](https://www.isca-speech.org/archive/Odyssey_2020/pdfs/5.pdf), in Proc. of Odyssey 2020, Tokyo, Japan, Nov. 2020*

and in

*Ferrer, L. and McLaren, M. and Brümmer N., ["A Speaker Verification Backend with Robust Performance across Conditions"](https://arxiv.org/pdf/2102.01760), arXiv:2102.01760*

This backend has the same functional form as the usual probabilistic discriminant analysis (PLDA) backend which is commonly used for speaker verification, including the preprocessing stages. It also integrates the calibration stage as part of the backend, where the calibration parameters depend on an estimated condition for the signal. The condition is internally represented by a very low dimensional vector. See the paper for more details on the mathematical formulation of the backend.

We have found this system to provide great out-of-the-box performance across a very wide range of conditions, when training the backend with a variety of data including Voxceleb, SRE (from the NIST speaker recognition evaluations), Switchboard, Mixer 6, RATS and FVC Australian datasets, as described in the above paper. 

The code can also be used to train and evaluate a standard PLDA pipeline. Basically, the initial model before any training epochs is identical to a PLDA system, with an option for weighting the samples during training to compensate for imbalance across training domains.

This code was written by Luciana Ferrer. The pre-computed embeddings provided to run the example were computed using SRI's software and infrastructure. We thank Niko Brummer for his help with the calibration code in the calibration.py file.

We will appreciate any feedback about the code or the approaches. Also, please let us know if you find bugs.


## How to install

1. Clone this repository:  

   ```git clone https://github.com/luferrer/DCA-PLDA.git```

2. Install the requirements:  
   
   ```pip install -r requirements.txt```

3. If you want to run the example code, download the pre-computed embeddings from:  

   ```https://sftp.speech.sri.com/forms/DCA-DPLDA```   
   
   Untar the file and move (or link) the resulting data/ dir inside the example dir. Note that, to run the latest version of the example code you need the latest embeddings from that page. If you downloaded them before Feb 1st 2021, please download them again (they should say v2 in the file name). 

4. You can then run the run_all script which runs several experiments using different configuration files and training sets. You can edit it to just try a single configuration, if you want. Please, see the top of that script for an explanation on what is run and where the output results end up. The script will take a few hours to run (on a GPU) if all configurations are run.

## About the example

The example provided with the repository includes the Voxceleb and FVC Australian subsets of the training data used in the paper, since the other datasets are not freely available. As such, the resulting system will only work well on conditions similar to those present in that data. For this reason, we test the resulting model on SITW and Voxceleb2 test dataset, which are very similar in nature to the Voxceleb data used for training. We also test on a set of FVC speakers which are held-out from training.

## How to change the example to use your own data and embeddings

The example code runs using speaker embeddings extracted at **SRI International** using a standard x-vector architecture. See the paper cited above for a description of the characteristics of the embedding extractor. Unfortunately, we are unable to release the embedding extractor, but you should be able to replace these embeddings with any type of speaker embedding (eg, those that can be extracted with Kaldi).

The audio files corresponding to the databases used in the example above can be obtained for free:

* Voxceleb1 and Voxceleb2 data can be obtained from http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
* FVC Australian dataset can be requested in http://databases.forensic-voice-comparison.net/
* The SITW database can be requested at sitw_poc@speech.sri.com

Once you have extracted embeddings for all that data using your own procedure, you can set up all the lists and embeddings in the same way and with the same format (hdf5 or npz in the case of embeddings) as in the example data dir and use the run_all script. 


## Note on scoring multi-sample enrollment models

For now, the DCA-PLDA model only knows how to calibrate trials that are given by a comparison of two individual speech waveforms since that is the way we create trials during training. 
The code in this repo can still score trials with multi-file enrollment models, but it does it in a hacky way. Basically, it scores each enrollment sample against the test sample for the trial and then averages the scores. This works reasonably well but it is not ideal. A generalization to scoring multi-sample enrollment trials within the model is left as future work. 


