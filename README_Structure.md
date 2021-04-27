# Further Details
Some more details about the project (and how to use it without docker).

## Structure
- The defender directory is the root directory of our python project.
If you import our defense into an IDE, for instance, make sure that you
set *defender* as root directory.
- Within *defender*, you find the following directories:
  - *models*: Here, we just have an ensemble.yml that defines the relative
  paths to the models.
  - *learning*: This directory contains the various implemented defenses.
    - Please note that the semantic gap detectors (called predators in the
      project) are within the *statefuldefense* directory.
      - **However, The semantic gap detectors are not part of the stateful
      defense** which is located in the *statefuldefense/stateful* directory!
      - Our whitepaper shows an overview of the defense in Figure 4 and Table I.
    - The directory *statefuldefense/stateful* contains the stateful defense.
    - The directory *statefuldefense/predators* contains the semantic gap detectors.
    - The directory *expert* contains the signature-based model.
    - The directory *emberboost* contains the Ember-related models.
  - *features*: This directory contains some scripts to train the skipgrams model
  as well as to extract skipgrams.

## Without docker
- In principle, you can use our defense without docker. To this end, please
check how the defense is created and used in the ```apps.py``` and ```__main__.py```
in the *defender* directory.
- You can test the individual components as well without docker. To this end,
please look at
  - ```defender/learning/ensemble/scripts/test_ensemble.py```,
  - ```defender/learning/statefuldefense/stateful/test_stateful.py```,
  - ```defender/learning/statefuldefense/predators/scan_sections.py```.
  - Adjust the paths and run the scripts directly.
  - Note that for the signature-based model and skipgram model, there is no
  stand-alone script to apply them, yet. In this case, look
  at ```apps.py``` and ```__main__.py``` in the *defender* directory to find
  out how to apply them.


## Training
The repository contains the scripts to train own models. This might be a good
choice if you have a large dataset. While our defense could make use of the
pre-extracted Ember features, we only had 22,000 benign and malicious PE files
that we have used for the skipgrams and signature-based model.
So we expect a higher performance of these defenses if also trained
with a larger dataset. If you want to train your own models, please contact
us and we will try to add the respective documentation as well.

## Issues
- The ngrams/skipgrams detection is currently done in python. This is slow.
  We've also developed a cython version to reduce the classification time (which
  can be provided as well).
