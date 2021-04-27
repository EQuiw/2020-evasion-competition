# Against All Odds: Winning the Defense Challenge in an Evasion Competition with Diversification

This repository contains the defense *PEberus** that got the first place in the
[Machine Learning Security Evasion Competition 2020](https://mlsec.io/),
resisting a variety of attacks from independent attackers.

You can find the whitepaper that outlines our defense here:

---

[
Erwin Quiring, Lukas Pirch, Michael Reimsbach, Daniel Arp, Konrad Rieck. *Against All Odds: Winning the Defense Challenge in an Evasion Competition with Diversification*, 2020
](https://arxiv.org/abs/2010.09569)

---

<sub>*PEberus --- Named after the mythic three-headed dog Cerberus, recently seen
as guard in Hogwarts.</sub>

## Bibtex
If you are using our defense, please cite our whitepaper.
You may use the following BibTex entry:
```
@misc{QuiPirReiArpRie20,
        title={Against All Odds: Winning the Defense Challenge in an Evasion Competition with Diversification},
        author={Erwin Quiring and Lukas Pirch and Michael Reimsbach and Daniel Arp and Konrad Rieck},
        year={2020},
        eprint={2010.09569},
        archivePrefix={arXiv},
        primaryClass={cs.CR}
  }
```

## Summary
Machine learning-based systems for malware detection operate in a
hostile environment. Consequently, adversaries will also target the learning
system and use evasion attacks / adversarial examples to bypass the detection
of malware. The competition gives us the unique opportunity to examine such
attacks under a realistic scenario.

<p align="center">
<img src="./2020-evasion.jpg" width="400" alt="Implemented defenses" />
</p>

Our solution is based on diversification and consists of three main
concepts:
- Various heuristics address the semantic gap. This gap between
the semantics of a PE program and its feature representation
allows relatively simple functionality-preserving attacks.
- Multiple classification models use distinct feature sets to
classify malware reliably while considering targeted attacks against
the model.
- A stateful defense finally detects iterative attacks that
exploit the API access to the classifier.

The developed defense highlights that existing machine learning methods
can be hardened against attacks by thoroughly analyzing the attack surface and
implementing concepts from adversarial learning.
Although our solution fends off the majority of attacks in the
competition, it is limited to static analysis, and thus a few attacks
based on obfuscation succeeded.

Our defense can serve as an additional baseline in the future to strengthen the
research on secure learning.

## Getting Started
### Installation
First, make sure you have installed Docker.
Then, get the repository:
```
git clone https://github.com/EQuiw/2020-evasion-competition.git
cd 2020-evasion-competition
```

### Deployment
Build docker image (ensure you are still in the repo directory):
```
docker build -t adv-challenge .
```
Start container (interactive mode),
(CPU and memory settings as given in the competition):
```
docker run -it -p 8080:8080 --memory=1.5g --cpus=1 adv-challenge /bin/bash
python __main__.py  # start flask app
```

### Testing
Open a second shell, as the container has been started interactively.

Create a python environment (to keep your system clean). The following commands
assume that you have installed Anaconda.
```
conda create --name scaling-attack python=3.7
conda activate scaling-attack
```
Next, go to the repository (replace the path) and activate conda environment:
```
cd <PATH-TO-REPO>/2020-evasion-challenge
```
Install python packages:
```
pip install -r requirements.txt
```
#### Test 1
- Our first simple test case uses the putty executable as a test object. Please download ```putty.exe```
from the [official website](https://www.putty.org/). Alternatively, you can use:
```
wget -O ./test/test_objects/putty.exe https://the.earth.li/~sgtatham/putty/latest/w64/putty.exe
```
The download link may change over time! Then simply replace it.
After that, start the tests.
```
python test/test_post_request.py
```

#### Test 2
- To test the defense against malicious files, please download ```MLSEC_2019_samples_and_variants.zip```.
You can find all instructions on
the [competition website](https://github.com/Azure/2020-machine-learning-security-evasion-competition/tree/master/defender).
Assuming you are working on a secure system, and you have unpacked the zip file,
you can run ```test_dataset.py```that runs over all PE files. Please
adjust the path to your directory in the following command:
```
python test/test_dataset.py --datasetpath /<TODO-PATH-TO>/MLSEC_2019_samples_and_variants/ --verbose
```
If you are only interested in the final outcome, remove the ```--verbose``` command.
The final classification accuracy should be 95.11784511784511%.

### Additional Dependencies

- [GNU Parallel](https://www.gnu.org/software/parallel/): Used in our shell scripts to parallelize the feature extraction.
Make sure to install it in the host environment if you would like to make use of it.

### More information
- You find the [Competition README here](./README_Competition.md)
- You find more details about the structure of the project in this [README](./README_Structure.md)
