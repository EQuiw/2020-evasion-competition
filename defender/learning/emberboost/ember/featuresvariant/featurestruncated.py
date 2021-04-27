#!/usr/bin/python
''' Extracts some basic features from PE files. Based on
https://github.com/elastic/ember/blob/master/ember/features.py

NOTE: In this file, we have adjusted the feature extraction. See below!
I recommend making a diff to default ember features.py file.
'''

import re
import lief
import hashlib
import numpy as np
import typing
from ..features import FeatureType, SectionInfo, ImportsInfo, ExportsInfo, GeneralFileInfo, HeaderFileInfo

LIEF_MAJOR, LIEF_MINOR, _ = lief.__version__.split('.')
LIEF_EXPORT_OBJECT = int(LIEF_MAJOR) > 0 or ( int(LIEF_MAJOR)==0 and int(LIEF_MINOR) >= 10 )


class StringExtractor(FeatureType):
    ''' Extracts strings from raw byte stream '''

    name = 'strings'
    dim = 1 + 1 + 1 + 96 + 1 + 1 + 1 + 1 + 1 - 96

    def __init__(self):
        super(FeatureType, self).__init__()
        # all consecutive runs of 0x20 - 0x7f that are 5+ characters
        self._allstrings = re.compile(b'[\x20-\x7f]{5,}')
        # occurances of the string 'C:\'.  Not actually extracting the path
        self._paths = re.compile(b'c:\\\\', re.IGNORECASE)
        # occurances of http:// or https://.  Not actually extracting the URLs
        self._urls = re.compile(b'https?://', re.IGNORECASE)
        # occurances of the string prefix HKEY_.  No actually extracting registry names
        self._registry = re.compile(b'HKEY_')
        # crude evidence of an MZ header (dropper?) somewhere in the byte stream
        self._mz = re.compile(b'MZ')

    def raw_features(self, bytez, lief_binary):
        allstrings = self._allstrings.findall(bytez)
        if allstrings:
            # statistics about strings:
            string_lengths = [len(s) for s in allstrings]
            avlength = sum(string_lengths) / len(string_lengths)
            # map printable characters 0x20 - 0x7f to an int array consisting of 0-95, inclusive
            as_shifted_string = [b - ord(b'\x20') for b in b''.join(allstrings)]
            c = np.bincount(as_shifted_string, minlength=96)  # histogram count
            # distribution of characters in printable strings
            csum = c.sum()
            p = c.astype(np.float32) / csum
            wh = np.where(c)[0]
            H = np.sum(-p[wh] * np.log2(p[wh]))  # entropy
        else:
            avlength = 0
            c = np.zeros((96,), dtype=np.float32)
            H = 0
            csum = 0

        return {
            'numstrings': len(allstrings),
            'avlength': avlength,
            'printabledist': c.tolist(),  # store non-normalized histogram
            'printables': int(csum),
            'entropy': float(H),
            'paths': len(self._paths.findall(bytez)),
            'urls': len(self._urls.findall(bytez)),
            'registry': len(self._registry.findall(bytez)),
            'MZ': len(self._mz.findall(bytez))
        }

    def process_raw_features(self, raw_obj):
        hist_divisor = float(raw_obj['printables']) if raw_obj['printables'] > 0 else 1.0
        return np.hstack([
            raw_obj['numstrings'], raw_obj['avlength'], raw_obj['printables'],
            raw_obj['entropy'], raw_obj['paths'], raw_obj['urls'],
            raw_obj['registry'], raw_obj['MZ']
        ]).astype(np.float32)

    def feature_names(self):
        a1 = [self.name + "_" + x for x in ['numstrings', 'avlength', 'printables']]
        a2 = []
        a3 = [self.name + "_" + x for x in ['entropy', 'paths', 'urls', 'registry', 'MZ']]
        out = a1 + a2 + a3
        assert len(out) == self.dim
        return out


class PEFeatureExtractorTruncated(object):
    ''' Extract useful features from a PE file, and return as a vector of fixed size.
    ByteHistogram, ByteEntropyHistogram, DataDirectories removed + StringExtractor modified.
    '''

    def __init__(self, feature_version=2, include_header=False):

        self.features = [
            #ByteHistogram(),
            #ByteEntropyHistogram(),
            StringExtractor(),
            GeneralFileInfo(),
        ]

        if include_header is True:
            self.features.append(HeaderFileInfo())

        self.features.extend([
            SectionInfo(),
            ImportsInfo(),
            ExportsInfo()
        ])

        if feature_version == 1:
            if not lief.__version__.startswith("0.8.3"):
                print(f"WARNING: EMBER feature version 1 were computed using lief version 0.8.3-18d5b75")
                print(f"WARNING:   lief version {lief.__version__} found instead. There may be slight inconsistencies")
                print(f"WARNING:   in the feature calculations.")
        elif feature_version == 2:
            #self.features.append(DataDirectories())
            if not lief.__version__.startswith("0.9.0") and not lief.__version__.startswith("0.10.1"):
                print(f"WARNING: EMBER feature version 2 were computed using lief version 0.9.0-")
                print(f"WARNING:   lief version {lief.__version__} found instead. There may be slight inconsistencies")
                print(f"WARNING:   in the feature calculations.")
        else:
            raise Exception(f"EMBER feature version must be 1 or 2. Not {feature_version}")
        self.dim = sum([fe.dim for fe in self.features])

    def raw_features(self, bytez, pe_binary: typing.Optional[lief.PE.Binary] = None, truncate: bool = False):
        """
        If truncate is true, we will truncate the file at the end of the virtual address.
        This leaves some overlay code if present. The idea is just to prevent very simple attacks that
        append a lot of bytes at the end of the PE file.
        """
        if pe_binary is not None:
            lief_binary = pe_binary
        else:
            lief_errors = (lief.bad_format, lief.bad_file, lief.pe_error, lief.parser_error, lief.read_out_of_bound,
                           RuntimeError)
            try:
                lief_binary = lief.PE.parse(list(bytez))
            except lief_errors as e:
                print("lief error: ", str(e))
                lief_binary = None  # TODO Should we raise Exception to malware prediction?
            except Exception:  # everything else (KeyboardInterrupt, SystemExit, ValueError):
                raise

        if truncate is True:  # TODO and lief_binary is not None:
            bytez = bytez[:lief_binary.virtual_size]

        features = {"sha256": hashlib.sha256(bytez).hexdigest()}
        features.update({fe.name: fe.raw_features(bytez, lief_binary) for fe in self.features})
        return features

    def process_raw_features(self, raw_obj):
        feature_vectors = [fe.process_raw_features(raw_obj[fe.name]) for fe in self.features]
        return np.hstack(feature_vectors).astype(np.float32)

    def feature_vector(self, bytez, pe_binary: typing.Optional[lief.PE.Binary] = None, truncate: bool = False):
        return self.process_raw_features(self.raw_features(bytez, pe_binary=pe_binary, truncate=truncate))

    def feature_names(self):
        feature_vectors = [fe.feature_names() for fe in self.features]
        return np.hstack(feature_vectors) #.astype(np.float32)
