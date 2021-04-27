#!/usr/bin/python
''' Extracts some basic features from PE files. Based on
https://github.com/elastic/ember/blob/master/ember/features.py

NOTE: In this file, we have adjusted the feature extraction. See below!
I recommend making a diff to default ember features.py file.
'''

import re
import lief
import hashlib
import typing
import numpy as np
from sklearn.feature_extraction import FeatureHasher

LIEF_MAJOR, LIEF_MINOR, _ = lief.__version__.split('.')
LIEF_EXPORT_OBJECT = int(LIEF_MAJOR) > 0 or ( int(LIEF_MAJOR)==0 and int(LIEF_MINOR) >= 10 )


def remove_strings_from_bytez(bytez, re_strings):
    # return re.sub(b'[\x20-\x7f]{4,}', b'', bytez)
    return re_strings.sub(b'', bytez)


class FeatureType(object):
    ''' Base class from which each feature type may inherit '''

    name = ''
    dim = 0

    def __repr__(self):
        return '{}({})'.format(self.name, self.dim)

    def raw_features(self, bytez, lief_binary):
        ''' Generate a JSON-able representation of the file '''
        raise (NotImplementedError)

    def process_raw_features(self, raw_obj):
        ''' Generate a feature vector from the raw features '''
        raise (NotImplementedError)

    def feature_vector(self, bytez, lief_binary):
        ''' Directly calculate the feature vector from the sample itself. This should only be implemented differently
        if there are significant speedups to be gained from combining the two functions. '''
        return self.process_raw_features(self.raw_features(bytez, lief_binary))

    def feature_names(self):
        ''' Generate a list that specifies the name of the columns '''
        raise (NotImplementedError)


class ByteHistogram(FeatureType):
    ''' Byte histogram (count + non-normalized) over the entire binary file '''

    name = 'histogram'
    dim = 256

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        return counts.tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized

    def feature_names(self):
        return [self.name + "_" + str(0) for i in range(0, self.dim)]

class ByteEntropyHistogram(FeatureType):
    ''' 2d byte/entropy histogram based loosely on (Saxe and Berlin, 2015).
    This roughly approximates the joint probability of byte value and local entropy.
    See Section 2.1.1 in https://arxiv.org/pdf/1508.03096.pdf for more info.
    '''

    name = 'byteentropy'
    dim = 256

    def __init__(self, step=1024, window=2048):
        super(FeatureType, self).__init__()
        self.window = window
        self.step = step

    def _entropy_bin_counts(self, block):
        # coarse histogram, 16 bytes per bin
        c = np.bincount(block >> 4, minlength=16)  # 16-bin histogram
        p = c.astype(np.float32) / self.window
        wh = np.where(c)[0]
        H = np.sum(-p[wh] * np.log2(
            p[wh])) * 2  # * x2 b.c. we reduced information by half: 256 bins (8 bits) to 16 bins (4 bits)

        Hbin = int(H * 2)  # up to 16 bins (max entropy is 8 bits)
        if Hbin == 16:  # handle entropy = 8.0 bits
            Hbin = 15

        return Hbin, c

    def raw_features(self, bytez, lief_binary):

        output = np.zeros((16, 16), dtype=np.int)
        a = np.frombuffer(bytez, dtype=np.uint8)
        if a.shape[0] < self.window:
            Hbin, c = self._entropy_bin_counts(a)
            output[Hbin, :] += c
        else:
            # strided trick from here: http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
            shape = a.shape[:-1] + (a.shape[-1] - self.window + 1, self.window)
            strides = a.strides + (a.strides[-1],)
            blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::self.step, :]

            # from the blocks, compute histogram
            for block in blocks:
                Hbin, c = self._entropy_bin_counts(block)
                output[Hbin, :] += c

        return output.flatten().tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized

    def feature_names(self):
        # todo get better name
        return [self.name + "_" + str(0) for i in range(0, self.dim)]


class SectionInfo(FeatureType):
    ''' Information about section names, sizes and entropy.  Uses hashing trick
    to summarize all this section info into a feature vector.
    '''

    name = 'section'
    dim = 5 + 50 + 50 + 50 + 50 + 50

    def __init__(self):
        super(FeatureType, self).__init__()

    @staticmethod
    def _properties(s):
        return [str(c).split('.')[-1] for c in s.characteristics_lists]

    def raw_features(self, bytez, lief_binary):
        if lief_binary is None:
            return {"entry": "", "sections": []}

        # properties of entry point, or if invalid, the first executable section
        try:
            entry_section = lief_binary.section_from_offset(lief_binary.entrypoint).name
        except lief.not_found:
            # bad entry point, let's find the first executable section
            entry_section = ""
            for s in lief_binary.sections:
                if lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE in s.characteristics_lists:
                    entry_section = s.name
                    break

        raw_obj = {"entry": entry_section}
        raw_obj["sections"] = [{
            'name': s.name,
            'size': s.size,
            'entropy': s.entropy,
            'vsize': s.virtual_size,
            'props': self._properties(s)
        } for s in lief_binary.sections]
        return raw_obj

    def process_raw_features(self, raw_obj):
        sections = raw_obj['sections']
        general = [
            len(sections),  # total number of sections
            # number of sections with nonzero size
            sum(1 for s in sections if s['size'] == 0),
            # number of sections with an empty name
            sum(1 for s in sections if s['name'] == ""),
            # number of RX
            sum(1 for s in sections if 'MEM_READ' in s['props'] and 'MEM_EXECUTE' in s['props']),
            # number of W
            sum(1 for s in sections if 'MEM_WRITE' in s['props'])
        ]
        # gross characteristics of each section
        section_sizes = [(s['name'], s['size']) for s in sections]
        section_sizes_hashed = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
        section_entropy = [(s['name'], s['entropy']) for s in sections]
        section_entropy_hashed = FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0]
        section_vsize = [(s['name'], s['vsize']) for s in sections]
        section_vsize_hashed = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
        entry_name_hashed = FeatureHasher(50, input_type="string").transform([raw_obj['entry']]).toarray()[0]
        characteristics = [p for s in sections for p in s['props'] if s['name'] == raw_obj['entry']]
        characteristics_hashed = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]

        return np.hstack([
            general, section_sizes_hashed, section_entropy_hashed, section_vsize_hashed, entry_name_hashed,
            characteristics_hashed
        ]).astype(np.float32)

    def feature_names(self):
        return \
        [self.name + "_" + x for x in ["totalNoSections", "NoSectionsNonZeroSize", "NoSectionsEmptyName", "NoRX", "NoW"]] + \
        [self.name + "_" + "SectionSizes_H" + str(i) for i in range(50)] + \
        [self.name + "_" +"SectionEntropies_H" + str(i) for i in range(50)] + \
        [self.name + "_" +"SectionVSizes_H" + str(i) for i in range(50)] + \
        [self.name + "_" +"SectionEntryName_H" + str(i) for i in range(50)] + \
        [self.name + "_" +"CharacteristicsEntry_H" + str(i) for i in range(50)]


class ImportsInfo(FeatureType):
    ''' Information about imported libraries and functions from the
    import address table.  Note that the total number of imported
    functions is contained in GeneralFileInfo.
    '''

    name = 'imports'
    dim = 1280

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        imports = {}
        if lief_binary is None:
            return imports

        for lib in lief_binary.imports:
            if lib.name not in imports:
                imports[lib.name] = []  # libraries can be duplicated in listing, extend instead of overwrite

            # Clipping assumes there are diminishing returns on the discriminatory power of imported functions
            #  beyond the first 10000 characters, and this will help limit the dataset size
            for entry in lib.entries:
                if entry.is_ordinal:
                    imports[lib.name].append("ordinal" + str(entry.ordinal))
                else:
                    imports[lib.name].append(entry.name[:10000])

        return imports

    def process_raw_features(self, raw_obj):
        # unique libraries
        libraries = list(set([l.lower() for l in raw_obj.keys()]))
        libraries_hashed = FeatureHasher(256, input_type="string").transform([libraries]).toarray()[0]

        # A string like "kernel32.dll:CreateFileMappingA" for each imported function
        imports = [lib.lower() + ':' + e for lib, elist in raw_obj.items() for e in elist]
        imports_hashed = FeatureHasher(1024, input_type="string").transform([imports]).toarray()[0]

        # Two separate elements: libraries (alone) and fully-qualified names of imported functions
        return np.hstack([libraries_hashed, imports_hashed]).astype(np.float32)

    def feature_names(self):
        return \
        [self.name + "_" + "Libraries_H" + str(i) for i in range(256)] + \
        [self.name + "_" + "Imports_H" + str(i) for i in range(1024)]


class ExportsInfo(FeatureType):
    ''' Information about exported functions. Note that the total number of exported
    functions is contained in GeneralFileInfo.
    '''

    name = 'exports'
    dim = 128

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        if lief_binary is None:
            return []

        # Clipping assumes there are diminishing returns on the discriminatory power of exports beyond
        #  the first 10000 characters, and this will help limit the dataset size
        if LIEF_EXPORT_OBJECT:
            # export is an object with .name attribute (0.10.0 and later)
            clipped_exports = [export.name[:10000] for export in lief_binary.exported_functions]
        else:
            # export is a string (LIEF 0.9.0 and earlier)
            clipped_exports = [export[:10000] for export in lief_binary.exported_functions]
        

        return clipped_exports

    def process_raw_features(self, raw_obj):
        exports_hashed = FeatureHasher(128, input_type="string").transform([raw_obj]).toarray()[0]
        return exports_hashed.astype(np.float32)

    def feature_names(self):
        return [self.name + "_H" + str(i) for i in range(128)]


class GeneralFileInfo(FeatureType):
    ''' General information about the file '''

    name = 'general'
    dim = 10

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        if lief_binary is None:
            return {
                'size': len(bytez),
                'vsize': 0,
                'has_debug': 0,
                'exports': 0,
                'imports': 0,
                'has_relocations': 0,
                'has_resources': 0,
                'has_signature': 0,
                'has_tls': 0,
                'symbols': 0
            }

        return {
            'size': len(bytez),
            'vsize': lief_binary.virtual_size,
            'has_debug': int(lief_binary.has_debug),
            'exports': len(lief_binary.exported_functions),
            'imports': len(lief_binary.imported_functions),
            'has_relocations': int(lief_binary.has_relocations),
            'has_resources': int(lief_binary.has_resources),
            'has_signature': int(lief_binary.has_signature),
            'has_tls': int(lief_binary.has_tls),
            'symbols': len(lief_binary.symbols),
        }

    def process_raw_features(self, raw_obj):
        return np.asarray([
            raw_obj['size'], raw_obj['vsize'], raw_obj['has_debug'], raw_obj['exports'], raw_obj['imports'],
            raw_obj['has_relocations'], raw_obj['has_resources'], raw_obj['has_signature'], raw_obj['has_tls'],
            raw_obj['symbols']
        ],
                          dtype=np.float32)

    def feature_names(self):
        return [self.name + "_" + x for x in ['size', 'vsize', 'has_debug', 'exports', 'imports',
                                              'has_relocations', 'has_resources', 'has_signature', 'has_tls',
                                              'symbols']]


class HeaderFileInfo(FeatureType):
    ''' Machine, architecure, OS, linker and other information extracted from header '''

    name = 'header'
    dim = 62

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        raw_obj = {}
        raw_obj['coff'] = {'timestamp': 0, 'machine': "", 'characteristics': []}
        raw_obj['optional'] = {
            'subsystem': "",
            'dll_characteristics': [],
            'magic': "",
            'major_image_version': 0,
            'minor_image_version': 0,
            'major_linker_version': 0,
            'minor_linker_version': 0,
            'major_operating_system_version': 0,
            'minor_operating_system_version': 0,
            'major_subsystem_version': 0,
            'minor_subsystem_version': 0,
            'sizeof_code': 0,
            'sizeof_headers': 0,
            'sizeof_heap_commit': 0
        }
        if lief_binary is None:
            return raw_obj

        raw_obj['coff']['timestamp'] = lief_binary.header.time_date_stamps
        raw_obj['coff']['machine'] = str(lief_binary.header.machine).split('.')[-1]
        raw_obj['coff']['characteristics'] = [str(c).split('.')[-1] for c in lief_binary.header.characteristics_list]
        raw_obj['optional']['subsystem'] = str(lief_binary.optional_header.subsystem).split('.')[-1]
        raw_obj['optional']['dll_characteristics'] = [
            str(c).split('.')[-1] for c in lief_binary.optional_header.dll_characteristics_lists
        ]
        raw_obj['optional']['magic'] = str(lief_binary.optional_header.magic).split('.')[-1]
        raw_obj['optional']['major_image_version'] = lief_binary.optional_header.major_image_version
        raw_obj['optional']['minor_image_version'] = lief_binary.optional_header.minor_image_version
        raw_obj['optional']['major_linker_version'] = lief_binary.optional_header.major_linker_version
        raw_obj['optional']['minor_linker_version'] = lief_binary.optional_header.minor_linker_version
        raw_obj['optional'][
            'major_operating_system_version'] = lief_binary.optional_header.major_operating_system_version
        raw_obj['optional'][
            'minor_operating_system_version'] = lief_binary.optional_header.minor_operating_system_version
        raw_obj['optional']['major_subsystem_version'] = lief_binary.optional_header.major_subsystem_version
        raw_obj['optional']['minor_subsystem_version'] = lief_binary.optional_header.minor_subsystem_version
        raw_obj['optional']['sizeof_code'] = lief_binary.optional_header.sizeof_code
        raw_obj['optional']['sizeof_headers'] = lief_binary.optional_header.sizeof_headers
        raw_obj['optional']['sizeof_heap_commit'] = lief_binary.optional_header.sizeof_heap_commit
        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            raw_obj['coff']['timestamp'],
            FeatureHasher(10, input_type="string").transform([[raw_obj['coff']['machine']]]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([raw_obj['coff']['characteristics']]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([[raw_obj['optional']['subsystem']]]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([raw_obj['optional']['dll_characteristics']]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([[raw_obj['optional']['magic']]]).toarray()[0],
            raw_obj['optional']['major_image_version'],
            raw_obj['optional']['minor_image_version'],
            raw_obj['optional']['major_linker_version'],
            raw_obj['optional']['minor_linker_version'],
            raw_obj['optional']['major_operating_system_version'],
            raw_obj['optional']['minor_operating_system_version'],
            raw_obj['optional']['major_subsystem_version'],
            raw_obj['optional']['minor_subsystem_version'],
            raw_obj['optional']['sizeof_code'],
            raw_obj['optional']['sizeof_headers'],
            raw_obj['optional']['sizeof_heap_commit'],
        ]).astype(np.float32)

    def feature_names(self):
        out_vec = \
        [self.name + "_" + "timestamp"] + \
        [self.name + "_" +"coff_machine_H" + str(i) for i in range(10)] + \
        [self.name + "_" +"coff_characteristics_H" + str(i) for i in range(10)] + \
        [self.name + "_" +"optional_subsystem_H" + str(i) for i in range(10)] + \
        [self.name + "_" +"optional_dll_characteristics_H" + str(i) for i in range(10)] + \
        [self.name + "_" +"optional_magic" + str(i) for i in range(10)] + \
        [self.name + "_" + "optional_"+ x for x in ['major_image_version',
                                                    'minor_image_version',
                                                    'major_linker_version',
                                                    'minor_linker_version',
                                                    'major_operating_system_version',
                                                    'minor_operating_system_version',
                                                    'major_subsystem_version',
                                                    'minor_subsystem_version',
                                                    'sizeof_code',
                                                    'sizeof_headers',
                                                    'sizeof_heap_commit']]
        assert len(out_vec) == self.dim
        return out_vec




class StringExtractor(FeatureType):
    ''' Extracts strings from raw byte stream '''

    name = 'strings'
    dim = 1 + 1 + 1 + 96 + 1 + 1 + 1 + 1 + 1

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
            np.asarray(raw_obj['printabledist']) / hist_divisor, raw_obj['entropy'], raw_obj['paths'], raw_obj['urls'],
            raw_obj['registry'], raw_obj['MZ']
        ]).astype(np.float32)

    def feature_names(self):
        a1 = [self.name + "_" + x for x in ['numstrings', 'avlength', 'printables']]
        a2 = [self.name + "_printabledist_normed_" + str(i) for i in range(96)]
        a3 = [self.name + "_" + x for x in ['entropy', 'paths', 'urls', 'registry', 'MZ']]
        out = a1 + a2 + a3
        assert len(out) == self.dim
        return out



class DataDirectories(FeatureType):
    ''' Extracts size and virtual address of the first 15 data directories '''

    name = 'datadirectories'
    dim = 15 * 2

    def __init__(self):
        super(FeatureType, self).__init__()
        self._name_order = [
            "EXPORT_TABLE", "IMPORT_TABLE", "RESOURCE_TABLE", "EXCEPTION_TABLE", "CERTIFICATE_TABLE",
            "BASE_RELOCATION_TABLE", "DEBUG", "ARCHITECTURE", "GLOBAL_PTR", "TLS_TABLE", "LOAD_CONFIG_TABLE",
            "BOUND_IMPORT", "IAT", "DELAY_IMPORT_DESCRIPTOR", "CLR_RUNTIME_HEADER"
        ]

    def raw_features(self, bytez, lief_binary):
        output = []
        if lief_binary is None:
            return output

        for data_directory in lief_binary.data_directories:
            output.append({
                "name": str(data_directory.type).replace("DATA_DIRECTORY.", ""),
                "size": data_directory.size,
                "virtual_address": data_directory.rva
            })
        return output

    def process_raw_features(self, raw_obj):
        features = np.zeros(2 * len(self._name_order), dtype=np.float32)
        for i in range(len(self._name_order)):
            if i < len(raw_obj):
                features[2 * i] = raw_obj[i]["size"]
                features[2 * i + 1] = raw_obj[i]["virtual_address"]
        return features

    def feature_names(self):
        out = []
        for x in self._name_order:
            out.append(self.name + "_" + x + "_Size")
            out.append(self.name + "_" + x + "_VirtAdr")
        assert len(out) == self.dim
        return out


class PEFeatureExtractor(object):
    ''' Extract useful features from a PE file, and return as a vector of fixed size. '''

    def __init__(self, feature_version=2, include_header: bool = False, remove_strings_stream: bool = False):

        self.remove_strings_stream = remove_strings_stream
        self.features = [
            ByteHistogram(),
            ByteEntropyHistogram(),
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
            self.features.append(DataDirectories())
            if not lief.__version__.startswith("0.9.0") and not lief.__version__.startswith("0.10.1"):
                print(f"WARNING: EMBER feature version 2 were computed using lief version 0.9.0-")
                print(f"WARNING:   lief version {lief.__version__} found instead. There may be slight inconsistencies")
                print(f"WARNING:   in the feature calculations.")
        else:
            raise Exception(f"EMBER feature version must be 1 or 2. Not {feature_version}")
        self.dim = sum([fe.dim for fe in self.features])

        self.re_strings = None
        if self.remove_strings_stream is True:
            self.re_strings = re.compile(b'[\x20-\x7f]{4,}')


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

        # Extract features, we modify this here to avoid string removal in each class, we do it here once.
        # features.update({fe.name: fe.raw_features(bytez, lief_binary) for fe in self.features})
        bytez_removed_strings = None
        for fe in self.features:
            if self.remove_strings_stream is True:
                if isinstance(fe, ByteHistogram) or isinstance(fe, ByteEntropyHistogram):
                    if bytez_removed_strings is None:
                        bytez_removed_strings = remove_strings_from_bytez(bytez=bytez, re_strings=self.re_strings)
                    features.update({fe.name: fe.raw_features(bytez_removed_strings, lief_binary)})
                    continue
            features.update({fe.name: fe.raw_features(bytez, lief_binary)})

        return features

    def process_raw_features(self, raw_obj):
        feature_vectors = [fe.process_raw_features(raw_obj[fe.name]) for fe in self.features]
        return np.hstack(feature_vectors).astype(np.float32)

    def feature_vector(self, bytez, pe_binary: typing.Optional[lief.PE.Binary] = None, truncate: bool = False):
        return self.process_raw_features(self.raw_features(bytez, pe_binary=pe_binary, truncate=truncate))

    def feature_names(self):
        feature_vectors = [fe.feature_names() for fe in self.features]
        return np.hstack(feature_vectors) #.astype(np.float32)