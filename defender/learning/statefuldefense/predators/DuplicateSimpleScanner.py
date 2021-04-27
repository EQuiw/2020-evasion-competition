import lief

from learning.statefuldefense.predators.PredatorInterface import PredatorInterface


class DuplicateSimpleScanner(PredatorInterface):
    """
    Checks sections for duplicates.
    Implements a simple, but fast solution.
    Better strategies should be explored in future work.
    """

    def __init__(self, min_number_matches: int, verbose: bool):
        """
        min_number_matches = minimum number of sections that must repeat in order to return true.
        """
        self.min_number_matches = min_number_matches
        self.verbose = verbose

    # @Overwrite
    def check_file(self, bytez, lief_binary) -> bool:

        # we prefer the already extracted binary
        if lief_binary is not None:
            binary = lief_binary
        else:
            assert bytez is not None
            binary = lief.parse(bytez)

        setOfElements = {}
        matches: int = 0
        for section in binary.sections:
            # print(section.size)
            if section.size > 0:  # sections like .bss or .tls may have zero size, leading to FPs
                elem = str(section.content)
                if elem in setOfElements:
                    setOfElements[elem] += 1
                    matches += 1
                    if self.verbose is False and matches >= self.min_number_matches:
                        return True  # already enough to break here
                else:
                    setOfElements[elem] = 1

        # some optional debug information..
        if self.verbose is True and matches > 0:
            matched = []
            for k, v in setOfElements.items():
                if v > 1:
                    matched.append(v)

            if lief_binary is not None:
                print("has same sections again", matched, len(lief_binary.sections), matches >= self.min_number_matches)
            else:
                print("has same sections again", matched, matches >= self.min_number_matches)

        return matches >= self.min_number_matches
