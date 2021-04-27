import lief

from learning.statefuldefense.predators.PredatorInterface import PredatorInterface


class SlackSpaceScanner(PredatorInterface):
    """
    Checks if slack space (space between sections due to alignment) is filled.
    """

    def __init__(self, min_number_absolute_matches: int, min_number_relative_matches: float, verbose: bool):
        """
        min_number_absolute_matches = minimum absolute number of filled slack spaces in order to return true.
        min_number_relative_matches = minimum number of filled spaces relative to number of sections

        By setting min_number_absolute_matches, we only start to check for filled slack spaces if we have more than
        min_number_absolute_matches slack spaces. If so, we check the ratio of filled-spaces / no-sections.

        For example: min_number_absolute_matches = 3, min_number_relative_matches = 0.333334
        Now assume that File A has 4 sections, and 3 are filled; and File B has 2 sections, and 2 are filled.
        As File A has 3 filled sections, we return True.
        However, we return False for File B, although it has 2 filled sections (2 out of 2), but with 2
        sections, we have not enough information to tell if we have adversarially added slack spaces.
        """

        self.min_number_absolute_matches: int = min_number_absolute_matches
        self.min_number_relative_matches: float = min_number_relative_matches
        self.verbose: bool = verbose
        self.ignore_rsrc: bool = False

    # @Overwrite
    def check_file(self, bytez, lief_binary) -> bool:
        return self.scan_slack_space(bytez=bytez, lief_binary=lief_binary)

    def scan_slack_space(self, bytez, lief_binary) -> bool:
        count_method_with_next_section = False
        assert bytez is not None

        # we prefer the already extracted binary
        if lief_binary is not None:
            lief_binary = lief_binary
        else:
            lief_binary = lief.parse(bytez)

        # possibleslackspaces = []

        matched_sectionnames = 0
        sections_list = list(lief_binary.sections)
        for isec, section in enumerate(sections_list):

            # I found that rsrc is quite often filled in benign files, we could ignore it therefore.
            if self.ignore_rsrc is True and ".rsrc" in str(section.name):
                continue

            minvalx = min(section.virtual_size, section.sizeof_raw_data)

            if count_method_with_next_section is True and isec < len(sections_list) - 1:
                nextoffset = sections_list[isec + 1].pointerto_raw_data
            else:
                nextoffset = section.pointerto_raw_data + section.sizeof_raw_data

            slackspace = (section.pointerto_raw_data + minvalx, nextoffset)
            # possibleslackspaces.append(slackspace)

            ismatch = any(bytez[slackspace[0]: slackspace[1]])
            if ismatch:
                matched_sectionnames += 1

        if self.verbose:
            print("{} {}%".format(matched_sectionnames, round(matched_sectionnames / len(sections_list) * 100, 4)))

        if matched_sectionnames < self.min_number_absolute_matches:
            return False

        ratio: float = matched_sectionnames / len(sections_list)
        return ratio >= self.min_number_relative_matches
