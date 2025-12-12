from enum import Enum


class License(str, Enum):
    MIT = "MIT"
    APACHE_2_0 = "Apache license 2.0"
    CC_BY_4_0 = "Creative Commons Attribution 4.0"
    CC_BY_SA_4_0 = "Creative Commons Attribution ShareAlike 4.0"
    CC_BY_NC_SA_4_0 = "Creative Commons Attribution NonCommercial ShareAlike 4.0"
    ODC_BY = "Open Data Commons Attribution License"
    UNLICENSED = "Unlicensed"

    # AFL_3_0 = "Academic Free License v3.0"
    # ARTISTIC_2_0 = "Artistic license 2.0"
    # BSL_1_0 = "Boost Software License 1.0"
    #
    # BSD_2_CLAUSE = "BSD 2-clause license"
    # BSD_3_CLAUSE = "BSD 3-clause license"
    # BSD_3_CLAUSE_CLEAR = "BSD 3-clause Clear license"
    # BSD_4_CLAUSE = "BSD 4-clause license"
    # BSD_0 = "BSD Zero-Clause license"
    #
    # CC = "Creative Commons license family"
    # CC0_1_0 = "Creative Commons Zero v1.0 Universal"
    #
    # ECL_2_0 = "Educational Community License v2.0"
    # EPL_1_0 = "Eclipse Public License 1.0"
    # EPL_2_0 = "Eclipse Public License 2.0"
    # EUPL_1_1 = "European Union Public License 1.1"
    #
    # AGPL_3_0 = "GNU Affero General Public License v3.0"
    # GPL = "GNU General Public License family"
    # GPL_2_0 = "GNU General Public License v2.0"
    # GPL_3_0 = "GNU General Public License v3.0"
    # LGPL = "GNU Lesser General Public License family"
    # LGPL_2_1 = "GNU Lesser General Public License v2.1"
    # LGPL_3_0 = "GNU Lesser General Public License v3.0"
    #
    # ISC = "ISC"
    # LPPL_1_3C = "LaTeX Project Public License v1.3c"
    # MS_PL = "Microsoft Public License"
    # MPL_2_0 = "Mozilla Public License 2.0"
    # OSL_3_0 = "Open Software License 3.0"
    # POSTGRESQL = "PostgreSQL License"
    # OFL_1_1 = "SIL Open Font License 1.1"
