import argparse
import re
from collections import Counter

def main() -> None:
    parser = argparse.ArgumentParser(description="ananlyze suffix array log")
    parser.add_argument("logfile", help="input log file path")
    
    args = parser.parse_args()
    logfile = args.logfile
    
    matched = []
    tests = {"test_matched_sublines": {}, "test_matched_original_sublines": {}}
    status = None
    low_threshold = None
    for line in open(logfile):
        # matched chars
        if "matched: " in line:
            line = line.replace("matched: ", "")
            matched.append(line)
        # flag of test_matched_sublines or test_matched_original_sublines
        elif "test_matched_sublines:" in line:
            status = "test_matched_sublines"
        elif "test_matched_original_sublines:" in line:
            status = "test_matched_original_sublines"
        elif status is not None:
            m = re.match(r"low: ([\d\.]+)", line)
            if m:
                low_threshold = float(m.group(1))
                tests[status][low_threshold] = []
            else:
                m = re.match(r"\d+\s", line)
                if m:
                    tests[status][low_threshold].append(line[m.end():])
    print("result:")
    print(Counter(matched))
    
    for status,v in tests.items():
        for low_threshold,v2 in v.items():
            print(status, low_threshold)
            print(Counter(v2))

if __name__ == "__main__":
    main()