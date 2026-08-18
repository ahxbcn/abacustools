from typing import Dict, Any
import os
import re

def ReadInput(INPUTf: str = None, input_lines: str = None) -> Dict[str, Any]:
    """
    Read the INPUT file and return a dictionary of input parameters.

    Args:
        INPUTf (str): the filename of the INPUT file.
        input_lines (str): the lines of the INPUT file.
    
    Returns:
        Dict[str, Any]: a dictionary of input parameters.
    
    If both `INPUTf` and `input_lines` are provided, `input_lines` will be ignored.
    The value of each parameter will be converted to int or float if possible.
    """
    def str2intfloat(s: str):
        s_split = s.split()
        if len(s_split) > 1:
            try:
                return [int(x) for x in s_split]
            except:
                pass
            try:
                return [float(x) for x in s_split]
            except:
                pass
        else:
            try:
                return int(s)
            except:
                pass
            try:
                return float(s)
            except:
                pass

    input_content = {}

    if INPUTf != None:
        if not os.path.isfile(INPUTf):
            print(f"Can not find the file {INPUTf}")
            return input_content
        else:
            with open(INPUTf, "r") as f:
                input_lines = f.readlines()

    if input_lines == None:
        print(INPUTf)
        print("Please provide the INPUT file name of INPUT lines")
        return input_content

    for i, iline in enumerate(input_lines):
        if iline.strip() == "" or iline.strip()[0] in ["#"]:
            continue
        else:
            sline = re.split('[ \t]', iline.split("#")[0].strip(), maxsplit=1)
            if len(sline) == 2:
                k = sline[0].lower().strip()
                v = str2intfloat(sline[1].strip())
                input_content[k] = v

    return input_content
