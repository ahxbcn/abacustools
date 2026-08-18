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


def WriteInput(input_context: Dict[str, any],
               INPUTf: str = "INPUT",
               categorized: bool = True):
    """Write the input parameters to the INPUT file.
    
    Args:
        input_context (Dict[str, any]): a dictionary of input parameters.
        INPUTf (str): the file name of the INPUT file to write.
        categorized (bool): whether to organize parameters into categorized groups.
            When True (default), parameters are grouped by category with comments
            for better readability. When False, parameters are written in
            dictionary order without category grouping.
            
    Returns:
        None: the function will write the input parameters to the INPUT file.
        
    If the value of a parameter is None, it will be written as a comment line.
    If the value is a list or tuple, it will be joined with spaces.
    """

    INPUT_CATEGORIES = {}
    
    if categorized:
        import json
        import warnings
        
        try:
            from importlib.resources import files
            json_path = files('abacustools.io') / 'input-params.json'
            if json_path.is_file():
                data = json.loads(json_path.read_text())
                for item in data:
                    name = item.get('name', '')
                    category = item.get('category', 'Other')
                    if name and category:
                        names = [n.strip() for n in name.split(',')]
                        if category not in INPUT_CATEGORIES:
                            INPUT_CATEGORIES[category] = []
                        INPUT_CATEGORIES[category].extend(names)
        except Exception as e:
            warnings.warn(f"Failed to load input categories from package data: {e}. "
                           f"Parameters will not be categorized.")
        
        if not INPUT_CATEGORIES:
            categorized = False

    def _format_param(key: str, value: any, key_width: int) -> str:
        padding = max(key_width - len(key), 1)
        if value is not None:
            if isinstance(value, (list, tuple)):
                value = ' '.join([str(i) for i in value])
            return f"{key}{' ' * padding}{value}\n"
        return f"#{key}\n"
    
    out = "INPUT_PARAMETERS\n"

    if categorized:
        used_keys = set()
        
        for category, keys in INPUT_CATEGORIES.items():
            category_params = [k for k in keys if k in input_context]
            used_keys.update(category_params)
            
            if category_params:
                out += f"\n# {category}\n"
                out += ''.join(_format_param(k, input_context[k], 20) for k in category_params)

        remaining_params = [k for k in input_context.keys() if k not in used_keys]
        if remaining_params:
            out += "\n# Other Parameters\n"
            out += ''.join(_format_param(k, input_context[k], 20) for k in remaining_params)
    else:
        out += ''.join(_format_param(k, v, 20) for k, v in input_context.items())
    
    with open(INPUTf, 'w') as f1:
        f1.write(out)
