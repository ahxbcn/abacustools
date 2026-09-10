from typing import Dict, Any, List
import os
import re
import sys

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
            except (TypeError, ValueError):
                pass
            try:
                return [float(x) for x in s_split]
            except (TypeError, ValueError):
                pass
        else:
            try:
                return int(s)
            except (TypeError, ValueError):
                pass
            try:
                return float(s)
            except (TypeError, ValueError):
                pass
            return s

    input_content = {}

    if INPUTf is not None:
        if not os.path.isfile(INPUTf):
            print(f"Can not find the file {INPUTf}")
            return input_content
        else:
            with open(INPUTf, "r") as f:
                input_lines = f.readlines()

    if input_lines is None:
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

def kspacing2kpt(kspacing, cell):
    """
    Convert kspacing to kpt.
    """
    import numpy as np

    if isinstance(kspacing, (float,int)):
        kspacing = [kspacing, kspacing, kspacing]
    elif isinstance(kspacing, str):
        a = kspacing.split()
        if len(a) == 1:
            kspacing = [float(a[0]), float(a[0]), float(a[0])]
        elif len(a) == 3:
            kspacing = [float(a[0]), float(a[1]), float(a[2])]
        else:
            raise ValueError("kspacing must be one or three floats")
    elif not isinstance(kspacing, list):
        raise TypeError("kspacing must be float or list")
    
    assert len(kspacing) == 3, "kspacing must be 3-dim"
    kpt = []
    V = abs(np.linalg.det(np.array(cell)))
    for i in range(3):
        kpt.append(int(np.linalg.norm(np.cross(cell[(i+1)%3],cell[(i+2)%3])) * 2 * np.pi / V / kspacing[i] + 1))
    return kpt

def IsTrue(param):
    '''
    judge if a parameter is True
    
    If param is a string, then judge if it is "True" or "true" or "T" or "t" or "1", if yes return True, elif is "False" or "false" or "F" or "f" or "0", return False, else return None.
    If param is int or bool, return True if param is True, else return False.
    
    '''
    if isinstance(param,str):
        param = param.rstrip(".").lstrip(".")  # for the case of "True."
        if param.lower() in ["true","t","1"]:
            return True
        elif param.lower() in ["false","f","0"]:
            return False
        else:
            return None
    elif isinstance(param,(int,bool)):
        return True if param else False
    else:
        return None

def ReadKpt(kptpath):
    '''
    kptpath should be a file name of KPT file or a path of ABACUS inputs.
    
    return kpt,model
    - kpt is a list of k-point + shift, such as [1,1,1,0,0,0] 
    - model is the model of k-point, such as "mp","gamma","direct","line"
    '''
    if os.path.isdir(kptpath):
        # try to file input
        if os.path.isfile(os.path.join(kptpath,"INPUT")):
            input_param = ReadInput(os.path.join(kptpath,"INPUT"))
            kptf = os.path.join(kptpath,input_param.get("kpoint_file","KPT"))
            struf = os.path.join(kptpath,input_param.get("stru_file","STRU"))
            kspacing = input_param.get("kspacing",None)
            if input_param.get("basis_type","").lower() == "lcao" and IsTrue(input_param.get("gamma_only",False)):
                print("Have set gamma_only in INPUT file, will use 1 1 1 for KPOINT.")
                return [1,1,1,0,0,0],"gamma"
            elif kspacing is not None and kspacing != 0:
                
                if not os.path.isfile(struf):
                    print("  Can not find the STRU file, and try to read KPOINT from KPT file")
                    if os.path.isfile(kptf):
                        return ReadKpt(kptf)
                    else:
                        print("  Can not find the KPT file:",kptf)
                        sys.exit(1)
                else:
                    from abacustools.io.stru import AbacusSTRU
                    
                    stru = AbacusSTRU.read(struf)
                    if stru is None:
                        print("  Can not read the STRU file:",struf)
                        sys.exit(1)
                    cell = stru.get_cell(bohr=True)
                    kpt = kspacing2kpt(kspacing,cell) + [0,0,0]
                    print(f"Transfer kspacing: {kspacing} to K points {kpt[:3]}.")
                    return kpt,"gamma"
            elif os.path.isfile(kptf):
                return ReadKpt(kptf)
            else:
                print("ERROR: Can not find the KPT file:",kptf)
                sys.exit(1)
        elif os.path.isfile(os.path.join(kptpath,"KPT")):
            return ReadKpt(os.path.join(kptpath,"KPT"))
        else:
            print("ERROR: Can not find the INPUT/KPT file in the path:",kptpath)
            sys.exit(1)
    elif os.path.isfile(kptpath):
        with open(kptpath) as f1:
            lines = [i for i in f1.readlines() if i.split("#")[0].strip() != ""]
        model = lines[2].split()[0].lower()
        if model.startswith("m"):
            model = "mp"
        elif model.startswith("g"):
            model = "gamma"
        elif model.startswith("d"):
            model = "direct"
        elif model.startswith("c"):
            model = "cartesian"
        elif model.lower() == "line":
            model = "line"
        elif model.lower() == "line_cartesian":
            model = "line_cartesian"
        else:
            print(f"ERROR: the model of KPT file is not support now!!!\n{model}")
            sys.exit(1)
            
        if model in ["mp","gamma"]:
            kpt = [int(i) for i in lines[3].split()[:3]] + [float(i) for i in lines[3].split()[3:6]]
            return kpt,model
        elif model in ["direct", "cartesian"]:
            nk = int(lines[1].split()[0])
            kpt = []
            for i in range(nk):
                kpt.append([float(ii) for ii in lines[2+i].split()[:4]])
            return kpt,model
        elif model in ["line", "line_cartesian"]:
            kpt = []
            for line in lines[3:]:
                if line.strip() == "":
                    break
                sline = line.split()
                ik = [float(i) for i in sline[:3]] + [int(sline[3])]
                if len(sline) == 4:
                    kpt.append(ik)
                elif len(sline) > 4:
                    kpt.append(ik + [" ".join(sline[4:])])
            return kpt,model
    else:
        print(f"ERROR: {kptpath} is not a file or path!!!")
        sys.exit(1)

def WriteKpt(kpoint_list:List = [1,1,1,0,0,0],file_name:str = "KPT", model="gamma"):
    '''
    Docs for KPT file: https://abacus.deepmodeling.com/en/latest/advanced/input_files/kpt.html
    ABACUS KPT support three models:
    - gamma/mp: is the Monkhorst-Pack method, such as:
    
        K_POINTS //keyword for start
        0 //total number of k-point, `0' means generate automatically
        Gamma //which kind of Monkhorst-Pack method, `Gamma' or `MP'
        2 2 2 0 0 0 //first three number: subdivisions along recpri. vectors
                    //last three number: shift of the mesh
        
    - direct/cartessian: set the k-point explicitly, such as:
        
            K_POINTS
            1
            Direct
            0.0 0.0 0.0 1.0  // the last number is the weight of this k-point
            
    - line: set the k-point along a line, such as:
            
                K_POINTS
                2  // number of high symmetry k-points along the
                Line
                0.0 0.0 0.0 10  // Gamma the last number is number of k-points between this and next k-point
                0.5 0.0 0.0 1 
    
    For gamma/mp model, the k-point_list should be a list of 6 values, such as:
    [2,2,2,0,0,0]
    
    For explicitly model, the k-point_list should be a list of list of 3 or 4 values, such as:
    [[0.0,0.0,0.0,1.0],[0.5,0.0,0.0,1.0]]
    
    For line model, the k-point_list should be a list of list of 4 or 5 values (the last value is a string of comment), such as:
    [[0.0,0.0,0.0,10],[0.5,0.0,0.0,1]] or
    [[0.0,0.0,0.0,10 "#Gamma"],[0.5,0.0,0.0,1,"//"],[0.5,0.5,0.0,1,"//"]]         
    '''
    if model.lower() in ["gamma","mp"]:
        model_ = "Gamma" if model.lower() == "gamma" else "MP"
        with open(file_name,'w') as f1:
            f1.write(F"K_POINTS\n0\n{model_}\n")
            if len(kpoint_list) == 3:
                kpoint_list += [0,0,0]
            f1.write(" ".join([str(i) for i in kpoint_list]))
    elif model.lower() in ["direct","cartessian"]:
        # normalize the weight
        kpt = []
        if len(kpoint_list[0]) == 3:
            kpt = [i+[1.0/len(kpoint_list)] for i in kpoint_list]
        elif len(kpoint_list[0]) == 4:
            total_weight = sum([i[3] for i in kpoint_list])
            kpt = [i[:3]+[i[3]/total_weight] for i in kpoint_list]
        else:
            print(f"ERROR: model is {model}, the kpoint_list is not correct!!!\n{kpoint_list}")
            sys.exit(1)
            
        with open(file_name,'w') as f1:
            f1.write(f"K_POINTS\n{len(kpoint_list)}\n{model.capitalize()}\n")
            for i in kpt:
                f1.write("%17.11f %17.11f %17.11f %17.11f\n" % tuple(i))
    elif model.lower() in ["line", "line_cartesian"]:
        with open(file_name,'w') as f1:
            f1.write(f"K_POINTS\n{len(kpoint_list)}\n")
            if model.lower() == "line":
                f1.write("Line\n")
            else:
                f1.write("Line_Cartesian\n")
            for i in kpoint_list:
                if len(i) == 4:
                    f1.write("%17.11f %17.11f %17.11f %4d\n" % tuple(i))
                elif len(i) == 5:
                    if not (i[-1].startswith("#") or i[-1].startswith("//")):
                        i[-1] = "#"+i[-1]
                    f1.write("%17.11f %17.11f %17.11f %4d %s\n" % tuple(i))
    else:
        print(f"ERROR: model is {model}, not support now!!!")
        sys.exit(1)
