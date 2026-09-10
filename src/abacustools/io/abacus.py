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


def IsEnabled(param) -> bool:
    """Return whether an ABACUS flag-like value is enabled.

    Booleans and numbers are compared with zero, the strings true/t/yes/y/1 are
    enabled, and every other string or value is disabled.  A list or tuple is
    enabled when any of its entries is enabled.
    """
    if isinstance(param, (list, tuple)):
        return any(IsEnabled(item) for item in param)
    if isinstance(param, str):
        text = param.strip().lower()
        if text in {"true", "t", "yes", "y", "1"}:
            return True
        if text in {"false", "f", "no", "n", "0", ""}:
            return False
        try:
            return float(text) > 0
        except ValueError:
            return False
    try:
        return float(param) > 0
    except (TypeError, ValueError):
        return False


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
                # lines[2] holds the model name, so the k-points start at lines[3].
                kpt.append([float(ii) for ii in lines[3+i].split()[:4]])
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

_KPT_MODELS = ("gamma", "mp", "direct", "cartesian", "line", "line_cartesian")
_KPT_MODEL_ALIASES = {"cartessian": "cartesian"}
_KPT_HEADERS = {
    "gamma": "Gamma",
    "mp": "MP",
    "direct": "Direct",
    "cartesian": "Cartesian",
    "line": "Line",
    "line_cartesian": "Line_Cartesian",
}


def NormalizeKptModel(model: str) -> str:
    """Return the canonical KPT model name.

    The misspelling ``cartessian`` is accepted as an alias of ``cartesian``
    because it was the only spelling the writer understood historically.
    """
    name = str(model).strip().lower()
    name = _KPT_MODEL_ALIASES.get(name, name)
    if name not in _KPT_MODELS:
        raise ValueError(
            f"unsupported KPT model: {model!r}; supported models are {list(_KPT_MODELS)}"
        )
    return name


def _kpt_number(value, description: str) -> float:
    """Convert one KPT value to a float with a readable error message."""
    if isinstance(value, bool):
        raise ValueError(f"KPT {description} must be a number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"KPT {description} must be a number, got {value!r}") from error


def _kpt_text(value: float) -> str:
    """Format a KPT number, keeping integral values free of a decimal part."""
    number = float(value)
    if number == int(number):
        return str(int(number))
    return repr(number)


def _kpt_groups(kpoint_list) -> list:
    """Split a flat or nested k-point argument into a list of groups."""
    if isinstance(kpoint_list, (str, bytes)) or not isinstance(kpoint_list, (list, tuple)):
        raise ValueError("kpoint_list must be a list of values")
    values = list(kpoint_list)
    if not values:
        raise ValueError("kpoint_list must not be empty")
    if all(isinstance(value, (list, tuple)) for value in values):
        return [list(value) for value in values]
    return [values]


def _gamma_kpt_values(kpoint_list) -> list:
    """Validate a gamma/MP mesh and return its six values."""
    groups = _kpt_groups(kpoint_list)
    if len(groups) != 1:
        raise ValueError("gamma/mp KPT accepts a single mesh group")
    values = groups[0]
    if len(values) not in (3, 6):
        raise ValueError(f"gamma/mp KPT expects three or six values, got {len(values)}")
    mesh = []
    for value in values[:3]:
        number = _kpt_number(value, "mesh subdivision")
        if number <= 0 or number != int(number):
            raise ValueError(
                f"KPT mesh subdivisions must be positive integers, got {value!r}"
            )
        mesh.append(int(number))
    shifts = [_kpt_number(value, "mesh shift") for value in values[3:]]
    return mesh + shifts + [0.0] * (3 - len(shifts))


def _explicit_kpt_points(kpoint_list) -> list:
    """Validate explicit k-points and return them with normalized weights."""
    groups = _kpt_groups(kpoint_list)
    widths = {len(group) for group in groups}
    if not widths <= {3, 4}:
        raise ValueError(
            "each explicit k-point needs three coordinates and an optional weight"
        )
    if len(widths) > 1:
        raise ValueError("explicit k-points must either all define a weight or none")
    points = [
        [_kpt_number(value, "coordinate") for value in group[:3]] for group in groups
    ]
    if widths == {4}:
        weights = [_kpt_number(group[3], "weight") for group in groups]
        total = sum(weights)
        if total == 0:
            raise ValueError("KPT weights must not sum to zero")
        weights = [weight / total for weight in weights]
    else:
        weights = [1.0 / len(groups)] * len(groups)
    return [point + [weight] for point, weight in zip(points, weights)]


def _line_kpt_nodes(kpoint_list) -> list:
    """Validate line-mode nodes and return coordinates, counts and comments."""
    groups = _kpt_groups(kpoint_list)
    if len(groups) < 2:
        raise ValueError("a line KPT needs at least two high-symmetry points")
    nodes = []
    for group in groups:
        if len(group) not in (4, 5):
            raise ValueError(
                "each line KPT node needs three coordinates, a point count and an "
                "optional comment"
            )
        coords = [_kpt_number(value, "coordinate") for value in group[:3]]
        count = _kpt_number(group[3], "point count")
        if count <= 0 or count != int(count):
            raise ValueError(
                f"line KPT point counts must be positive integers, got {group[3]!r}"
            )
        comment = None
        if len(group) == 5:
            comment = str(group[4])
            if not comment.startswith(("#", "//")):
                comment = "#" + comment
        nodes.append((coords, int(count), comment))
    return nodes


def FormatKpt(kpoint_list:List = [1,1,1,0,0,0], model="gamma") -> str:
    """Return the content of an ABACUS KPT file.

    Args:
        kpoint_list: Mesh values for gamma/mp, or one group per k-point/node.
        model (str): One of gamma, mp, direct, cartesian, line, line_cartesian.

    Returns:
        str: The KPT file content.

    Raises:
        ValueError: If the model is unknown or the values do not match it.
    """
    name = NormalizeKptModel(model)
    if name in ("gamma", "mp"):
        values = _gamma_kpt_values(kpoint_list)
        body = " ".join(_kpt_text(value) for value in values)
        return f"K_POINTS\n0\n{_KPT_HEADERS[name]}\n{body}\n"
    if name in ("direct", "cartesian"):
        points = _explicit_kpt_points(kpoint_list)
        rows = "".join(
            "%17.11f %17.11f %17.11f %17.11f\n" % tuple(point) for point in points
        )
        return f"K_POINTS\n{len(points)}\n{_KPT_HEADERS[name]}\n{rows}"

    nodes = _line_kpt_nodes(kpoint_list)
    rows = ""
    for coords, count, comment in nodes:
        row = "%17.11f %17.11f %17.11f %4d" % (coords[0], coords[1], coords[2], count)
        if comment:
            row += " " + comment
        rows += row + "\n"
    return f"K_POINTS\n{len(nodes)}\n{_KPT_HEADERS[name]}\n{rows}"


def WriteKpt(kpoint_list:List = [1,1,1,0,0,0],file_name:str = "KPT", model="gamma"):
    """Write an ABACUS KPT file.

    Docs for KPT file: https://abacus.deepmodeling.com/en/latest/advanced/input_files/kpt.html
    Supported models are gamma, mp, direct, cartesian, line and line_cartesian.

    For the gamma/mp model the k-point_list holds three or six mesh values, such
    as ``[2,2,2,0,0,0]``.  For the direct/cartesian model it holds one group of
    three coordinates plus an optional weight per k-point, such as
    ``[[0.0,0.0,0.0,1.0],[0.5,0.0,0.0,1.0]]``.  For the line models it holds one
    group of three coordinates, a point count and an optional comment per node,
    such as ``[[0.0,0.0,0.0,10],[0.5,0.0,0.0,1]]``.

    Args:
        kpoint_list (List): K-point mesh or node list, see above.
        file_name (str): Output file name.
        model (str): KPT model name.

    Raises:
        ValueError: If the model is unknown or the values do not match it.
    """
    with open(file_name, "w") as f1:
        f1.write(FormatKpt(kpoint_list, model))
