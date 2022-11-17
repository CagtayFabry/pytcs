"""Create scope exporter configuration files and export sample file.

The auto generated export settings file can be found at
`%AppData%\Beckhoff\TcXaeShell Application` .
"""

import itertools
import subprocess
from pathlib import Path
from xml.dom.minidom import parseString

import yaml
from dicttoxml import dicttoxml

_PREFIX = "ExporterSettings"

with open("./csv_properties.yaml", "rt") as f:
    csv = yaml.safe_load(f.read())

options_full = csv["options"]

options_default = {k: v[0] for k, v in options_full.items()}

# default exporter settings
xml = dicttoxml(dict(CSVProperties=options_default), root=False, attr_type=False)
dom = parseString(xml)
with open(f"{_PREFIX}-default.xml", "w") as f:
    f.write(dom.childNodes[0].toprettyxml())

# matrix arrangements
exporter_setups = [
    ["SortChannels", "ExcludeDoubleTimestamp"],
    ["Seperator"],
    ["DecimalMark"],
    ["ContainEOF"],
]

for keys in exporter_setups:
    matrix_options = {k: v for k, v in options_full.items() if k in keys}
    matrix_configs = [
        dict(zip(matrix_options.keys(), v))
        for v in itertools.product(*matrix_options.values())
    ]

    for c in matrix_configs:
        name = f"{_PREFIX}-"
        name += "-".join([f"{k}_{v}" for k, v in c.items()])
        cfg = dict(options_default, **c)
        if cfg == options_default:
            continue
        exporter_settings = {"ExporterSettings": dict(CSVProperties=cfg)}
        xml = dicttoxml(exporter_settings, root=False, attr_type=False)
        dom = parseString(xml)
        with open(f"{name}.xml", "w") as f:
            f.write(dom.childNodes[0].toprettyxml())

# create matrix configurations for channel arrangements
path = Path(__file__).parent.resolve()
file = path / "tc3_scope_3_4_3145_3"
svd = file.with_suffix(".svdx")
exporter = Path(r"C:\TwinCAT\Functions\TF3300-Scope-Server\TC3ScopeExportTool.exe")

config_files = Path(".").glob(f"{_PREFIX}-*.xml")
for config in config_files:
    csv = file.parent.parent / "tests/data" / file.name
    csv = str(csv) + f"{config.stem.removeprefix(_PREFIX)}.csv"
    proc = subprocess.run(
        f'{exporter} svd="{svd}" target="{csv}" config="{config.resolve()}" silent'
    )
